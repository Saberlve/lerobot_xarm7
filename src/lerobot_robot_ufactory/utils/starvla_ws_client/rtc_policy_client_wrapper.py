# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Generic client-side scheduler for real-time action chunking (RTC).

The wrapper exposes one action per :meth:`get_action` call. Its RTC clock is a
*control-step* clock, not a wall clock: ``absolute_control_step`` advances
exactly once after an action is returned and never advances while inference is
blocking.

For action horizon ``H``, rolling stride ``s`` (``execution_horizon``), and
predicted inference delay ``d`` (``inference_delay``), requests are anchored to
absolute steps ``s, 2s, 3s, ...``. A request made at step ``T`` creates a new
chunk whose item zero is also anchored at ``T``. If its response is collected
at step ``T + e``, the wrapper immediately switches to item ``e``. Thus an
early, expected, or late result uses actual elapsed control steps rather than
blindly using ``d``.

Only one inference may be in flight. A late response does not block execution
while the active chunk still covers the current absolute step. Blocking is a
last-target-hold fallback used only when that chunk is genuinely exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTCRequestContext:
    """Immutable timing/ownership information bound to one RTC request."""

    request_id: int
    generation: int
    observation: Dict[str, Any] = field(repr=False, compare=False)
    request_control_step: int = 0
    new_chunk_origin_step: int = 0
    prev_chunk_offset: int = 0
    active_chunk_origin_step: int = 0
    active_chunk_length: int = 0
    request_wall_time: float = 0.0


@dataclass
class RTCRequestDiagnostic:
    """Lightweight timing record for a completed or skipped RTC request."""

    request_control_step: int
    response_control_step: Optional[int]
    request_wall_time: float
    response_wall_time: Optional[float]
    latency_ms: Optional[float]
    elapsed_control_steps: Optional[int]
    predicted_inference_delay: int
    new_chunk_origin_step: int
    splice_index: Optional[int]
    used_old_steps_after_request: Optional[int]
    buffer_remaining_at_response: Optional[int]
    blocked_ms: float
    stale_response: bool
    request_skipped: bool
    request_id: Optional[int] = None
    ownership_mismatch: bool = False


@dataclass
class _PendingRequest:
    context: RTCRequestContext
    thread: threading.Thread
    box: Dict[str, Any]


class RTCPolicyClientWrapper:
    """Turn a chunk policy client into an absolute-time per-step RTC source.

    Args:
        client: exposes ``predict_action`` and, for RTC, also
            ``predict_action_realtime``.
        inference_delay: predicted latency prefix ``d`` in control steps. A
            value of zero disables RTC and retains the classic blocking loop.
        execution_horizon: rolling stride ``s``. It is required when RTC is
            enabled and is forwarded to the server/model; it is not model
            horizon ``H``. ``H`` is learned from the returned model chunk.
        **rtc_kwargs: model RTC options such as ``mode``,
            ``prefix_attention_schedule``, and ``max_guidance_weight``.

    A valid RTC configuration satisfies ``0 < d <= s <= H-d`` and ``s < H``.
    The ``H``-dependent part is checked immediately after the initial blocking
    model response, before any action is returned.
    """

    def __init__(
        self,
        client: Any,
        inference_delay: int,
        execution_horizon: Optional[int] = None,
        **rtc_kwargs,
    ) -> None:
        if inference_delay < 0:
            raise ValueError(f"inference_delay must be >= 0, got {inference_delay}")
        if execution_horizon is not None and int(execution_horizon) <= 0:
            raise ValueError(
                f"execution_horizon (rolling stride s) must be > 0, got {execution_horizon}"
            )

        self._client = client
        self._delay = int(inference_delay)
        self._stride = (
            int(execution_horizon) if execution_horizon is not None else None
        )
        self._rtc_kwargs = dict(rtc_kwargs)
        self._rtc_enabled = self._delay > 0 and callable(
            getattr(client, "predict_action_realtime", None)
        )

        if self._rtc_enabled and self._stride is None:
            raise ValueError(
                "execution_horizon (rolling stride s) is required when RTC is enabled"
            )
        if self._rtc_enabled and self._delay > self._stride:
            raise ValueError(
                "Kinetix-style RTC requires inference_delay <= execution_horizon, "
                f"got d={self._delay}, s={self._stride}"
            )

        self._generation = 0
        self._next_request_id = 0
        self._clear_episode_state()

        if not self._rtc_enabled:
            logger.warning(
                "RTCPolicyClientWrapper: client has no predict_action_realtime or "
                "inference_delay=0; using blocking chunk inference."
            )

    # ------------------------------------------------------------------
    # Public API and diagnostics
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """End an episode and prevent an old response entering the new one.

        The underlying websocket connection is request/response serialized, so
        an outstanding worker is allowed to finish before sending the server
        reset. Its result is discarded and never activated.
        """
        self._generation += 1
        pending = self._pending
        if pending is not None:
            pending.thread.join()
        self._clear_episode_state()
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    @property
    def absolute_control_step(self) -> int:
        """Absolute index of the next action that :meth:`get_action` returns."""
        return self._absolute_control_step

    @property
    def active_chunk_origin_step(self) -> Optional[int]:
        """Absolute step represented by ``active_chunk[0]``."""
        return self._active_chunk_origin_step

    @property
    def current_step_in_chunk(self) -> int:
        """Chunk-local index returned by the most recent ``get_action`` call."""
        return self._last_returned_chunk_index

    @property
    def pending_request_context(self) -> Optional[RTCRequestContext]:
        return self._pending.context if self._pending is not None else None

    @property
    def last_request_diagnostic(self) -> Optional[RTCRequestDiagnostic]:
        return self.request_diagnostics[-1] if self.request_diagnostics else None

    def get_action(self, query_info: Dict) -> np.ndarray:
        """Return exactly one action for the current absolute control step.

        ``query_info`` must contain the fresh observation for this step. It is
        bound to a request context only when this step is a rolling request
        origin. Returning from this method represents sending one target; only
        then is the absolute control step incremented.
        """
        if self._chunk is None:
            # Cold start is deliberately plain/blocking. Besides avoiding a
            # meaningless previous-prefix request, this seeds the server cache.
            self._blocking_fresh_infer(query_info, self._absolute_control_step)

        if self._rtc_enabled:
            # A result that became ready since the previous control tick can be
            # activated before deciding whether this tick is a request origin.
            self._collect_pending(block=False)
            self._schedule_due_request(query_info)
            # Strategy A: an unusually fast result may be used immediately at
            # new_chunk[actual elapsed], including index zero.
            self._collect_pending(block=False)

            if not self._active_chunk_covers(self._absolute_control_step):
                self._recover_from_buffer_exhaustion(query_info)
        else:
            self._maybe_advance_blocking_mode(query_info)

        if not self._active_chunk_covers(self._absolute_control_step):
            raise RuntimeError(
                "RTC client has no action covering absolute control step "
                f"{self._absolute_control_step}"
            )

        local_index = (
            self._absolute_control_step - int(self._active_chunk_origin_step)
        )
        action = self._chunk[local_index]
        self._last_action = action
        self._last_returned_chunk_index = local_index
        self._absolute_control_step += 1
        return action

    # ------------------------------------------------------------------
    # State and validation
    # ------------------------------------------------------------------
    def _clear_episode_state(self) -> None:
        self._chunk: Optional[np.ndarray] = None
        self._active_chunk_origin_step: Optional[int] = None
        self._absolute_control_step = 0
        self._last_returned_chunk_index = -1
        self._last_action: Optional[np.ndarray] = None
        self._action_horizon: Optional[int] = None
        self._next_request_step: Optional[int] = None
        self._pending: Optional[_PendingRequest] = None

        self.chunk_index = -1
        self.last_inference_ms: Optional[float] = None
        self.last_used_prefix = False
        self.last_issued_request_context: Optional[RTCRequestContext] = None
        self.request_diagnostics: List[RTCRequestDiagnostic] = []
        self.rtc_buffer_exhaustion_count = 0
        self.rtc_blocking_time_ms = 0.0
        self.rtc_stale_response_count = 0
        self.rtc_request_skipped_due_to_inflight = 0
        self.rtc_response_ownership_mismatch_count = 0

    def _validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or len(chunk) == 0:
            raise RuntimeError(
                "RTC client expected a non-empty action chunk shaped [H, D], "
                f"got {chunk.shape}"
            )

        horizon = int(len(chunk))
        if self._action_horizon is None:
            self._action_horizon = horizon
            if self._rtc_enabled:
                self._validate_rtc_timing(horizon)
                self._next_request_step = self._absolute_control_step + int(
                    self._stride
                )
        elif horizon != self._action_horizon:
            raise RuntimeError(
                "Model action horizon changed within an episode: "
                f"expected H={self._action_horizon}, got H={horizon}"
            )
        return chunk

    def _validate_rtc_timing(self, horizon: int) -> None:
        d = self._delay
        s = int(self._stride)
        if d <= 0:
            raise ValueError(f"RTC requires inference_delay d > 0, got d={d}")
        if d > s:
            raise ValueError(f"RTC requires d <= s, got d={d}, s={s}")
        if s > horizon - d:
            raise ValueError(
                "RTC requires s <= H-d, "
                f"got H={horizon}, d={d}, s={s}, H-d={horizon - d}"
            )
        if s >= horizon:
            raise ValueError(f"RTC requires s < H, got H={horizon}, s={s}")

    def _active_chunk_covers(self, absolute_step: int) -> bool:
        if self._chunk is None or self._active_chunk_origin_step is None:
            return False
        local_index = absolute_step - self._active_chunk_origin_step
        return 0 <= local_index < len(self._chunk)

    # ------------------------------------------------------------------
    # Inference and activation
    # ------------------------------------------------------------------
    def _infer(
        self,
        query_info: Dict,
        prev_chunk_offset: int = 0,
        *,
        realtime: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Perform one blocking policy round-trip.

        The RTC path always adds ``execution_horizon=s`` to the websocket
        payload. ``H`` is intentionally absent: it belongs to the model/server.
        """
        use_realtime = self._rtc_enabled if realtime is None else bool(realtime)
        start = time.perf_counter()
        if use_realtime:
            rtc_kwargs = dict(self._rtc_kwargs)
            rtc_kwargs["prev_chunk_offset"] = int(prev_chunk_offset)
            rtc_kwargs["execution_horizon"] = int(self._stride)
            resp = self._client.predict_action_realtime(
                query_info,
                inference_delay=self._delay,
                **rtc_kwargs,
            )
        else:
            resp = self._client.predict_action(query_info)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if not isinstance(resp, dict):
            raise RuntimeError(f"Policy client returned {type(resp)!r}, expected dict")
        if resp.get("status") == "error" or resp.get("ok") is False:
            error = resp.get("error", {})
            raise RuntimeError(f"Policy inference failed: {error.get('message', error)}")
        data = resp.get("data")
        if not isinstance(data, dict) or "actions" not in data:
            raise RuntimeError("Policy response is missing data.actions")
        rtc_info = data.get("rtc") or {}
        return {
            "chunk": np.asarray(data["actions"][0]),
            "elapsed_ms": elapsed_ms,
            "used_prefix": bool(rtc_info.get("used_prefix", False)),
        }

    def _activate_result(self, result: Dict[str, Any], origin_step: int) -> None:
        chunk = self._validate_chunk(result["chunk"])
        self._chunk = chunk
        self._active_chunk_origin_step = int(origin_step)
        self.last_inference_ms = float(result["elapsed_ms"])
        self.last_used_prefix = bool(result["used_prefix"])
        self.chunk_index += 1

    def _blocking_fresh_infer(self, query_info: Dict, origin_step: int) -> None:
        # Plain inference also replaces the server's cached normalized chunk,
        # which is essential after a stale RTC result is dropped.
        result = self._infer(query_info, realtime=False)
        self._activate_result(result, origin_step=origin_step)

    # ------------------------------------------------------------------
    # Rolling-s scheduling
    # ------------------------------------------------------------------
    def _schedule_due_request(self, query_info: Dict) -> None:
        if self._next_request_step is None:
            return

        while self._next_request_step <= self._absolute_control_step:
            scheduled_step = self._next_request_step
            self._next_request_step += int(self._stride)

            if (
                scheduled_step != self._absolute_control_step
                or self._pending is not None
            ):
                self._record_skipped_request(scheduled_step)
                continue
            self._start_request(query_info, scheduled_step)

    def _start_request(self, query_info: Dict, request_step: int) -> None:
        if self._chunk is None or self._active_chunk_origin_step is None:
            raise RuntimeError("Cannot issue RTC request without an active chunk")
        if self._pending is not None:
            raise RuntimeError("RTC invariant violated: more than one request in flight")

        prev_offset = request_step - self._active_chunk_origin_step
        if prev_offset < 0:
            raise RuntimeError(
                "RTC request precedes its active chunk origin: "
                f"request={request_step}, origin={self._active_chunk_origin_step}"
            )

        context = RTCRequestContext(
            request_id=self._next_request_id,
            generation=self._generation,
            observation=query_info,
            request_control_step=request_step,
            new_chunk_origin_step=request_step,
            prev_chunk_offset=prev_offset,
            active_chunk_origin_step=self._active_chunk_origin_step,
            active_chunk_length=len(self._chunk),
            request_wall_time=time.perf_counter(),
        )
        self._next_request_id += 1
        self.last_issued_request_context = context
        box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                box["result"] = self._infer(
                    context.observation,
                    prev_chunk_offset=context.prev_chunk_offset,
                    realtime=True,
                )
            except Exception as exc:  # surfaced on the control thread
                box["error"] = exc
            finally:
                box["response_wall_time"] = time.perf_counter()

        thread = threading.Thread(
            target=worker,
            name=f"rtc-inference-{context.request_id}",
            daemon=True,
        )
        self._pending = _PendingRequest(context=context, thread=thread, box=box)
        thread.start()

    def _record_skipped_request(self, scheduled_step: int) -> None:
        self.rtc_request_skipped_due_to_inflight += 1
        now = time.perf_counter()
        remaining = None
        if self._chunk is not None and self._active_chunk_origin_step is not None:
            remaining = max(
                0,
                self._active_chunk_origin_step
                + len(self._chunk)
                - self._absolute_control_step,
            )
        self.request_diagnostics.append(
            RTCRequestDiagnostic(
                request_control_step=scheduled_step,
                response_control_step=None,
                request_wall_time=now,
                response_wall_time=None,
                latency_ms=None,
                elapsed_control_steps=None,
                predicted_inference_delay=self._delay,
                new_chunk_origin_step=scheduled_step,
                splice_index=None,
                used_old_steps_after_request=None,
                buffer_remaining_at_response=remaining,
                blocked_ms=0.0,
                stale_response=False,
                request_skipped=True,
            )
        )

    def _collect_pending(self, block: bool) -> bool:
        """Collect one response; return whether it became the active chunk."""
        pending = self._pending
        if pending is None:
            return False
        if not block and pending.thread.is_alive():
            return False

        wait_start = time.perf_counter()
        pending.thread.join()
        blocked_ms = (time.perf_counter() - wait_start) * 1000.0 if block else 0.0
        # Clear before surfacing an error so a failed request cannot be joined a
        # second time on the next control tick.
        self._pending = None

        context = pending.context
        box = pending.box
        response_wall_time = float(box.get("response_wall_time", time.perf_counter()))
        response_step = self._absolute_control_step
        elapsed_steps = response_step - context.new_chunk_origin_step
        buffer_remaining = max(
            0,
            context.active_chunk_origin_step
            + context.active_chunk_length
            - response_step,
        )

        if "error" in box:
            self.request_diagnostics.append(
                RTCRequestDiagnostic(
                    request_control_step=context.request_control_step,
                    response_control_step=response_step,
                    request_wall_time=context.request_wall_time,
                    response_wall_time=response_wall_time,
                    latency_ms=(response_wall_time - context.request_wall_time)
                    * 1000.0,
                    elapsed_control_steps=elapsed_steps,
                    predicted_inference_delay=self._delay,
                    new_chunk_origin_step=context.new_chunk_origin_step,
                    splice_index=None,
                    used_old_steps_after_request=max(elapsed_steps, 0),
                    buffer_remaining_at_response=buffer_remaining,
                    blocked_ms=blocked_ms,
                    stale_response=False,
                    request_skipped=False,
                    request_id=context.request_id,
                )
            )
            raise RuntimeError(
                f"RTC prefetch inference failed: {box['error']}"
            ) from box["error"]

        ownership_mismatch = context.generation != self._generation
        result = box["result"]
        raw_chunk = np.asarray(result["chunk"])
        if raw_chunk.ndim != 2 or len(raw_chunk) == 0:
            raise RuntimeError(
                "RTC client expected a non-empty action chunk shaped [H, D], "
                f"got {raw_chunk.shape}"
            )
        # An old episode's response must not initialize any part of the new
        # episode, including its learned H or next-request schedule.
        chunk = raw_chunk if ownership_mismatch else self._validate_chunk(raw_chunk)
        stale = ownership_mismatch or elapsed_steps < 0 or elapsed_steps >= len(chunk)
        if stale:
            self.rtc_stale_response_count += 1
            if ownership_mismatch:
                self.rtc_response_ownership_mismatch_count += 1
        else:
            result = dict(result)
            result["chunk"] = chunk
            self._activate_result(result, origin_step=context.new_chunk_origin_step)

        self.request_diagnostics.append(
            RTCRequestDiagnostic(
                request_control_step=context.request_control_step,
                response_control_step=response_step,
                request_wall_time=context.request_wall_time,
                response_wall_time=response_wall_time,
                latency_ms=(response_wall_time - context.request_wall_time) * 1000.0,
                elapsed_control_steps=elapsed_steps,
                predicted_inference_delay=self._delay,
                new_chunk_origin_step=context.new_chunk_origin_step,
                splice_index=None if stale else elapsed_steps,
                used_old_steps_after_request=max(elapsed_steps, 0),
                buffer_remaining_at_response=buffer_remaining,
                blocked_ms=blocked_ms,
                stale_response=stale,
                request_skipped=False,
                request_id=context.request_id,
                ownership_mismatch=ownership_mismatch,
            )
        )
        return not stale

    # ------------------------------------------------------------------
    # Fallback paths
    # ------------------------------------------------------------------
    def _recover_from_buffer_exhaustion(self, query_info: Dict) -> None:
        """Hold the last target and wait only after the active chunk expires."""
        self.rtc_buffer_exhaustion_count += 1
        blocking_start = time.perf_counter()
        try:
            if self._pending is not None:
                self._collect_pending(block=True)
            if not self._active_chunk_covers(self._absolute_control_step):
                self._blocking_fresh_infer(
                    query_info, origin_step=self._absolute_control_step
                )
        finally:
            # Wall-clock blocking is diagnostic only. It must never be turned
            # into synthetic action/control steps.
            self.rtc_blocking_time_ms += (
                time.perf_counter() - blocking_start
            ) * 1000.0

    def _maybe_advance_blocking_mode(self, query_info: Dict) -> None:
        if self._chunk is None or self._active_chunk_origin_step is None:
            return
        local_index = self._absolute_control_step - self._active_chunk_origin_step
        execute_length = len(self._chunk)
        if self._stride is not None:
            # Preserve the pre-RTC classic behavior: an optional execution
            # horizon limits how much of each plain chunk is consumed.
            execute_length = min(execute_length, self._stride)
        if local_index >= execute_length:
            self._blocking_fresh_infer(
                query_info, origin_step=self._absolute_control_step
            )
