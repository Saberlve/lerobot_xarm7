# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Generic RTC (real-time chunking) client wrapper.

Wraps any policy client that exposes ``predict_action(query_info)`` and
``predict_action_realtime(query_info, inference_delay, **kwargs)`` (e.g.
:class:`WebsocketClientPolicy` against a ``server_policy.py`` instance with
``rtc_supported=True``) and turns it into a per-timestep ``get_action``
interface that overlaps inference with execution:

- the robot executes the current action chunk one step per ``get_action`` call;
- ``inference_delay`` steps before the chunk exhausts, the next inference is
  issued on a background thread using the observation at that moment, while
  the main thread keeps executing the remaining steps;
- when the response arrives, the new chunk is spliced in at the aligned offset
  (its first ``inference_delay`` steps are pinned by the server to the old
  chunk's tail, so any splice offset within that region is seamless).

This is robot-agnostic: the caller only has to build ``query_info`` (the same
dict ``predict_action`` takes) from a fresh observation on every call and send
the returned action to the robot. Works with any framework the server exposes
RTC for (PI0/PI05 via prefix pinning, QwenPI via ΠGDM, QwenDiscreteDiffusion
via MaskGIT prefix decode); with RTC unsupported servers it degrades to the
classic blocking "infer every N steps" loop.

Usage sketch::

    rtc = RTCPolicyClientWrapper(WebsocketClientPolicy(host, port), inference_delay=8)
    for episode in episodes:
        rtc.reset()
        while not done:
            query = {"examples": [{"image": imgs, "lang": task, "state": state}]}
            action = rtc.get_action(query)   # fresh observation every call
            robot.send_action(action)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RTCPolicyClientWrapper:
    """Per-timestep action source with background RTC prefetch.

    Args:
        client: policy client with ``predict_action(query_info)``; RTC is used
            when it also has ``predict_action_realtime``.
        inference_delay: d — steps the robot keeps executing from the old chunk
            while the next inference runs. Set it to
            ``ceil(measured_inference_seconds * control_fps)``. ``0`` disables
            prefetching (pure blocking loop).
        execution_horizon: execute at most this many steps of each chunk before
            re-inferring; ``None`` = the full chunk.
        **rtc_kwargs: forwarded to ``client.predict_action_realtime``. ``mode``
            is accepted by every RTC-capable framework (PI0/PI05:
            ``"prefix_pin"``; QwenPI: ``"pigdm"`` / ``"simulated_delay"``) —
            pass it explicitly for clarity; other knobs are framework-specific.

    Diagnostics: :attr:`last_inference_ms`, :attr:`last_used_prefix`, and
    :attr:`chunk_index` (increments on every accepted inference result; reset
    to -1 by :meth:`reset`).
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
        self._client = client
        self._delay = int(inference_delay)
        self._exec_horizon = execution_horizon
        self._rtc_kwargs = dict(rtc_kwargs)
        self._rtc_enabled = self._delay > 0 and callable(
            getattr(client, "predict_action_realtime", None)
        )
        self.last_inference_ms: Optional[float] = None
        self.last_used_prefix: bool = False
        self._clear_state()
        if not self._rtc_enabled:
            logger.warning(
                "RTCPolicyClientWrapper: client has no predict_action_realtime or "
                "inference_delay=0 — falling back to blocking chunk inference."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Episode boundary: drop local chunk state and server-side prev chunk."""
        self._clear_state()
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    @property
    def current_step_in_chunk(self) -> int:
        """Chunk-local index of the action returned by the last ``get_action``."""
        return self._step - 1

    def get_action(self, query_info: Dict) -> np.ndarray:
        """Return the action for the current timestep.

        ``query_info`` must be built from a FRESH observation on every call:
        it is used as-is when a (prefetch) inference fires at this timestep.
        """
        if self._chunk is None:
            self._fresh_infer(query_info)  # first chunk of the episode: blocking
        else:
            self._maybe_start_prefetch(query_info)
            if self._step >= self._exec_len():
                self._advance_chunk(query_info)
            else:
                self._maybe_collect()  # early splice if the response is ready
        action = self._chunk[self._step]
        self._step += 1
        return action

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clear_state(self) -> None:
        self._chunk: Optional[np.ndarray] = None  # (T, D), denormalized
        self._step: int = 0
        self.chunk_index: int = -1  # increments on every accepted inference result
        # [thread, result_box, issue_step]; box gets "result"/"error"
        self._pending: Optional[list] = None

    def _exec_len(self) -> int:
        n = len(self._chunk)
        return min(self._exec_horizon, n) if self._exec_horizon else n

    def _infer(self, query_info: Dict, prev_chunk_offset: int = 0) -> Dict:
        """One inference round-trip (blocking).

        ``prev_chunk_offset`` tells the server how many steps of the chunk it
        tracks had already been executed when this request's observation was
        captured, so the prefix is pinned to the old chunk's *tail* (the
        not-yet-executed steps) instead of its head.
        """
        start = time.perf_counter()
        if self._rtc_enabled:
            resp = self._client.predict_action_realtime(
                query_info,
                inference_delay=self._delay,
                prev_chunk_offset=prev_chunk_offset,
                **self._rtc_kwargs,
            )
        else:
            resp = self._client.predict_action(query_info)
        elapsed_ms = (time.perf_counter() - start) * 1000
        data = resp["data"]
        rtc_info = data.get("rtc") or {}
        return {
            "chunk": np.asarray(data["actions"][0]),
            "elapsed_ms": elapsed_ms,
            "used_prefix": bool(rtc_info.get("used_prefix", False)),
        }

    def _apply_result(self, result: Dict, splice_offset: int) -> None:
        self.last_inference_ms = result["elapsed_ms"]
        self.last_used_prefix = result["used_prefix"]
        self._chunk = result["chunk"]
        self._step = int(splice_offset)
        self.chunk_index += 1

    def _fresh_infer(self, query_info: Dict, prev_chunk_offset: int = 0) -> None:
        """Blocking inference used when there is nothing left to execute."""
        self._apply_result(self._infer(query_info, prev_chunk_offset), splice_offset=0)
        if self._step >= self._exec_len():
            raise RuntimeError("RTC client received an empty action chunk.")

    def _maybe_start_prefetch(self, query_info: Dict) -> None:
        if not self._rtc_enabled or self._pending is not None:
            return
        # Fire when exactly `inference_delay` executable steps remain.
        if self._step != self._exec_len() - self._delay:
            return
        box: Dict[str, Any] = {}

        def worker():
            try:
                box["result"] = self._infer(query_info, prev_chunk_offset=self._step)
            except Exception as e:  # surfaced on the main thread at collect time
                box["error"] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._pending = [thread, box, self._step]

    def _collect_pending(self, block: bool) -> bool:
        """Splice a finished prefetch into the current chunk. Returns True if spliced."""
        if self._pending is None:
            return False
        thread, box, issue_step = self._pending
        if not block and thread.is_alive():
            return False
        thread.join()
        self._pending = None
        if "error" in box:
            raise RuntimeError(f"RTC prefetch inference failed: {box['error']}") from box["error"]
        # Alignment: new_chunk[i] corresponds to old_chunk[issue_step + i]
        # (the server pinned the first `inference_delay` steps to the old
        # chunk's tail), so the aligned continuation offset is the number of
        # old-chunk steps executed since the request was issued.
        elapsed = self._step - issue_step
        self._apply_result(box["result"], splice_offset=elapsed)
        return True

    def _maybe_collect(self) -> None:
        self._collect_pending(block=False)

    def _advance_chunk(self, query_info: Dict) -> None:
        """Current chunk exhausted: take the prefetched chunk, else block."""
        if self._collect_pending(block=True) and self._step < self._exec_len():
            return
        # Prefetch was too slow to cover the gap (or was never started):
        # fall back to a blocking inference from the current observation.
        # The robot stalls during this round-trip, so the new chunk still
        # starts at old-chunk index `self._step`.
        self._fresh_infer(query_info, prev_chunk_offset=self._step)
