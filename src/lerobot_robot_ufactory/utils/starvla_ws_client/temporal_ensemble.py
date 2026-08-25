# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Temporal ensembling (TE) client wrapper for chunked action policies.

Implements the exponential-weighting scheme of ACT's Algorithm 2
(https://huggingface.co/papers/2304.13705), mirroring lerobot's
``ACTTemporalEnsembler`` (``lerobot/policies/act/modeling_act.py``):
each absolute timestep averages the predictions of every chunk covering it,
where a chunk's contribution to a timestep is weighted by
``w_i = exp(-coeff * i)`` with ``i`` the number of chunks already merged
into that timestep (positive coeff favors OLDER chunks; 0 = uniform;
negative favors newer chunks; ACT's default is 0.01).

lerobot's ensembler is not reused directly because it hardcodes the
"query the policy every step, advance the window by 1" loop (it requires
``n_action_steps=1``) and is a torch API. Here inference runs on a
background thread every ``query_period`` steps, so chunks arrive late and
at arbitrary offsets; the same weighting math is therefore re-implemented
in numpy with absolute-timestep alignment.

Usage sketch::

    te = TemporalEnsembleClientWrapper(client, query_period=10, action_horizon=40)
    for episode in episodes:
        te.reset()
        while not done:
            query = {"examples": [{"image": imgs, "lang": task, "state": state}]}
            action = te.get_action(query)   # fresh observation every call
            robot.send_action(action)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemporalEnsembler:
    """Online exponentially-weighted average of action chunks, aligned by
    absolute timestep.

    ``insert(chunk, start_step)`` declares that ``chunk[j]`` is the
    prediction for absolute step ``start_step + j``. ``get_action(t)``
    returns the weighted average of everything predicted for step ``t``.
    Per timestep, the k-th merged chunk gets weight ``exp(-coeff * k)``
    (same online scheme as lerobot's ``ACTTemporalEnsembler``; the
    sum/accumulator form used here is mathematically identical to its
    multiply-add update).
    """

    def __init__(
        self,
        action_horizon: int,
        coeff: float = 0.01,
        ensemble_gripper: bool = True,
    ) -> None:
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        self.horizon = int(action_horizon)
        self.coeff = float(coeff)
        # When False, the last action dim (gripper) is not averaged; the
        # newest chunk's value is used instead (averaging a 0/1-style
        # gripper command produces intermediate openings).
        self.ensemble_gripper = bool(ensemble_gripper)
        self.reset()

    def reset(self) -> None:
        self._off: Optional[int] = None  # absolute step of buffer index 0
        self._floor: int = 0  # steps below this are consumed/dead
        self._wsum: Optional[np.ndarray] = None  # (L, D) weighted sums
        self._wtot: Optional[np.ndarray] = None  # (L,) weight sums
        self._count: Optional[np.ndarray] = None  # (L,) chunks merged per step
        self._grip_latest: Optional[np.ndarray] = None  # (L,) newest gripper value

    # ------------------------------------------------------------------
    def insert(self, chunk: np.ndarray, start_step: int) -> None:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"chunk must be (T, D), got shape {chunk.shape}")
        T, D = chunk.shape
        if self._off is None:
            self._off = int(start_step)
            self._wsum = np.zeros((2 * self.horizon, D))
            self._wtot = np.zeros(2 * self.horizon)
            self._count = np.zeros(2 * self.horizon)
            self._grip_latest = np.zeros(2 * self.horizon)
        lo = int(start_step) - self._off
        hi = lo + T
        self._ensure_capacity(hi)
        # Drop predictions for already-consumed steps (late chunk head).
        skip = max(0, self._floor - int(start_step))
        if skip >= T:
            return
        lo += skip
        chunk = chunk[skip:]
        counts = self._count[lo:hi]
        w = np.exp(-self.coeff * counts)
        self._wsum[lo:hi] += w[:, None] * chunk
        self._wtot[lo:hi] += w
        self._count[lo:hi] = counts + 1
        # Inserts arrive in issue order (the wrapper allows one pending
        # inference at a time), so overwriting keeps the newest value.
        self._grip_latest[lo:hi] = chunk[:, -1]

    def get_action(self, t: int) -> np.ndarray:
        if self._off is None:
            raise RuntimeError("TemporalEnsembler is empty.")
        idx = int(t) - self._off
        if idx < 0 or idx >= len(self._count) or self._count[idx] == 0:
            raise RuntimeError(f"No prediction covers absolute step {t}.")
        action = self._wsum[idx] / self._wtot[idx]
        if not self.ensemble_gripper:
            action = action.copy()
            action[-1] = self._grip_latest[idx]
        self._floor = int(t) + 1
        # Compact once the consumed prefix is larger than one horizon so
        # buffer indices stay bounded over a long episode.
        if idx > self.horizon:
            self._wsum = self._wsum[idx:]
            self._wtot = self._wtot[idx:]
            self._count = self._count[idx:]
            self._grip_latest = self._grip_latest[idx:]
            self._off += idx
        return action.astype(np.float32)

    # ------------------------------------------------------------------
    def _ensure_capacity(self, hi: int) -> None:
        if hi <= len(self._count):
            return
        grow = hi - len(self._count)
        self._wsum = np.concatenate([self._wsum, np.zeros((grow, self._wsum.shape[1]))])
        self._wtot = np.concatenate([self._wtot, np.zeros(grow)])
        self._count = np.concatenate([self._count, np.zeros(grow)])
        self._grip_latest = np.concatenate([self._grip_latest, np.zeros(grow)])


class TemporalEnsembleClientWrapper:
    """Per-timestep action source: async periodic inference + temporal ensemble.

    Args:
        client: policy client with ``predict_action(query_info)`` (e.g.
            :class:`WebsocketClientPolicy`).
        query_period: issue a new inference every this many executed steps.
            With horizon H and period P, ~H/P chunks overlap in the ensemble.
        action_horizon: length T of the chunks returned by the server.
        coeff: temporal-ensemble coefficient (see :class:`TemporalEnsembler`).
        ensemble_gripper: average the gripper dim too (True, like lerobot)
            or always take the newest chunk's gripper value (False).

    Diagnostics: :attr:`last_inference_ms` and :attr:`chunk_index`
    (increments on every accepted inference result; reset to -1 by
    :meth:`reset`).
    """

    def __init__(
        self,
        client: Any,
        query_period: int,
        action_horizon: int,
        coeff: float = 0.01,
        ensemble_gripper: bool = True,
    ) -> None:
        if query_period < 1:
            raise ValueError(f"query_period must be >= 1, got {query_period}")
        self._client = client
        self._period = int(query_period)
        self._horizon = int(action_horizon)
        self._ensembler = TemporalEnsembler(action_horizon, coeff, ensemble_gripper)
        self.last_inference_ms: Optional[float] = None
        self._clear_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Episode boundary: drop the ensemble, pending inference, and
        server-side policy state."""
        self._clear_state()
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    def get_action(self, query_info: Dict) -> np.ndarray:
        """Return the ensembled action for the current timestep.

        ``query_info`` must be built from a FRESH observation on every call:
        it is used as-is when a (background) inference fires at this timestep.
        """
        self._collect_pending(block=False)
        if self._t == 0:
            self._blocking_infer(query_info, start_step=0)  # first chunk: blocking
        else:
            self._maybe_start_prefetch(query_info)
        try:
            action = self._ensembler.get_action(self._t)
        except RuntimeError:
            # Coverage gap: wait for the in-flight inference first (the
            # websocket is not thread-safe, never call predict_action while
            # a prefetch is running), then fall back to a blocking inference.
            self._collect_pending(block=True)
            try:
                action = self._ensembler.get_action(self._t)
            except RuntimeError:
                self._blocking_infer(query_info, start_step=self._t)
                action = self._ensembler.get_action(self._t)
        self._t += 1
        return action

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clear_state(self) -> None:
        self._t: int = 0
        self._last_issue_step: Optional[int] = None
        self.chunk_index: int = -1  # increments on every accepted inference result
        # [thread, result_box, issue_step]; box gets "result"/"error"
        self._pending: Optional[list] = None
        self._ensembler.reset()

    def _infer(self, query_info: Dict) -> tuple[np.ndarray, float]:
        """One inference round-trip (blocking)."""
        start = time.perf_counter()
        resp = self._client.predict_action(query_info)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return np.asarray(resp["data"]["actions"][0]), elapsed_ms

    def _accept(self, chunk: np.ndarray, start_step: int, elapsed_ms: float) -> None:
        if self._t - start_step >= self._horizon:
            logger.warning(
                "TE: inference took %d steps (>= horizon %d); dropping stale chunk.",
                self._t - start_step,
                self._horizon,
            )
            return
        self._ensembler.insert(chunk, start_step)
        self.last_inference_ms = elapsed_ms
        self.chunk_index += 1

    def _blocking_infer(self, query_info: Dict, start_step: int) -> None:
        chunk, elapsed_ms = self._infer(query_info)
        self._accept(chunk, start_step, elapsed_ms)
        self._last_issue_step = start_step

    def _maybe_start_prefetch(self, query_info: Dict) -> None:
        if self._pending is not None:
            return
        if (
            self._last_issue_step is not None
            and self._t - self._last_issue_step < self._period
        ):
            return
        box: Dict[str, Any] = {}

        def worker():
            try:
                box["result"] = self._infer(query_info)
            except Exception as e:  # surfaced on the main thread at collect time
                box["error"] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._pending = [thread, box, self._t]
        self._last_issue_step = self._t

    def _collect_pending(self, block: bool) -> None:
        if self._pending is None:
            return
        thread, box, issue_step = self._pending
        if not block and thread.is_alive():
            return
        thread.join()
        self._pending = None
        if "error" in box:
            raise RuntimeError(f"TE prefetch inference failed: {box['error']}") from box["error"]
        chunk, elapsed_ms = box["result"]
        self._accept(chunk, issue_step, elapsed_ms)
