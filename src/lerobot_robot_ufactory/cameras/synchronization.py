"""Host-clock timestamp queues for RGB cameras without a native frame timestamp.

LeRobot camera backends expose ``async_read`` but do not consistently expose
the time at which a returned frame was received.  This adapter is the sole
``async_read`` consumer during recording: it attaches the recorder's monotonic
clock when a fresh frame reaches the host, retains a short history, and selects
only a sample within an explicit time bound.

The timestamp is intentionally called ``capture_monotonic_s`` for a common
sidecar schema with Photon.  For generic USB cameras it means *host receipt*,
not the sensor's exposure time.
"""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import perf_counter, sleep
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimestampedRGBSample:
    """An RGB camera frame and the host time at which it became available."""

    frame: NDArray[Any]
    capture_monotonic_s: float


def select_synchronized_samples(
    sources: dict[str, Any],
    target_monotonic_s: float,
    max_skew_ms: float,
    pair_max_skew_ms: float,
    wait_ms: float,
) -> dict[str, Any]:
    """Select a feasible combination from immutable camera histories.

    All streams share one wait budget. Search time windows, rather than taking
    each stream's nearest frame independently: those minima can lie on opposite
    sides of the state anchor. Keep waiting for alternatives if necessary.
    """
    if not all(np.isfinite(v) and v >= 0 for v in
               (target_monotonic_s, max_skew_ms, pair_max_skew_ms, wait_ms)):
        raise ValueError("camera synchronization values must be finite and non-negative")
    if not sources:
        return {}
    deadline = perf_counter() + wait_ms / 1_000
    bound = max_skew_ms / 1_000
    pair_bound = pair_max_skew_ms / 1_000
    while True:
        histories = {name: source.sync_samples() for name, source in sources.items()}
        candidates = {
            name: tuple(s for s in samples
                        if abs(s.capture_monotonic_s - target_monotonic_s) <= bound)
            for name, samples in histories.items()
        }
        best, best_score = None, None
        for left in sorted({s.capture_monotonic_s for ss in candidates.values() for s in ss}):
            chosen = {}
            for name, samples in candidates.items():
                window = [s for s in samples if left <= s.capture_monotonic_s <= left + pair_bound]
                if not window:
                    break
                chosen[name] = min(window, key=lambda s: abs(s.capture_monotonic_s - target_monotonic_s))
            if len(chosen) == len(sources):
                offsets = [abs(s.capture_monotonic_s - target_monotonic_s) for s in chosen.values()]
                score = (max(offsets), sum(offsets))
                if best_score is None or score < best_score:
                    best, best_score = chosen, score
        remaining = deadline - perf_counter()
        bracketed = all(any(s.capture_monotonic_s >= target_monotonic_s for s in ss)
                        for ss in histories.values())
        if best is not None and (bracketed or remaining <= 0):
            return best
        if remaining <= 0:
            offsets = {name: [round((s.capture_monotonic_s - target_monotonic_s) * 1_000, 3)
                              for s in ss] for name, ss in candidates.items()}
            raise TimeoutError(
                f"No synchronized camera combination within {wait_ms:.3f} ms "
                f"(state limit={max_skew_ms}, pair limit={pair_max_skew_ms} ms; "
                f"candidate offsets ms={offsets})"
            )
        sleep(min(0.001, remaining))


class TimestampedCameraBuffer:
    """Continuously timestamp a camera's fresh asynchronous frames.

    The queue can pair a frame to a robot-state time anchor.  It always waits
    for at least one frame at or after the anchor, then chooses the closest
    queued frame.  If that frame exceeds ``max_skew_ms``, it raises instead of
    returning mismatched data.
    """

    def __init__(
        self,
        camera: Any,
        *,
        history_size: int,
        read_timeout_ms: float = 200.0,
        stop_timeout_ms: float = 2_000.0,
    ) -> None:
        if not isinstance(history_size, int) or isinstance(history_size, bool) or history_size <= 0:
            raise ValueError("history_size must be a positive integer")
        if not np.isfinite(read_timeout_ms) or read_timeout_ms <= 0:
            raise ValueError("read_timeout_ms must be finite and positive")
        if not np.isfinite(stop_timeout_ms) or stop_timeout_ms <= 0:
            raise ValueError("stop_timeout_ms must be finite and positive")
        self.camera = camera
        self._history = deque(maxlen=history_size)
        self._read_timeout_ms = float(read_timeout_ms)
        self._stop_timeout_s = float(stop_timeout_ms) / 1_000
        fps = getattr(camera, "fps", None)
        self._minimum_period_s = 0.0 if not isinstance(fps, (int, float)) or fps <= 0 else 1 / fps
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._error: BaseException | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        with self._condition:
            self._history.clear()
            self._error = None
        self._thread = Thread(
            target=self._capture_loop,
            name=f"timestamped-{self.camera}",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        previous_frame = None
        try:
            while not self._stop.is_set():
                started_at = perf_counter()
                try:
                    frame = self.camera.async_read(timeout_ms=self._read_timeout_ms)
                except TimeoutError:
                    # A transient camera wait timeout is not a capture failure.
                    continue
                received_at = perf_counter()
                # Some cache-only readers return the same array until a new
                # frame arrives. Never give that cached frame a fresh timestamp.
                if frame is previous_frame and frame is not None:
                    self._stop.wait(max(0.001, self._minimum_period_s))
                    continue
                if not isinstance(frame, np.ndarray):
                    raise ValueError(
                        f"Camera {self.camera} returned {type(frame).__name__}, "
                        "expected numpy.ndarray"
                    )
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(
                        f"Camera {self.camera} returned shape {frame.shape}, expected HxWx3"
                    )
                sample = TimestampedRGBSample(frame=frame.copy(), capture_monotonic_s=received_at)
                previous_frame = frame
                with self._condition:
                    self._history.append(sample)
                    self._condition.notify_all()
                # Standard LeRobot async readers wait for a new frame. The
                # period cap also protects CPU use for a backend that returns
                # immediately after producing a frame.
                self._stop.wait(max(0.0, self._minimum_period_s - (perf_counter() - started_at)))
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def sync_samples(self) -> tuple[TimestampedRGBSample, ...]:
        """Return immutable sample references; only the selected frame is copied."""
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"RGB camera capture failed for {self.camera}") from self._error
            if self._stop.is_set():
                raise RuntimeError(f"RGB camera buffer stopped for {self.camera}")
            return tuple(self._history)

    @staticmethod
    def export_sync_sample(sample: TimestampedRGBSample) -> tuple[NDArray[Any], dict]:
        return sample.frame.copy(), {"capture_monotonic_s": sample.capture_monotonic_s}

    @staticmethod
    def _copy_sample(sample: TimestampedRGBSample) -> TimestampedRGBSample:
        return TimestampedRGBSample(
            frame=sample.frame.copy(),
            capture_monotonic_s=sample.capture_monotonic_s,
        )

    def nearest(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float,
    ) -> tuple[NDArray[Any], dict[str, float]]:
        """Return the closest captured RGB frame within the configured budget."""
        if not all(
            np.isfinite(value) and value >= 0
            for value in (target_monotonic_s, max_skew_ms, wait_ms)
        ):
            raise ValueError("camera synchronization values must be finite and non-negative")
        deadline = perf_counter() + wait_ms / 1_000
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(
                        f"RGB camera capture failed for {self.camera}"
                    ) from self._error
                samples = tuple(self._history)
                if samples and any(
                    sample.capture_monotonic_s >= target_monotonic_s for sample in samples
                ):
                    sample = min(
                        samples,
                        key=lambda item: abs(item.capture_monotonic_s - target_monotonic_s),
                    )
                    skew_ms = abs(sample.capture_monotonic_s - target_monotonic_s) * 1_000
                    if skew_ms > max_skew_ms:
                        raise TimeoutError(
                            f"Nearest RGB frame for {self.camera} is {skew_ms:.3f} ms from "
                            f"the synchronization anchor; limit is {max_skew_ms:.3f} ms"
                        )
                    copied = self._copy_sample(sample)
                    return copied.frame, {
                        "capture_monotonic_s": copied.capture_monotonic_s,
                        "sync_target_monotonic_s": float(target_monotonic_s),
                        "sync_offset_ms": float(skew_ms),
                    }
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No RGB frame at or after the synchronization anchor for {self.camera} "
                        f"within {wait_ms:.3f} ms"
                    )
                self._condition.wait(remaining)

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(self._stop_timeout_s)
            if self._thread.is_alive():
                raise TimeoutError(f"RGB camera read is blocked for {self.camera}")
        self._thread = None
        with self._condition:
            self._history.clear()
