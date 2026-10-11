"""Host-clock timestamp queues for ordinary RGB cameras.

The adapter timestamps fresh ``async_read`` frames when they reach the host,
retains a short history, and selects frames on the monotonic action clock.
"""

from collections import deque
from dataclasses import dataclass, field, replace
from threading import Condition, Event, Thread
from time import perf_counter, perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimestampedRGBSample:
    """An RGB frame with its host receipt time and timing diagnostics."""

    frame: NDArray[Any]
    capture_monotonic_s: float
    capture_monotonic_ns: int | None = None
    timing: dict[str, Any] = field(default_factory=dict)


def select_synchronized_samples(
    sources: dict[str, Any],
    target_monotonic_s: float,
    max_skew_ms: float,
    pair_max_skew_ms: float,
    wait_ms: float,
) -> dict[str, Any]:
    """Select the latest causal sample from every ordinary RGB stream.

    ``max_skew_ms`` is the maximum age relative to the target and
    ``pair_max_skew_ms`` bounds the spread among selected host receipt timestamps.
    Selection uses only frames already in each stream's history and never
    waits for a post-target frame. ``wait_ms`` is retained for API compatibility.
    """
    if not all(
        np.isfinite(v) and v >= 0
        for v in (target_monotonic_s, max_skew_ms, pair_max_skew_ms, wait_ms)
    ):
        raise ValueError("camera synchronization values must be finite and non-negative")
    if not sources:
        return {}
    bound = max_skew_ms / 1_000
    pair_bound = pair_max_skew_ms / 1_000
    histories = {name: source.sync_samples() for name, source in sources.items()}
    chosen = {}
    ages_ms = {}
    for name, samples in histories.items():
        causal = tuple(
            sample for sample in samples if sample.capture_monotonic_s <= target_monotonic_s
        )
        if not causal:
            raise TimeoutError(
                f"No causal RGB frame for {name} at or before {target_monotonic_s:.9f}"
            )
        sample = max(causal, key=lambda item: item.capture_monotonic_s)
        age_s = target_monotonic_s - sample.capture_monotonic_s
        if age_s > bound:
            raise TimeoutError(
                f"Latest causal RGB frame for {name} is {age_s * 1_000:.3f} ms old; "
                f"limit is {max_skew_ms:.3f} ms"
            )
        chosen[name] = sample
        ages_ms[name] = age_s * 1_000

    capture_times = [sample.capture_monotonic_s for sample in chosen.values()]
    pair_skew_s = max(capture_times) - min(capture_times)
    if pair_skew_s > pair_bound:
        raise TimeoutError(
            "Causal RGB frame pair skew is "
            f"{pair_skew_s * 1_000:.3f} ms; limit is "
            f"{pair_max_skew_ms:.3f} ms (ages ms={ages_ms})"
        )
    return chosen


class TimestampedCameraBuffer:
    """Continuously timestamp a camera's fresh asynchronous frames.

    The queue pairs a frame to a host monotonic action anchor. Only a frame at
    or before the anchor is eligible; stale frames are rejected.
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
        self._newest_evicted_s = None
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
            self._newest_evicted_s = None
            self._error = None
        self._thread = Thread(
            target=self._capture_loop,
            name=f"timestamped-{self.camera}",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        previous_frame = None
        previous_capture_s = None
        try:
            while not self._stop.is_set():
                started_at = perf_counter()
                try:
                    frame = self.camera.async_read(timeout_ms=self._read_timeout_ms)
                except TimeoutError:
                    # A transient camera wait timeout is not a capture failure.
                    continue
                received_at_ns = perf_counter_ns()
                received_at = received_at_ns / 1_000_000_000
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
                sample = TimestampedRGBSample(
                    frame=frame.copy(),
                    capture_monotonic_s=received_at,
                    capture_monotonic_ns=received_at_ns,
                    timing={
                        "timestamp_source": "host_receipt",
                        "received_monotonic_s": received_at,
                        "received_monotonic_ns": received_at_ns,
                    },
                )
                if not np.isfinite(sample.capture_monotonic_s) or (
                    previous_capture_s is not None
                    and sample.capture_monotonic_s <= previous_capture_s
                ):
                    # Ordered interval boundaries require a monotonic clock.
                    raise RuntimeError("RGB capture clock moved backwards or repeated")
                previous_capture_s = sample.capture_monotonic_s
                previous_frame = frame
                with self._condition:
                    if len(self._history) == self._history.maxlen:
                        self._newest_evicted_s = self._history[0].capture_monotonic_s
                    self._history.append(sample)
                    self._condition.notify_all()
                # Standard LeRobot async readers wait for a new frame. The
                # period cap also protects CPU use for a backend that returns
                # immediately after producing a frame.
                self._stop.wait(
                    max(0.0, self._minimum_period_s - (perf_counter() - started_at))
                )
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

    def samples_between(self, start_monotonic_s, end_monotonic_s, wait_ms=0.0):
        """Return every fresh RGB frame in (start, end], rejecting lost history."""
        if not all(np.isfinite(v) for v in (start_monotonic_s, end_monotonic_s, wait_ms)):
            raise ValueError("Camera interval bounds must be finite")
        if end_monotonic_s < start_monotonic_s or wait_ms < 0:
            raise ValueError("Invalid camera interval end or wait")
        deadline = perf_counter() + wait_ms / 1_000
        with self._condition:
            if wait_ms > 0:
                self.wait_until_after(end_monotonic_s, deadline)
            samples = self.sync_samples()
            if self._newest_evicted_s is not None and self._newest_evicted_s > start_monotonic_s:
                raise RuntimeError(
                    "RGB history no longer covers camera interval; increase sync_history_size"
                )
            return tuple(
                self._copy_sample(sample)
                for sample in samples
                if start_monotonic_s < sample.capture_monotonic_s <= end_monotonic_s
            )

    @staticmethod
    def export_sync_sample(sample: TimestampedRGBSample) -> tuple[NDArray[Any], dict]:
        return sample.frame.copy(), {
            **sample.timing,
            "capture_monotonic_s": sample.capture_monotonic_s,
            "capture_monotonic_ns": sample.capture_monotonic_ns,
        }

    @staticmethod
    def _copy_sample(sample: TimestampedRGBSample) -> TimestampedRGBSample:
        return replace(sample, frame=sample.frame.copy(), timing=sample.timing.copy())

    def wait_until_after(self, target_monotonic_s: float, deadline: float) -> None:
        """Wait until the ordered capture stream crosses the action boundary."""
        with self._condition:
            while True:
                samples = self.sync_samples()  # Surface failures even with a watermark.
                if samples and samples[-1].capture_monotonic_s > target_monotonic_s:
                    return
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError("RGB stream did not advance past camera interval end")
                self._condition.wait(remaining)

    def latest_before(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float = 0.0,
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Return the latest frame at/before ``target_monotonic_s``.

        Use existing history immediately; ``wait_ms`` is kept for compatibility.
        """
        if not all(
            np.isfinite(value) and value >= 0
            for value in (target_monotonic_s, max_skew_ms, wait_ms)
        ):
            raise ValueError("camera synchronization values must be finite and non-negative")
        with self._condition:
            self.sync_samples()
            samples = tuple(
                sample
                for sample in self._history
                if sample.capture_monotonic_s <= target_monotonic_s
            )
            if not samples:
                raise TimeoutError(
                    f"No causal RGB frame at or before the synchronization anchor for {self.camera}"
                )
            sample = max(samples, key=lambda item: item.capture_monotonic_s)
            age_ms = (target_monotonic_s - sample.capture_monotonic_s) * 1_000
            if age_ms > max_skew_ms:
                raise TimeoutError(
                    f"Latest causal RGB frame for {self.camera} is {age_ms:.3f} ms old; "
                    f"limit is {max_skew_ms:.3f} ms"
                )
            copied = self._copy_sample(sample)
            return copied.frame, {
                **copied.timing,
                "capture_monotonic_s": copied.capture_monotonic_s,
                "capture_monotonic_ns": copied.capture_monotonic_ns,
                "sync_target_monotonic_s": float(target_monotonic_s),
                "sync_offset_ms": float(age_ms),
                "sync_signed_offset_ms": float(-age_ms),
            }

    def nearest(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float,
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Backward-compatible alias with causal ``latest_before`` semantics."""
        return self.latest_before(target_monotonic_s, max_skew_ms, wait_ms)

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
