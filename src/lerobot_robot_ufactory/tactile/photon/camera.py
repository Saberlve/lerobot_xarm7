"""Xense SDK images exposed through LeRobot's camera interface.

Each sensor has its own capture thread. Only that thread reads the SDK;
consumers receive timestamped copies with bounded cache age.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread, RLock
from time import perf_counter, perf_counter_ns
from typing import Any

import cv2
import numpy as np
from lerobot.cameras.configs import ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy.typing import NDArray

from ..base import TactileCamera
from .config import XensePhotonCameraConfig


def _sensor_class():
    # Keep the SDK optional for users of other cameras.
    try:
        from xensesdk import Sensor
    except ImportError as exc:
        raise ImportError('Xense Photon requires xensesdk: pip install -e ".[xense]"') from exc
    return Sensor


@dataclass(frozen=True)
class XensePhotonSample:
    """One SDK sample; every field comes from one selectSensorInfo call."""

    frame_bgr: NDArray[Any]
    marker_motion_3d: NDArray[Any] | None
    sensor_timestamp_s: float | None
    capture_monotonic_s: float
    capture_monotonic_ns: int | None = None


class _TrackedXenseHistory(deque):
    """Bounded history that records actual evictions, including test appends."""

    def __init__(self, maxlen: int):
        super().__init__(maxlen=maxlen)
        self.total_appended = 0
        self.eviction_count = 0
        self.first_seen_timestamp: float | None = None
        self.newest_evicted_timestamp: float | None = None

    def append(self, sample: XensePhotonSample) -> None:
        if self.first_seen_timestamp is None:
            self.first_seen_timestamp = sample.capture_monotonic_s
        if self.maxlen is not None and len(self) == self.maxlen:
            self.eviction_count += 1
            self.newest_evicted_timestamp = self[0].capture_monotonic_s
        super().append(sample)
        self.total_appended += 1

    def extend(self, samples) -> None:
        for sample in samples:
            self.append(sample)

    def clear(self) -> None:
        super().clear()
        self.total_appended = 0
        self.eviction_count = 0
        self.first_seen_timestamp = None
        self.newest_evicted_timestamp = None


class XensePhotonCamera(TactileCamera):
    def __init__(self, config: XensePhotonCameraConfig):
        super().__init__(config)
        self.config = config
        self._sensor = None
        self._solver = None
        self._solver_runtime = None
        self._solver_lock = RLock()
        self._thread = None
        self._condition = Condition()
        self._stop = Event()
        self._frame = None
        self._sample = None
        self._sample_history = _TrackedXenseHistory(self.config.sync_history_size)
        self._frame_time = 0.0
        self._error = None
        # Set by the recorder before connect; export before the reader starts.
        self.runtime_export_dir: Path | None = None

    @property
    def deferred_feature_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"mesh_motion_3d": (self.config.marker_rows, self.config.marker_cols, 3)}

    def runtime_manifest(self) -> dict[str, Any]:
        from importlib.metadata import version

        return {
            "sdk_version": version("xensesdk"),
            "serial_number": self.config.serial_number,
            "runtime_file": f"runtime_{self.config.serial_number}",
            "infer_mode": self.config.infer_mode,
            "disable_infer": self.config.disable_infer,
            "output_type": self.config.output_type,
            "motion_3d_output": self.config.motion_3d_output,
        }

    @contextmanager
    def deferred_session(self, runtime_dir: Path):
        """One solver per sensor/runtime batch, including nested episode calls."""
        runtime = (Path(runtime_dir) / f"runtime_{self.config.serial_number}").resolve()
        with self._solver_lock:
            if self._solver is not None:
                if runtime != self._solver_runtime:
                    raise RuntimeError("Cannot switch runtime inside a deferred session")
                yield self
                return
            sensor_class = _sensor_class()
            solver = sensor_class.createSolver(runtime, overrides={"dev.disable_infer": False})
            if not solver:
                raise RuntimeError(f"Cannot create offline solver: {runtime}")
            self._solver, self._solver_runtime = solver, runtime
            try:
                yield self
            finally:
                self._solver = self._solver_runtime = None
                solver.release()

    def compute_deferred_features(
        self, image_bgr: NDArray[np.uint8], runtime_dir: Path
    ) -> dict[str, NDArray[Any]]:
        with self.deferred_session(runtime_dir):
            flow = np.array(
                self._solver.selectSensorInfo(
                    _sensor_class().OutputType.Mesh3DFlow, rectify_image=image_bgr
                ),
                copy=True,
            )
            expected_shape = self.deferred_feature_shapes["mesh_motion_3d"]
            if (
                flow.shape != expected_shape
                or flow.dtype.kind != "f"
                or not np.isfinite(flow).all()
            ):
                raise RuntimeError(
                    f"Invalid offline Mesh3DFlow for {self.config.serial_number}: "
                    f"expected finite floating array {expected_shape}, got {flow.shape}, {flow.dtype}"
                )
            return {"mesh_motion_3d": flow}

    @property
    def is_connected(self) -> bool:
        return self._sensor is not None

    @property
    def saves_marker_motion_3d(self) -> bool:
        return self.config.save_marker_motion_3d

    @property
    def motion_3d_feature_key(self) -> str:
        return (
            "mesh_motion_3d" if self.config.motion_3d_output == "Mesh3DFlow" else "marker_motion_3d"
        )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return [
            {"serial_number": sn, "camera_id": camera_id, "type": "photon"}
            for sn, camera_id in _sensor_class().scanSerialNumber().items()
        ]

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError()
        sensor_class = _sensor_class()
        output = getattr(sensor_class.OutputType, self.config.output_type)
        kwargs = {"disable_infer": self.config.disable_infer}
        if self.config.config_path is not None:
            kwargs["config_path"] = self.config.config_path
        if self.config.infer_mode is not None:
            kwargs["infer_mode"] = self.config.infer_mode
        self._sensor = sensor_class.create(self.config.serial_number, **kwargs)
        if self._sensor is None:
            raise ConnectionError(f"Xense SDK could not open {self.config.serial_number}")
        self._stop.clear()
        self._frame = None
        self._sample = None
        self._sample_history.clear()
        self._error = None
        self._frame_time = 0.0
        self._thread = Thread(
            target=self._capture_loop,
            args=(output,),
            name=f"xense-{self.config.serial_number}",
            daemon=True,
        )
        try:
            if self.runtime_export_dir is not None:
                self.runtime_export_dir.mkdir(parents=True, exist_ok=True)
                self._sensor.exportRuntimeConfig(str(self.runtime_export_dir))
                runtime = self.runtime_export_dir / f"runtime_{self.config.serial_number}"
                if not runtime.is_file() or runtime.stat().st_size == 0:
                    raise RuntimeError(f"Xense runtime export missing or empty: {runtime}")
            self._thread.start()
            if warmup:
                self.async_read()
        except BaseException:
            if self._thread.ident is None:
                self._thread = None
            self.disconnect()
            raise

    def _capture_loop(self, output) -> None:
        last_sensor_timestamp = None
        clock_offset = None
        last_mapped_monotonic_s = None
        try:
            while not self._stop.is_set():
                started = perf_counter()
                if self.config.save_marker_motion_3d:
                    output_types = self._sensor.OutputType
                    result = self._sensor.selectSensorInfo(
                        output,
                        getattr(output_types, self.config.motion_3d_output),
                        output_types.TimeStamp,
                    )
                    if not isinstance(result, tuple) or len(result) != 3:
                        raise ValueError(
                            f"Xense SDK did not return image, {self.config.motion_3d_output} and TimeStamp"
                        )
                    frame, marker_motion_3d, sensor_timestamp = result
                else:
                    frame, sensor_timestamp = self._sensor.selectSensorInfo(
                        output, self._sensor.OutputType.TimeStamp
                    )
                    marker_motion_3d = None
                captured_at_ns = perf_counter_ns()
                captured_at = captured_at_ns / 1_000_000_000
                if frame is not None:
                    timestamp_values = np.asarray(sensor_timestamp)
                    if timestamp_values.size != 1:
                        raise ValueError("Expected Xense TimeStamp to contain one scalar")
                    sensor_timestamp = float(timestamp_values.reshape(-1)[0])
                    if not np.isfinite(sensor_timestamp):
                        raise ValueError("Xense TimeStamp must be finite")
                    # Xense SDK 2.1 reports Unix seconds. Reject another unit
                    # instead of silently pairing it against perf_counter().
                    if not 1e8 <= sensor_timestamp <= 1e11:
                        raise ValueError(
                            f"Xense TimeStamp is not in Unix seconds: {sensor_timestamp}"
                        )
                    if sensor_timestamp == last_sensor_timestamp:
                        self._stop.wait(max(0, 1 / self.fps - (perf_counter() - started)))
                        continue
                    # The minimum observed receive-minus-sensor offset is the
                    # least transport delay seen so far. Mapping every SDK
                    # timestamp with it removes variable USB/SDK queue delay
                    # while never placing a frame after its host receipt time.
                    current_offset = captured_at - sensor_timestamp
                    if clock_offset is None or current_offset < clock_offset:
                        clock_offset = current_offset
                    mapped_monotonic_s = sensor_timestamp + clock_offset
                    mapped_monotonic_ns = round(mapped_monotonic_s * 1_000_000_000)
                    if (
                        last_mapped_monotonic_s is not None
                        and mapped_monotonic_s <= last_mapped_monotonic_s
                    ):
                        last_sensor_timestamp = sensor_timestamp
                        self._stop.wait(max(0, 1 / self.fps - (perf_counter() - started)))
                        continue
                    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                        raise ValueError("Xense image must be a uint8 numpy array")
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError(f"Expected Xense HxWx3 image, got {frame.shape}")
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    if self.config.save_marker_motion_3d:
                        marker_motion_3d = np.asarray(marker_motion_3d)
                        if (
                            marker_motion_3d.dtype.kind != "f"
                            or not np.isfinite(marker_motion_3d).all()
                        ):
                            raise ValueError("Xense displacement must be a finite floating array")
                        expected_shape = (
                            self.config.marker_rows,
                            self.config.marker_cols,
                            3,
                        )
                        if marker_motion_3d.shape != expected_shape:
                            raise ValueError(
                                f"Expected Xense {self.config.motion_3d_output} shape "
                                f"{expected_shape}, got {marker_motion_3d.shape}"
                            )
                    last_sensor_timestamp = sensor_timestamp
                    last_mapped_monotonic_s = mapped_monotonic_s
                    with self._condition:
                        self._frame = frame.copy()
                        self._sample = XensePhotonSample(
                            frame_bgr=self._frame,
                            marker_motion_3d=(
                                None if marker_motion_3d is None else marker_motion_3d.copy()
                            ),
                            sensor_timestamp_s=sensor_timestamp,
                            capture_monotonic_s=mapped_monotonic_s,
                            capture_monotonic_ns=mapped_monotonic_ns,
                        )
                        self._sample_history.append(self._sample)
                        self._frame_time = captured_at
                        self._condition.notify_all()
                self._stop.wait(max(0, 1 / self.fps - (perf_counter() - started)))
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def _latest_frame(self, timeout_ms: float) -> NDArray[Any]:
        return self._latest_sample(timeout_ms).frame_bgr.copy()

    def sync_samples(self) -> tuple[XensePhotonSample, ...]:
        with self._condition:
            if not self.is_connected or self._stop.is_set():
                raise DeviceNotConnectedError()
            if self._error is not None:
                raise RuntimeError(
                    f"Xense capture failed for {self.config.serial_number}"
                ) from self._error
            return tuple(self._sample_history)

    def export_sync_sample(self, sample: XensePhotonSample) -> tuple[NDArray[Any], dict]:
        sample = self._copy_sample(sample)
        frame = sample.frame_bgr
        if self.config.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timing = {
            "capture_monotonic_s": sample.capture_monotonic_s,
            "capture_monotonic_ns": sample.capture_monotonic_ns,
            "sensor_timestamp_s": sample.sensor_timestamp_s,
            "device_to_host_offset_s": (
                None
                if sample.sensor_timestamp_s is None
                else sample.capture_monotonic_s - sample.sensor_timestamp_s
            ),
        }
        if self.saves_marker_motion_3d:
            timing[self.motion_3d_feature_key] = sample.marker_motion_3d
        return frame, timing

    @staticmethod
    def _copy_sample(sample: XensePhotonSample) -> XensePhotonSample:
        return XensePhotonSample(
            frame_bgr=sample.frame_bgr.copy(),
            marker_motion_3d=(
                None if sample.marker_motion_3d is None else sample.marker_motion_3d.copy()
            ),
            sensor_timestamp_s=sample.sensor_timestamp_s,
            capture_monotonic_s=sample.capture_monotonic_s,
            capture_monotonic_ns=sample.capture_monotonic_ns,
        )

    def history_status(self) -> dict[str, int | float | None]:
        with self._condition:
            oldest = (
                None if not self._sample_history else self._sample_history[0].capture_monotonic_s
            )
            return {
                "history_first_seen_timestamp": self._sample_history.first_seen_timestamp,
                "history_total_appended": self._sample_history.total_appended,
                "history_eviction_count": self._sample_history.eviction_count,
                "oldest_retained_timestamp": oldest,
                "newest_evicted_timestamp": self._sample_history.newest_evicted_timestamp,
            }

    def _latest_sample(self, timeout_ms: float) -> XensePhotonSample:
        if timeout_ms < 0 or not np.isfinite(timeout_ms):
            raise ValueError("timeout_ms must be finite and non-negative")
        deadline = perf_counter() + timeout_ms / 1000
        with self._condition:
            while True:
                if not self.is_connected or self._stop.is_set():
                    raise DeviceNotConnectedError()
                if self._error is not None:
                    raise RuntimeError(
                        f"Xense capture failed for {self.config.serial_number}"
                    ) from self._error
                if (
                    self._frame is not None
                    and self._sample is not None
                    and perf_counter() - self._frame_time <= self.config.max_frame_age_ms / 1000
                ):
                    return self._copy_sample(self._sample)
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"No recent Xense frame from {self.config.serial_number}")
                self._condition.wait(remaining)

    def latest_before_sample(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float = 0.0,
    ) -> tuple[XensePhotonSample, float]:
        """Select the latest buffered Photon sample at/before an anchor.

        The mapped timestamp is in the recorder's ``perf_counter`` domain.
        ``wait_ms`` remains accepted for API compatibility; post-anchor samples
        are never eligible and are not awaited.
        """
        if not all(
            np.isfinite(value) and value >= 0
            for value in (target_monotonic_s, max_skew_ms, wait_ms)
        ):
            raise ValueError("Xense synchronization values must be finite and non-negative")
        with self._condition:
            if not self.is_connected or self._stop.is_set():
                raise DeviceNotConnectedError()
            if self._error is not None:
                raise RuntimeError(
                    f"Xense capture failed for {self.config.serial_number}"
                ) from self._error
            samples = tuple(
                sample
                for sample in self._sample_history
                if sample.capture_monotonic_s <= target_monotonic_s
            )
            if not samples:
                raise TimeoutError(
                    f"No causal Xense frame at or before the synchronization anchor for "
                    f"{self.config.serial_number}"
                )
            sample = max(samples, key=lambda item: item.capture_monotonic_s)
            age_ms = (target_monotonic_s - sample.capture_monotonic_s) * 1000
            if age_ms > max_skew_ms:
                raise TimeoutError(
                    f"Latest causal Xense frame for {self.config.serial_number} is "
                    f"{age_ms:.3f} ms old; limit is {max_skew_ms:.3f} ms"
                )
            return self._copy_sample(sample), age_ms

    def samples_between(
        self,
        start_monotonic_s: float,
        end_monotonic_s: float,
        wait_ms: float = 0.0,
    ) -> tuple[XensePhotonSample, ...]:
        """Return all buffered samples in ``(start_monotonic_s, end_monotonic_s]``.

        A positive ``wait_ms`` waits only for a timestamp beyond ``end`` to
        watermark the ordered stream. That future sample is never returned.
        """
        if not all(np.isfinite(value) for value in (start_monotonic_s, end_monotonic_s, wait_ms)):
            raise ValueError("Xense window bounds must be finite")
        if end_monotonic_s < start_monotonic_s or wait_ms < 0:
            raise ValueError("Xense window end must be at or after its start")
        deadline = perf_counter() + wait_ms / 1_000
        with self._condition:
            while wait_ms > 0 and not any(
                sample.capture_monotonic_s > end_monotonic_s for sample in self._sample_history
            ):
                if not self.is_connected or self._stop.is_set():
                    raise DeviceNotConnectedError()
                if self._error is not None:
                    raise RuntimeError(
                        f"Xense capture failed for {self.config.serial_number}"
                    ) from self._error
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Xense stream {self.config.serial_number} did not advance "
                        f"past tactile window end within {wait_ms:.3f} ms"
                    )
                self._condition.wait(remaining)
            if not self.is_connected or self._stop.is_set():
                raise DeviceNotConnectedError()
            if self._error is not None:
                raise RuntimeError(
                    f"Xense capture failed for {self.config.serial_number}"
                ) from self._error
            history = tuple(self._sample_history)
            first_seen = self._sample_history.first_seen_timestamp
            newest_evicted = self._sample_history.newest_evicted_timestamp
            if (
                self._sample_history.eviction_count > 0
                and first_seen is not None
                and newest_evicted is not None
                and first_seen <= end_monotonic_s
                and newest_evicted > start_monotonic_s
            ):
                raise RuntimeError(
                    f"Xense history for {self.config.serial_number} no longer covers "
                    f"tactile window start {start_monotonic_s:.9f}; increase "
                    "sync_history_size or reduce recorder stalls"
                )
            return tuple(
                self._copy_sample(sample)
                for sample in history
                if start_monotonic_s < sample.capture_monotonic_s <= end_monotonic_s
            )

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Read the latest image; SDK access stays on the capture thread."""
        mode = self.config.color_mode if color_mode is None else ColorMode(color_mode)
        frame = self._latest_frame(self.config.timeout_ms)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if mode == ColorMode.RGB else frame

    def async_read(self, timeout_ms: float | None = None) -> NDArray[Any]:
        frame = self._latest_frame(self.config.timeout_ms if timeout_ms is None else timeout_ms)
        if self.config.color_mode == ColorMode.RGB:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def async_read_with_marker_motion_3d(
        self, timeout_ms: float | None = None
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Return RGB/BGR and marker displacement from the same cached SDK sample."""
        if not self.saves_marker_motion_3d:
            raise RuntimeError("Marker3DFlow saving is disabled for this Xense camera")
        sample = self._latest_sample(self.config.timeout_ms if timeout_ms is None else timeout_ms)
        if sample.marker_motion_3d is None or sample.sensor_timestamp_s is None:
            raise RuntimeError("Latest Xense sample has no Marker3DFlow or TimeStamp")
        frame = sample.frame_bgr
        if self.config.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, {
            self.motion_3d_feature_key: sample.marker_motion_3d,
            "sensor_timestamp_s": sample.sensor_timestamp_s,
            "capture_monotonic_s": sample.capture_monotonic_s,
            "capture_monotonic_ns": sample.capture_monotonic_ns,
            "device_to_host_offset_s": (sample.capture_monotonic_s - sample.sensor_timestamp_s),
        }

    def async_read_with_marker_motion_3d_latest_before(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float,
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Return the latest causal marker sample within the age bound."""
        if not self.saves_marker_motion_3d:
            raise RuntimeError("Marker3DFlow saving is disabled for this Xense camera")
        sample, sync_offset_ms = self.latest_before_sample(
            target_monotonic_s,
            max_skew_ms,
            wait_ms,
        )
        if sample.marker_motion_3d is None or sample.sensor_timestamp_s is None:
            raise RuntimeError("Nearest Xense sample has no Marker3DFlow or TimeStamp")
        frame = sample.frame_bgr
        if self.config.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, {
            self.motion_3d_feature_key: sample.marker_motion_3d,
            "sensor_timestamp_s": sample.sensor_timestamp_s,
            "capture_monotonic_s": sample.capture_monotonic_s,
            "capture_monotonic_ns": sample.capture_monotonic_ns,
            "device_to_host_offset_s": (sample.capture_monotonic_s - sample.sensor_timestamp_s),
            "sync_target_monotonic_s": target_monotonic_s,
            "sync_offset_ms": sync_offset_ms,
        }

    def async_read_with_marker_motion_3d_nearest(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float,
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Backward-compatible alias with causal latest-before semantics."""
        return self.async_read_with_marker_motion_3d_latest_before(
            target_monotonic_s,
            max_skew_ms,
            wait_ms,
        )

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError()
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=self.config.timeout_ms / 1000)
            if self._thread.is_alive():
                # Do not release a device while a native SDK call is still using it.
                raise TimeoutError("Xense SDK read is blocked; retry disconnect after it returns")
        self._sensor.release()
        self._sensor = None
        self._thread = None
        self._frame = None
        self._sample = None
        self._sample_history.clear()
