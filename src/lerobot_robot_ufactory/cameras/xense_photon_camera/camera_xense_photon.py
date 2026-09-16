"""Xense SDK images exposed through LeRobot's camera interface.

Each sensor has its own capture thread. Only that thread reads the SDK;
consumers receive timestamped copies with bounded cache age.
"""

from collections import deque
from dataclasses import dataclass
import inspect
import logging
from threading import Condition, Event, Thread
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy.typing import NDArray

from .configuration_xense_photon import XensePhotonCameraConfig


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


class XensePhotonCamera(Camera):
    def __init__(self, config: XensePhotonCameraConfig):
        super().__init__(config)
        self.config = config
        self._sensor = None
        self._thread = None
        self._condition = Condition()
        self._stop = Event()
        self._frame = None
        self._sample = None
        self._sample_history = deque(maxlen=self.config.sync_history_size)
        self._frame_time = 0.0
        self._error = None

    @property
    def is_connected(self) -> bool:
        return self._sensor is not None

    @property
    def saves_marker_motion_3d(self) -> bool:
        return self.config.save_marker_motion_3d

    @property
    def motion_3d_feature_key(self) -> str:
        return "mesh_motion_3d" if self.config.motion_3d_output == "Mesh3DFlow" else "marker_motion_3d"

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
        kwargs = {"use_gpu": self.config.use_gpu, "disable_infer": self.config.disable_infer}
        parameters = inspect.signature(sensor_class.create).parameters
        if "use_gpu" not in parameters and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            kwargs.pop("use_gpu")
            logging.getLogger(__name__).info(
                "This Xense SDK selects its inference device from the sensor runtime "
                "configuration; the legacy use_gpu option is not supported."
            )
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
            self._thread.start()
            if warmup:
                self.async_read()
        except BaseException:
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
                captured_at = perf_counter()
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
                            "Xense TimeStamp is not in Unix seconds: "
                            f"{sensor_timestamp}"
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
                    if (
                        last_mapped_monotonic_s is not None
                        and mapped_monotonic_s <= last_mapped_monotonic_s
                    ):
                        last_sensor_timestamp = sensor_timestamp
                        self._stop.wait(
                            max(0, 1 / self.fps - (perf_counter() - started))
                        )
                        continue
                    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                        raise ValueError("Xense image must be a uint8 numpy array")
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError(f"Expected Xense HxWx3 image, got {frame.shape}")
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    if self.config.save_marker_motion_3d:
                        marker_motion_3d = np.asarray(marker_motion_3d, dtype=np.float32)
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
                raise RuntimeError(f"Xense capture failed for {self.config.serial_number}") from self._error
            return tuple(self._sample_history)

    def export_sync_sample(self, sample: XensePhotonSample) -> tuple[NDArray[Any], dict]:
        sample = self._copy_sample(sample)
        frame = sample.frame_bgr
        if self.config.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timing = {
            "capture_monotonic_s": sample.capture_monotonic_s,
            "sensor_timestamp_s": sample.sensor_timestamp_s,
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
        )

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

    def _nearest_sample(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float,
    ) -> tuple[XensePhotonSample, float]:
        """Select a buffered sample nearest to one host-clock time anchor.

        The call waits for a sample at or after the anchor before selecting, so
        a later frame cannot silently be closer than the returned one. A sample
        outside the explicit time budget is rejected instead of being recorded.
        """
        if not all(
            np.isfinite(value) and value >= 0
            for value in (target_monotonic_s, max_skew_ms, wait_ms)
        ):
            raise ValueError("Xense synchronization values must be finite and non-negative")
        deadline = perf_counter() + wait_ms / 1000
        with self._condition:
            while True:
                if not self.is_connected or self._stop.is_set():
                    raise DeviceNotConnectedError()
                if self._error is not None:
                    raise RuntimeError(
                        f"Xense capture failed for {self.config.serial_number}"
                    ) from self._error
                samples = tuple(self._sample_history)
                has_sample_at_or_after_anchor = any(
                    sample.capture_monotonic_s >= target_monotonic_s for sample in samples
                )
                if samples and has_sample_at_or_after_anchor:
                    sample = min(
                        samples,
                        key=lambda item: abs(item.capture_monotonic_s - target_monotonic_s),
                    )
                    skew_ms = abs(sample.capture_monotonic_s - target_monotonic_s) * 1000
                    if skew_ms <= max_skew_ms:
                        return self._copy_sample(sample), skew_ms
                    raise TimeoutError(
                        f"Nearest Xense frame for {self.config.serial_number} is "
                        f"{skew_ms:.3f} ms from the synchronization anchor; "
                        f"limit is {max_skew_ms:.3f} ms"
                    )
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No Xense frame at or after the synchronization anchor for "
                        f"{self.config.serial_number} within {wait_ms:.3f} ms"
                    )
                self._condition.wait(remaining)

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
        sample = self._latest_sample(
            self.config.timeout_ms if timeout_ms is None else timeout_ms
        )
        if sample.marker_motion_3d is None or sample.sensor_timestamp_s is None:
            raise RuntimeError("Latest Xense sample has no Marker3DFlow or TimeStamp")
        frame = sample.frame_bgr
        if self.config.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, {
            self.motion_3d_feature_key: sample.marker_motion_3d,
            "sensor_timestamp_s": sample.sensor_timestamp_s,
            "capture_monotonic_s": sample.capture_monotonic_s,
        }

    def async_read_with_marker_motion_3d_nearest(
        self,
        target_monotonic_s: float,
        max_skew_ms: float,
        wait_ms: float,
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """Return a marker sample within a configured host-clock error bound."""
        if not self.saves_marker_motion_3d:
            raise RuntimeError("Marker3DFlow saving is disabled for this Xense camera")
        sample, sync_offset_ms = self._nearest_sample(
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
            "sync_target_monotonic_s": target_monotonic_s,
            "sync_offset_ms": sync_offset_ms,
        }

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
