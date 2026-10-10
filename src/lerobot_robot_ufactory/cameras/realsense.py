"""RealSense RGB metadata read atomically with the pixels of each color frame."""

import logging
from time import perf_counter_ns, time_ns

import numpy as np
import pyrealsense2 as rs
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.utils.errors import DeviceNotConnectedError

from .synchronization import TimestampedRGBSample

logger = logging.getLogger(__name__)


class ExposureRealSenseCamera(RealSenseCamera):
    """Use SDK global time to bridge device exposure time to perf_counter.

    D400 USB RGB get_timestamp() is the UVC FRAME_TIMESTAMP mapped by the
    SDK to system time. Subtract the same-frame readout-to-exposure delta,
    then bridge system time to the action clock with a bracketed clock read.
    Raw metadata remains in microseconds; get_timestamp() is milliseconds.
    No offset is fitted from USB receipt times.
    """

    def _configure_capture_settings(self):
        super()._configure_capture_settings()
        self._fallback_reason = None
        for sensor in self.rs_profile.get_device().query_sensors():
            if any(p.stream_type() == rs.stream.color for p in sensor.get_stream_profiles()):
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)

    def read_timestamped(self, timeout_ms=200, *, require_exposure_timestamp=False):
        """Read one complete frame; recording uses this as the sole SDK reader."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Stop the RealSense async reader before timestamped capture")
        ok, frames = self.rs_pipeline.try_wait_for_frames(timeout_ms=int(timeout_ms))
        received_ns = perf_counter_ns()
        if not ok or frames is None:
            raise TimeoutError(f"No new RealSense frame for {self}")
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"Missing RealSense color frame for {self}")

        # Sample the wall/monotonic bridge on this read, before copying pixels.
        before_ns = perf_counter_ns()
        wall_ns = time_ns()
        after_ns = perf_counter_ns()
        wall_to_monotonic_ns = (before_ns + after_ns) // 2 - wall_ns
        timing = {
            "received_monotonic_ns": received_ns,
            "received_monotonic_s": received_ns / 1e9,
            "frame_number": int(color.get_frame_number()),
            "sdk_timestamp_ms": float(color.get_timestamp()),
            "sdk_timestamp_domain": str(color.get_frame_timestamp_domain()),
            "wall_to_monotonic_offset_ns": wall_to_monotonic_ns,
            "clock_bridge_uncertainty_ns": (after_ns - before_ns) // 2,
        }
        for name in ("sensor_timestamp", "frame_timestamp", "actual_exposure"):
            field = getattr(rs.frame_metadata_value, name)
            supported = color.supports_frame_metadata(field)
            timing[f"{name}_supported"] = bool(supported)
            timing[f"{name}_us"] = int(color.get_frame_metadata(field)) if supported else None

        reason = None
        sensor_us, readout_us = timing["sensor_timestamp_us"], timing["frame_timestamp_us"]
        if sensor_us is None:
            reason = "SENSOR_TIMESTAMP unsupported"
        elif readout_us is None:
            reason = "FRAME_TIMESTAMP unavailable for clock mapping"
        elif color.get_frame_timestamp_domain() != rs.timestamp_domain.global_time:
            reason = "SDK global clock mapping unavailable or not ready"
        else:
            # UVC timestamps wrap every 2**32 us. Align only the same-frame
            # delta; the SDK handles its global timestamp clock separately.
            delta_us = (sensor_us - readout_us + 2**31) % 2**32 - 2**31
            if not -1_000_000 <= delta_us <= 0:
                reason = "Invalid exposure-to-readout timestamp delta"
            elif not np.isfinite(timing["sdk_timestamp_ms"]):
                reason = "Invalid SDK global timestamp"
            else:
                capture_ns = (
                    round(timing["sdk_timestamp_ms"] * 1e6)
                    + delta_us * 1_000
                    + wall_to_monotonic_ns
                )
                if capture_ns > received_ns:
                    reason = "Mapped exposure is later than host receipt"

        if reason is not None:
            if require_exposure_timestamp:
                raise RuntimeError(f"{self}: exposure timestamp required: {reason}")
            capture_ns = received_ns
            timing.update(
                timestamp_source="host_receipt",
                clock_mapping_method="host_receipt",
                timestamp_fallback_reason=reason,
            )
            if reason != getattr(self, "_fallback_reason", None):
                logger.warning("%s: falling back to host receipt: %s", self, reason)
            self._fallback_reason = reason
        else:
            self._fallback_reason = None
            timing.update(
                timestamp_source="realsense_sensor_timestamp",
                clock_mapping_method="sdk_global_time_plus_sensor_delta_to_perf_counter",
                timestamp_fallback_reason=None,
            )
        timing.update(
            sensor_timestamp_s=None if sensor_us is None else sensor_us / 1e6,
            device_to_host_offset_s=None
            if reason is not None
            else capture_ns / 1e9 - sensor_us / 1e6,
            # Includes exposure-to-readout and SDK delivery, not USB alone.
            exposure_to_receipt_ms=None if reason is not None else (received_ns - capture_ns) / 1e6,
        )
        image = self._postprocess_image(np.asanyarray(color.get_data())).copy()
        return TimestampedRGBSample(image, capture_ns / 1e9, capture_ns, timing)
