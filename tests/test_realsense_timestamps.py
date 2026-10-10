"""Verify RealSense clock domains, atomic metadata, and in-flight RGB frames."""

import json
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pyrealsense2 as rs
import pytest
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from lerobot_robot_ufactory.cameras import realsense
from lerobot_robot_ufactory.cameras.realsense import ExposureRealSenseCamera
from lerobot_robot_ufactory.cameras.synchronization import (
    TimestampedCameraBuffer,
    TimestampedRGBSample,
    select_synchronized_samples,
)
from lerobot_robot_ufactory.scripts.uf_lerobot_record import EpisodeSynchronization
from lerobot_robot_ufactory.datasets.camera_streams import camera_samples_between


def camera_frame(
    monkeypatch,
    *,
    sensor_us=980_000,
    readout_us=1_000_000,
    domain=rs.timestamp_domain.global_time,
    sdk_ms=1_700_000_000_000.0,
):
    # Action monotonic=100s, wall=1.7e9s. SDK's frame/readout time
    # maps to 100s, while this frame's exposure midpoint maps to 99.98s.
    ticks = iter([100_020_000_000, 100_020_000_000, 100_020_000_100])
    monkeypatch.setattr(realsense, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(realsense, "time_ns", lambda: 1_700_000_000_020_000_050)
    metadata = {
        rs.frame_metadata_value.sensor_timestamp: sensor_us,
        rs.frame_metadata_value.frame_timestamp: readout_us,
        rs.frame_metadata_value.actual_exposure: 10_000,
    }
    accessed = []

    def get_metadata(field):
        assert metadata[field] is not None  # Never query unsupported fields.
        accessed.append(field)
        return metadata[field]

    frame = SimpleNamespace(
        get_frame_number=lambda: 42,
        get_timestamp=lambda: sdk_ms,
        get_frame_timestamp_domain=lambda: domain,
        supports_frame_metadata=lambda field: metadata[field] is not None,
        get_frame_metadata=get_metadata,
        get_data=lambda: np.full((2, 3, 3), 42, np.uint8),
    )
    pipeline_reads = []

    def read(timeout_ms):
        pipeline_reads.append(timeout_ms)
        return True, SimpleNamespace(get_color_frame=lambda: frame)

    camera = ExposureRealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name="123",
            width=3,
            height=2,
            fps=15,
        )
    )
    camera.rs_pipeline = SimpleNamespace(try_wait_for_frames=read)
    camera.rs_profile = object()
    return camera, accessed, pipeline_reads


def test_exposure_midpoint_uses_same_frame_metadata_and_global_clock(monkeypatch):
    camera, _, reads = camera_frame(monkeypatch)
    sample = camera.read_timestamped(require_exposure_timestamp=True)
    assert reads == [200]
    assert sample.frame[0, 0, 0] == sample.timing["frame_number"] == 42
    assert sample.capture_monotonic_s == pytest.approx(99.98)
    assert sample.timing["sensor_timestamp_us"] == 980_000
    assert sample.timing["frame_timestamp_us"] == 1_000_000
    assert sample.timing["sensor_timestamp_s"] == 0.98
    assert sample.timing["received_monotonic_s"] == 100.02
    assert sample.timing["exposure_to_receipt_ms"] == pytest.approx(40)
    assert sample.timing["timestamp_source"] == "realsense_sensor_timestamp"
    assert sample.timing["clock_bridge_uncertainty_ns"] == 50


def test_same_frame_delta_handles_device_timestamp_wrap(monkeypatch):
    camera, _, _ = camera_frame(monkeypatch, sensor_us=2**32 - 10_000, readout_us=10_000)
    assert camera.read_timestamped(require_exposure_timestamp=True).capture_monotonic_s == 99.98


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"sensor_us": None}, "SENSOR_TIMESTAMP unsupported"),
        ({"readout_us": None}, "FRAME_TIMESTAMP unavailable"),
        ({"domain": rs.timestamp_domain.hardware_clock}, "global clock mapping"),
        ({"domain": rs.timestamp_domain.system_time}, "global clock mapping"),
        ({"sensor_us": 1_001_000}, "Invalid exposure-to-readout"),
        ({"sdk_ms": float("nan")}, "Invalid SDK global timestamp"),
        ({"sdk_ms": 1_700_000_001_000.0}, "later than host receipt"),
    ],
)
def test_unsupported_or_unmapped_exposure_has_explicit_fallback(monkeypatch, kwargs, reason):
    camera, _, _ = camera_frame(monkeypatch, **kwargs)
    sample = camera.read_timestamped()
    assert sample.capture_monotonic_ns == sample.timing["received_monotonic_ns"]
    assert sample.timing["timestamp_source"] == "host_receipt"
    assert reason in sample.timing["timestamp_fallback_reason"]
    assert sample.timing["exposure_to_receipt_ms"] is None
    camera, _, _ = camera_frame(monkeypatch, **kwargs)
    with pytest.raises(RuntimeError, match=reason):
        camera.read_timestamped(require_exposure_timestamp=True)


def test_factory_routes_configured_realsense_to_exposure_backend():
    from lerobot_robot_ufactory.cameras.utils import make_cameras_from_configs

    cameras = make_cameras_from_configs(
        {
            "rgb": RealSenseCameraConfig(serial_number_or_name="123", width=3, height=2, fps=15),
        }
    )
    assert isinstance(cameras["rgb"], ExposureRealSenseCamera)


def test_selection_waits_for_late_causal_exposure_before_watermark():
    from time import perf_counter

    source = TimestampedCameraBuffer(SimpleNamespace(read_timestamped=lambda: None), history_size=8)
    image = np.zeros((2, 2, 3), np.uint8)
    source._history.append(TimestampedRGBSample(image, 99.97))
    entered_wait = Event()
    original_wait = source.wait_until_after

    def observed_wait(target, deadline):
        entered_wait.set()
        return original_wait(target, deadline)

    source.wait_until_after = observed_wait

    def delayed_delivery():
        assert entered_wait.wait(1)
        with source._condition:
            # Both arrive after action=100.0; the first exposure is causal.
            source._history.append(
                TimestampedRGBSample(
                    image,
                    99.99,
                    timing={
                        "received_monotonic_s": 100.02,
                        "frame_number": 43,
                    },
                )
            )
            source._history.append(TimestampedRGBSample(image, 100.01))
            source._condition.notify_all()

    thread = Thread(target=delayed_delivery)
    thread.start()
    try:
        chosen = select_synchronized_samples({"rgb": source}, 100.0, 70, 90, 1000)
        assert chosen["rgb"].capture_monotonic_s == 99.99
        assert chosen["rgb"].timing["received_monotonic_s"] > 100.0
        assert source.latest_before(100.0, 70, 100)[1]["frame_number"] == 43
        assert [s.capture_monotonic_s for s in source.samples_between(99.98, 100.0, 100)] == [99.99]
        with pytest.raises(TimeoutError, match="advance"):
            select_synchronized_samples({"rgb": source}, 100.1, 70, 90, 1)
        source._error = RuntimeError("device failed")
        with pytest.raises(RuntimeError, match="capture failed"):
            source.wait_until_after(100, perf_counter() + 1)
    finally:
        thread.join(1)


def test_frame_numbers_deduplicate_and_clock_rewind_is_rejected():
    frames = iter([(1, 1.0), (1, 1.0), (2, 1.02), (3, 0.99)])

    def read(**kwargs):
        number, stamp = next(frames)
        return TimestampedRGBSample(
            np.zeros((2, 2, 3), np.uint8), stamp, timing={"frame_number": number}
        )

    source = TimestampedCameraBuffer(SimpleNamespace(read_timestamped=read), history_size=8)
    source._capture_loop()
    assert [s.timing["frame_number"] for s in source._history] == [1, 2]
    assert "clock moved backwards" in str(source._error)
    with pytest.raises(RuntimeError, match="capture failed"):
        source.samples_between(0, 2)


@pytest.mark.parametrize("fallback", [False, True])
def test_rgb_metadata_survives_interval_and_action_sidecars(tmp_path, monkeypatch, fallback):
    camera, _, _ = camera_frame(
        monkeypatch,
        domain=rs.timestamp_domain.hardware_clock if fallback else rs.timestamp_domain.global_time,
    )
    sample = camera.read_timestamped(require_exposure_timestamp=not fallback)
    source = TimestampedCameraBuffer(camera, history_size=8)
    source._history.append(sample)
    robot = SimpleNamespace(cameras={"rgb": camera}, _rgb_sync_buffers={"rgb": source})
    samples = camera_samples_between(robot, 99.9, 100.03, ("rgb",))
    sync = EpisodeSynchronization(
        None, 15, dataset_root=tmp_path, episode_index=0, tactile_stream_names=("rgb",)
    )
    try:
        _, timing = source.export_sync_sample(sample)
        sync.add_frame(
            0,
            100.03,
            None,
            action_send_start_s=100.03,
            camera_timing={"rgb": timing},
            tactile_window_start_s=99.9,
            tactile_samples=samples,
        )
        action_timing = json.loads(sync.frames[0]["camera_timing_json"])["rgb"]
        assert action_timing["frame_number"] == 42
        expected_source = "host_receipt" if fallback else "realsense_sensor_timestamp"
        assert action_timing["timestamp_source"] == expected_source
        assert action_timing["received_monotonic_ns"] == 100_020_000_000
        sync.write(tmp_path, 0)
        rows = pq.read_table(
            tmp_path / "tactile_streams/rgb/episode_000000/samples.parquet"
        ).to_pylist()
        assert rows[0]["sensor_timestamp_s"] == 0.98
        assert json.loads(rows[0]["camera_timing_json"]) == sample.timing
        if fallback:
            assert action_timing["device_to_host_offset_s"] is None
            assert rows[0]["device_to_host_offset_s"] is None
            intervals = json.loads(sync.frames[0]["camera_intervals_json"])
            assert intervals["rgb"]["device_to_host_offset_s"] == [None]
    finally:
        sync.discard()
