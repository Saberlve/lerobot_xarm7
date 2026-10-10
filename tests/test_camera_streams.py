"""Native camera clocks and approved tactile video storage."""

from types import SimpleNamespace

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot_robot_ufactory.cameras.synchronization import (
    TimestampedCameraBuffer,
    TimestampedRGBSample,
)
from lerobot_robot_ufactory.scripts.uf_lerobot_record import EpisodeSynchronization
from lerobot_robot_ufactory.datasets import camera_streams
from lerobot_robot_ufactory.datasets.camera_streams import (
    apply_camera_storage_plan,
    camera_recording_plan,
    camera_samples_between,
)
from lerobot_robot_ufactory.datasets.tactile_indices import apply_tactile_index_plan


def sample(t, value=1):
    return SimpleNamespace(
        frame_bgr=np.full((64, 64, 3), value, np.uint8),
        capture_monotonic_s=t,
        capture_monotonic_ns=round(t * 1e9),
        sensor_timestamp_s=None,
        marker_motion_3d=None,
    )


def robot_with_rates():
    return SimpleNamespace(
        prefix="arm.",
        cameras={
            "equal": SimpleNamespace(fps=15),
            "rgb": SimpleNamespace(fps=30),
            "photon": SimpleNamespace(fps=60, samples_between=lambda *args: ()),
            "slow": SimpleNamespace(fps=10),
        },
    )


def test_native_rates_do_not_change_the_row_clock():
    robot = robot_with_rates()
    plan = camera_recording_plan(robot, 15)
    assert [item["fps"] for item in plan.values()] == [15, 30, 60, 10]
    features = {f"observation.images.{name}": {"dtype": "video"} for name in plan}
    apply_camera_storage_plan(features, plan)
    assert features["observation.images.arm.equal"]["dtype"] == "video"
    assert features["observation.images.arm.rgb"]["dtype"] == "video"
    assert features["observation.images.arm.photon"]["dtype"] == "video"
    assert features["observation.images.arm.slow"]["dtype"] == "video"
    robot.cameras["photon"].fps = 15
    plan = camera_recording_plan(robot, 15)
    assert not plan["arm.photon"]["interval"]
    assert plan["arm.photon"]["storage"] == "video"
    assert plan["arm.photon"]["codec"] == "h264"
    apply_camera_storage_plan(features, plan)
    assert features["observation.images.arm.photon"]["dtype"] == "video"
    multi = camera_recording_plan(SimpleNamespace(robots={"a": robot}), 15)
    assert multi == plan


@pytest.mark.parametrize("fps", [15, 60])
@pytest.mark.parametrize("mesh", [False, True])
@pytest.mark.parametrize("use_videos", [False, True])
def test_tactile_ranges_do_not_depend_on_mesh_or_matching_rates(fps, mesh, use_videos):
    camera = SimpleNamespace(
        fps=fps,
        samples_between=lambda *args: (),
        config=SimpleNamespace(motion_3d_output="Mesh3DFlow" if mesh else None),
    )
    plan = camera_recording_plan(
        SimpleNamespace(cameras={"photon": camera}), 15, use_videos=use_videos
    )
    key = "observation.photon.tactile_range"
    features = {}
    if mesh:
        features["observation.photon.mesh_motion_3d"] = {"dtype": "float32"}
    apply_tactile_index_plan(features, plan)
    assert features[key] == {"dtype": "int64", "shape": (2,), "names": None}
    assert "observation.photon.mesh_motion_3d" not in features
    targets = ["video" if use_videos else "image"] + (["mesh3dflow"] if mesh else [])
    assert plan["photon"]["tactile_range_targets"] == targets


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True])
def test_invalid_camera_rates_are_rejected(fps):
    with pytest.raises(ValueError, match="fps"):
        camera_recording_plan(SimpleNamespace(cameras={"bad": SimpleNamespace(fps=fps)}), 15)


def test_rgb_windows_have_no_future_duplicate_or_silent_eviction():
    source = TimestampedCameraBuffer(SimpleNamespace(fps=60), history_size=3)
    frames = [
        TimestampedRGBSample(np.full((2, 2, 3), i, np.uint8), t, round(t * 1e9))
        for i, t in enumerate([1.0, 1.02, 1.04])
    ]
    source._history.extend(frames)
    selected = source.samples_between(1.0, 1.03)
    assert [s.capture_monotonic_s for s in selected] == [1.02]
    selected[0].frame[:] = 255
    assert source._history[1].frame[0, 0, 0] == 1
    assert source.samples_between(1.02, 1.03) == ()
    source._newest_evicted_s = 1.0
    assert len(source.samples_between(1.0, 1.04)) == 2
    with pytest.raises(RuntimeError, match="history"):
        source.samples_between(0.99, 1.04)
    with pytest.raises(ValueError):
        source.samples_between(2, 1)


def test_rgb_streams_use_the_single_reader_and_preserve_channel_order():
    source = TimestampedCameraBuffer(SimpleNamespace(fps=30), history_size=3)
    image = np.zeros((2, 2, 3), np.uint8)
    image[:, :, 0] = 251
    source._history.append(TimestampedRGBSample(image, 1.02, 1_020_000_000))
    robot = SimpleNamespace(
        prefix="arm.", cameras={"rgb": SimpleNamespace(fps=30)}, _rgb_sync_buffers={"rgb": source}
    )
    result = camera_samples_between(robot, 1, 1.03, ("arm.rgb",))
    assert len(result["arm.rgb"]) == 1
    assert result["arm.rgb"][0].frame_bgr[0, 0].tolist() == [0, 0, 251]


def test_rgb_native_video_has_60fps_and_interval_frame_indexes(tmp_path):
    sync = EpisodeSynchronization(
        None,
        15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("rgb",),
        stream_fps={"rgb": 60},
        video_streams={"rgb": "h264"},
    )
    try:
        sync.add_frame(
            0,
            2.06,
            None,
            action_send_start_s=2.06,
            tactile_window_start_s=2.0,
            tactile_samples={"rgb": tuple(sample(2.0 + i / 60, i * 40) for i in [1, 2, 3])},
        )
        sync.add_frame(
            1,
            2.13,
            None,
            action_send_start_s=2.13,
            tactile_window_start_s=2.06,
            tactile_samples={"rgb": tuple(sample(2.0 + i / 60, i * 30) for i in [4, 5, 6, 7])},
        )
        sync.write(tmp_path, 0)
    finally:
        sync.discard()
    root = tmp_path / "tactile_streams/rgb/episode_000000"
    with av.open(str(root / "video.mp4")) as video:
        assert video.streams.video[0].average_rate == 60
        assert len(list(video.decode(video=0))) == 7
    rows = pq.read_table(root / "samples.parquet").to_pylist()
    assert [row["video_frame_index"] for row in rows] == list(range(7))
    assert [row["video_timestamp_s"] for row in rows] == pytest.approx([i / 60 for i in range(7)])
    assert all(row["frame_path"] is None for row in rows)
    assert all((tmp_path / row["video_path"]).is_file() for row in rows)
    assert not (root / "frames").exists()


def test_rgb_encode_failure_retains_staged_pngs(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(camera_streams, "encode_camera_interval", fail)
    sync = EpisodeSynchronization(
        None,
        15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("rgb",),
        stream_fps={"rgb": 60},
        video_streams={"rgb": "h264"},
    )
    try:
        sync.add_frame(
            0,
            3.1,
            None,
            action_send_start_s=3.1,
            tactile_window_start_s=3.0,
            tactile_samples={"rgb": (sample(3.05),)},
        )
        with pytest.raises(RuntimeError, match="encoder failed"):
            sync.write(tmp_path, 0)
        staged = sync.tactile_recorder.staging_root / "rgb/frames/frame_000000.png"
        assert staged.is_file()
        assert not (tmp_path / "timestamps/episode_000000_commit.json").exists()
    finally:
        sync.discard()


def test_rgb_watermark_wait_excludes_future_and_surfaces_timeout():
    from threading import Thread
    from time import sleep

    source = TimestampedCameraBuffer(SimpleNamespace(fps=30), history_size=4)
    source._history.append(TimestampedRGBSample(np.zeros((2, 2, 3), np.uint8), 1.01))

    def complete_inflight():
        sleep(0.01)
        with source._condition:
            source._history.append(TimestampedRGBSample(np.zeros((2, 2, 3), np.uint8), 1.02))
            source._history.append(TimestampedRGBSample(np.zeros((2, 2, 3), np.uint8), 1.04))
            source._condition.notify_all()

    thread = Thread(target=complete_inflight)
    thread.start()
    assert [s.capture_monotonic_s for s in source.samples_between(1, 1.03, 1000)] == [1.01, 1.02]
    thread.join()
    with pytest.raises(TimeoutError, match="advance"):
        source.samples_between(1.04, 1.1, 1)


def test_initial_representatives_copy_exact_rgb_and_photon_samples():
    from lerobot_robot_ufactory.datasets.camera_streams import initial_camera_samples

    source = TimestampedCameraBuffer(SimpleNamespace(fps=30), history_size=3)
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[:, :, 0] = 251
    source._history.append(TimestampedRGBSample(rgb, 0.99, 990_000_000))
    photon = sample(1.0)
    photon.marker_motion_3d = np.ones((2, 2, 3), np.float64)
    robot = SimpleNamespace(
        prefix="arm.", cameras={
            "rgb": SimpleNamespace(fps=30),
            "photon": SimpleNamespace(sync_samples=lambda: (photon,)),
        }, _rgb_sync_buffers={"rgb": source},
    )
    timing = {
        name: {"camera_stream_name": f"arm.{name}", "capture_monotonic_s": t,
               "capture_monotonic_ns": round(t * 1e9)}
        for name, t in [("rgb", 0.99), ("photon", 1.0)]
    }
    selected = initial_camera_samples(
        SimpleNamespace(robots={"arm": robot}), timing, 1.0, ("arm.rgb", "arm.photon")
    )
    assert selected["arm.rgb"].frame_bgr[0, 0].tolist() == [0, 0, 251]
    assert selected["arm.photon"].marker_motion_3d.dtype == np.float64
    selected["arm.photon"].marker_motion_3d[:] = 0
    selected["arm.rgb"].frame_bgr[:] = 0
    assert photon.marker_motion_3d.all() and rgb[0, 0, 0] == 251
    assert initial_camera_samples(robot, timing, 0.98, ("arm.rgb", "arm.photon")) == {}
    source._history.clear()
    with pytest.raises(RuntimeError, match="no longer available"):
        initial_camera_samples(robot, timing, 1.0, ("arm.rgb",))
