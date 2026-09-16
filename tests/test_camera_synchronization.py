from time import sleep

import numpy as np
import pytest

from lerobot_robot_ufactory.cameras.synchronization import (
    TimestampedCameraBuffer, TimestampedRGBSample, select_synchronized_samples,
)


class _Camera:
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.index = 0

    def __str__(self):
        return "fake-rgb"

    def async_read(self, timeout_ms: float):
        sleep(self.delay_s)
        self.index += 1
        return np.full((2, 3, 3), self.index % 255 or 1, dtype=np.uint8)


def test_timestamped_rgb_buffer_pairs_a_new_frame_to_anchor():
    buffer = TimestampedCameraBuffer(_Camera(), history_size=8)
    buffer.start()
    try:
        from time import perf_counter

        anchor = perf_counter()
        frame, timing = buffer.nearest(anchor, max_skew_ms=20, wait_ms=200)
        assert frame.shape == (2, 3, 3)
        assert timing["sync_target_monotonic_s"] == anchor
        assert timing["sync_offset_ms"] <= 20
        frame[:] = 0
        later, _ = buffer.nearest(perf_counter(), max_skew_ms=20, wait_ms=200)
        assert later[0, 0, 0] > 0
    finally:
        buffer.stop()


def test_timestamped_rgb_buffer_rejects_a_frame_outside_bound():
    buffer = TimestampedCameraBuffer(_Camera(delay_s=0.03), history_size=4)
    buffer.start()
    try:
        from time import perf_counter

        with pytest.raises(TimeoutError, match="Nearest RGB frame"):
            buffer.nearest(perf_counter(), max_skew_ms=2, wait_ms=150)
    finally:
        buffer.stop()


def test_cached_frame_does_not_receive_new_timestamps():
    from time import perf_counter
    from threading import Event

    class CachedCamera:
        fps = 100
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        read_once = Event()

        def async_read(self, timeout_ms):
            self.read_once.set()
            return self.frame

    camera = CachedCamera()
    buffer = TimestampedCameraBuffer(camera, history_size=8)
    buffer.start()
    try:
        assert camera.read_once.wait(1)
        sleep(0.06)
        with pytest.raises(TimeoutError):
            select_synchronized_samples({"camera": buffer}, perf_counter(), 20, 20, 30)
        assert len(buffer.sync_samples()) == 1
    finally:
        buffer.stop()


def _source(offsets, anchor=100.0):
    from types import SimpleNamespace
    samples = [TimestampedRGBSample(np.zeros((2, 3, 3), np.uint8), anchor + o / 1000)
               for o in offsets]
    return SimpleNamespace(sync_samples=lambda: tuple(samples))


def test_joint_selection_finds_feasible_non_nearest_pair():
    chosen = select_synchronized_samples(
        {"a": _source([-15, 18.333]), "b": _source([-18.333, 15])}, 100, 20, 20, 0,
    )
    times = [s.capture_monotonic_s for s in chosen.values()]
    assert (max(times) - min(times)) * 1000 == pytest.approx(3.333)
    assert all(abs(t - 100) * 1000 <= 20 for t in times)


def test_joint_selection_rejects_infeasible_pair():
    with pytest.raises(TimeoutError, match="No synchronized camera combination"):
        select_synchronized_samples({"a": _source([-15]), "b": _source([15])}, 100, 20, 20, 0)


def test_deadline_can_use_valid_past_frame():
    chosen = select_synchronized_samples({"a": _source([-2])}, 100, 20, 20, 0)
    assert chosen["a"].capture_monotonic_s == pytest.approx(99.998)


def test_joint_selection_waits_for_feasible_alternative():
    from types import SimpleNamespace
    initial = _source([15]).sync_samples()
    later = _source([-18.333, 15]).sync_samples()
    calls = []

    def read():
        calls.append(1)
        return initial if len(calls) == 1 else later

    chosen = select_synchronized_samples(
        {"a": _source([-15, 30]), "b": SimpleNamespace(sync_samples=read)}, 100, 20, 20, 50,
    )
    assert len(calls) >= 2
    assert chosen["b"].capture_monotonic_s < 100


def test_umi_never_returns_cached_frame_on_timeout():
    from lerobot_robot_ufactory.cameras.umi_camera.camera_umi import UmiCamera
    camera = UmiCamera.__new__(UmiCamera)
    camera.read = lambda: None
    camera.last_frame = np.ones((2, 3, 3), np.uint8)
    with pytest.raises(TimeoutError, match="No new UMI frame"):
        camera.async_read(timeout_ms=1)


def test_gripper_delay_does_not_move_arm_state_anchor(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    robot = uf_robot.UFRobot(UFRobotConfig(id="sync", calibration_dir=tmp_path, robot_dof=7, gripper_type=2))
    clock = [100.0]
    monkeypatch.setattr(uf_robot, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    robot._log_controller_error_if_changed = lambda *args: None

    def gripper_read():
        clock[0] += 0.08
        return 0, 42

    robot.real_arm = SimpleNamespace(
        get_joint_states=lambda **kw: (0, [[0.0] * 7] * 3),
        get_gripper_g2_position=gripper_read,
    )
    anchors = []
    robot._read_synchronized_cameras = lambda anchor: anchors.append(anchor) or {}
    robot.get_observation()
    assert anchors == [100.0]
    assert robot._last_observation_sync_timing["state_age_ms"] == pytest.approx(80)
