import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig
from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
from lerobot_robot_ufactory.scripts import uf_lerobot_record as record_module
from lerobot_robot_ufactory.scripts.uf_lerobot_record import (
    EpisodeSynchronization,
    _manual_action_from_observation,
    _dataset_robot_type,
    _update_manual_gripper_key_state,
    _update_manual_gripper_target,
    _prepare_dataset_root,
    _prepare_recording_episode,
    get_cfg,
)


@pytest.mark.parametrize(("robot_dof", "expected"), [(5, "xarm5"), (6, "xarm6"), (7, "xarm7")])
def test_dataset_robot_type_uses_concrete_xarm_model(robot_dof, expected):
    robot = SimpleNamespace(name="UFACTORY Robot", config=SimpleNamespace(robot_dof=robot_dof))

    assert _dataset_robot_type(robot) == expected


def test_dataset_robot_type_preserves_non_xarm_robot_name():
    robot = SimpleNamespace(name="other_robot", config=SimpleNamespace(robot_dof=None))

    assert _dataset_robot_type(robot) == "other_robot"


@pytest.mark.parametrize("robot_dof", [5, 6, 7])
def test_uf_robot_type_matches_concrete_xarm_model(robot_dof, tmp_path):
    robot = UFRobot(
        UFRobotConfig(
            id="test_robot_type",
            calibration_dir=tmp_path,
            robot_dof=robot_dof,
            gripper_type=0,
        )
    )

    assert robot.robot_type == f"xarm{robot_dof}"
    assert _dataset_robot_type(robot) == robot.robot_type


def test_synchronization_defaults_to_enabled():
    assert record_module.UFRecordConfig.__dataclass_fields__["synchronize"].default is True


def test_episode_synchronization_writes_training_schema_sidecars(tmp_path):
    class FakeController:
        def action_timings(self):
            return [
                {
                    "action_index": 0,
                    "gello_read_start_ns": 10,
                    "gello_read_end_ns": 20,
                    "command_send_start_ns": 21,
                    "command_send_end_ns": 30,
                }
            ]

    synchronization = EpisodeSynchronization(FakeController(), fps=30)
    synchronization.add_frame(
        frame_index=0,
        state_sample_s=1.001,
        state_rt_receive_s=1.0,
        action_send_start_s=1.002,
        action_send_end_s=1.003,
        action_index=0,
        camera_timing={"camera": {"frame_index": 0, "read_start_s": 1.0, "read_end_s": 1.002}},
    )
    synchronization.write(tmp_path, episode_index=3)

    import pyarrow.parquet as pq

    frame_rows = pq.read_table(tmp_path / "timestamps" / "episode_000003.parquet").to_pylist()
    action_rows = pq.read_table(
        tmp_path / "timestamps" / "episode_000003_actions.parquet"
    ).to_pylist()
    assert frame_rows[0]["frame_index"] == 0
    assert frame_rows[0]["action_send_start_ns"] == 1_002_000_000
    assert frame_rows[0]["action_send_end_ns"] == 1_003_000_000
    assert frame_rows[0]["action_state_age_ms"] == pytest.approx(2.0)
    assert action_rows[0]["action_index"] == 0


def test_tactile_tensor_and_unix_timestamp_survive_dataset_save(tmp_path):
    import pyarrow.parquet as pq
    mesh_key = "observation.photon.mesh_motion_3d"
    time_key = "observation.photon.sensor_timestamp"
    features = {
        mesh_key: {"dtype": "float32", "shape": (35, 20, 3), "names": None},
        time_key: {"dtype": "float64", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        "test/photon", fps=25, features=features, root=tmp_path / "photon",
        robot_type="xarm7", use_videos=False,
    )
    timestamps = [1789473516.7106972, 1789473516.7506971]
    for timestamp in timestamps:
        frame = record_module.build_dataset_frame(features, {
            "photon.mesh_motion_3d": np.full((35, 20, 3), 1.25, np.float32),
            "photon.sensor_timestamp": timestamp,
        }, prefix="observation")
        assert frame[time_key].dtype == np.float64
        dataset.add_frame({**frame, "task": "test"})
    dataset.save_episode()
    dataset.finalize()
    paths = list((tmp_path / "photon" / "data").rglob("*.parquet"))
    rows = pq.read_table(paths[0]).to_pylist()
    assert [r[time_key] for r in rows] == timestamps
    np.testing.assert_array_equal(rows[0][mesh_key], np.full((35, 20, 3), 1.25))


def test_sync_summary_uses_capture_error_not_read_duration(tmp_path):
    import json
    sync = EpisodeSynchronization(None, fps=25)
    sync.add_frame(0, 100.0, 99.999, camera_timing={
        "photon": {"read_start_s": 100.0, "read_end_s": 100.2,
                   "capture_monotonic_s": 99.997, "sync_target_monotonic_s": 99.999,
                   "sync_offset_ms": 2.0, "sync_signed_offset_ms": -2.0, "pair_skew_ms": 3.0},
    }, state_age_ms=1.0, action_send_start_s=100.0, action_send_end_s=100.001)
    sync.write(tmp_path, 0)
    stats = json.loads((tmp_path / "timestamps/episode_000000_summary.json").read_text())
    assert stats["camera_action_age_ms"]["photon"]["max"] == pytest.approx(3.0)
    assert stats["camera_pair_skew_ms"]["max"] == 3.0
    assert (tmp_path / "timestamps/episode_000000.csv").is_file()


def test_sidecar_rejects_future_state_and_camera_samples():
    sync = EpisodeSynchronization(None, fps=15)
    with pytest.raises(AssertionError, match="Future state"):
        sync.add_frame(
            0,
            10.001,
            10.001,
            action_send_start_s=10.0,
            action_send_end_s=10.002,
        )

    with pytest.raises(AssertionError, match="Future camera"):
        sync.add_frame(
            0,
            9.999,
            9.999,
            camera_timing={
                "wrist": {
                    "capture_monotonic_s": 10.001,
                    "read_start_s": 10.001,
                    "read_end_s": 10.002,
                }
            },
            action_send_start_s=10.0,
            action_send_end_s=10.002,
        )


def test_raw_tactile_stream_maps_variable_windows_to_lossless_frames(tmp_path):
    import json
    import pyarrow.parquet as pq
    from lerobot_robot_ufactory.tactile.photon.camera import XensePhotonSample

    def sample(timestamp, value):
        return XensePhotonSample(
            frame_bgr=np.full((3, 4, 3), value, dtype=np.uint8),
            marker_motion_3d=None,
            sensor_timestamp_s=1_700_000_000.0 + timestamp,
            capture_monotonic_s=timestamp,
        )

    sync = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("photon_left", "photon_right"),
    )
    sync.add_frame(
        0,
        10.05,
        10.05,
        camera_timing={},
        action_send_start_s=10.066,
        action_send_end_s=10.067,
        action_index=0,
        tactile_window_start_s=10.0,
        tactile_samples={
            "photon_left": tuple(sample(t, i) for i, t in enumerate((10.010, 10.027, 10.044, 10.061), 1)),
            "photon_right": tuple(sample(t, i) for i, t in enumerate((10.015, 10.035, 10.055), 10)),
        },
    )
    sync.write(tmp_path, 0)

    frame_row = pq.read_table(
        tmp_path / "timestamps/episode_000000.parquet"
    ).to_pylist()[0]
    mapping = json.loads(frame_row["tactile_timing_json"])
    assert mapping["photon_left"]["start_index"] == 0
    assert mapping["photon_left"]["end_index"] == 4
    assert mapping["photon_left"]["frame_count"] == 4
    assert mapping["photon_right"]["end_index"] == 3

    left_index = pq.read_table(
        tmp_path / "tactile_streams/photon_left/episode_000000/samples.parquet"
    ).to_pylist()
    assert len(left_index) == 4
    assert [row["capture_monotonic_s"] for row in left_index] == [
        10.010,
        10.027,
        10.044,
        10.061,
    ]
    assert all((tmp_path / row["frame_path"]).is_file() for row in left_index)


def test_discarded_episode_removes_staged_tactile_streams(tmp_path):
    sync = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=4,
        tactile_stream_names=("photon_left", "photon_right"),
    )
    sync.discard()
    staging = tmp_path / "tactile_streams/.staging"
    assert not staging.exists() or not any(staging.iterdir())
    assert not (tmp_path / "tactile_streams/photon_left/episode_000004").exists()


def test_recording_sync_timeout_stops_realtime_controller(monkeypatch):
    calls = []

    class Teleop:
        config = SimpleNamespace(realtime_control_fps=25)

    class Controller:
        period_s = 1 / 25

        def __init__(self, **kwargs):
            pass

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

        def latest_action_sample(self, *args, **kwargs):
            now = record_module.time.perf_counter()
            return SimpleNamespace(
                action_index=0,
                command={"J1.pos": 0.0},
                send_start_s=now,
                send_end_s=now,
            )

    def fail(_target_monotonic_s=None):
        raise TimeoutError("synchronization failed")

    robot = SimpleNamespace(
        _control_space="joint", enable_logs=False, action_features={"J1.pos": float},
        get_observation=lambda: {"J1.pos": 0.0}, get_realtime_observation=fail,
    )
    monkeypatch.setattr(record_module, "UFBaseTeleop", Teleop)
    monkeypatch.setattr(record_module, "RealtimeTeleopController", Controller)
    pipelines = record_module.make_default_processors()
    with pytest.raises(TimeoutError, match="synchronization failed"):
        record_module.record_loop(
            robot, {"exit_early": False}, 25, *pipelines,
            teleop=Teleop(), control_time_s=1,
        )
    assert calls == ["start", "stop"]


def test_first_tick_skips_stale_first_action_anchor(monkeypatch):
    # The first command's send-start is stamped before a blocking first move
    # (mode/state switch plus the initial wait=True servo command). The sample
    # only becomes visible after that move, so the newest eligible action at
    # the first record tick can be hundreds of ms old — far beyond the ~70 ms
    # max ages enforced by state/tactile pairing. The first tick must reject
    # anchors older than one control period and wait for a fresh command.
    import threading

    calls = []
    period_s = 1 / 25
    # The blocking first move: sample 0's send-start is stamped at
    # +first_send_start_s, but the sample only becomes visible at
    # +first_sample_visible_s, and the controller stays quiet (still
    # blocked) until +controller_quiet_end_s afterwards.
    first_send_start_s = 0.02
    first_sample_visible_s = 0.45
    controller_quiet_end_s = 0.75

    class Teleop:
        config = SimpleNamespace(realtime_control_fps=25)

    class Producer:
        """Mimics the realtime controller around a blocking first move."""

        def __init__(self):
            self._lock = threading.Lock()
            self._samples = []
            self._stop_event = threading.Event()
            self._next_index = 0
            self._started_s = record_module.time.perf_counter()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def _append(self, send_start_s):
            with self._lock:
                self._samples.append(
                    SimpleNamespace(
                        action_index=self._next_index,
                        command={"J1.pos": 0.0},
                        send_start_s=send_start_s,
                        send_end_s=send_start_s + 0.001,
                        send_start_ns=int(send_start_s * 1_000_000_000),
                        send_end_ns=int((send_start_s + 0.001) * 1_000_000_000),
                    )
                )
                self._next_index += 1

        def _elapsed(self):
            return record_module.time.perf_counter() - self._started_s

        def _run(self):
            # The first command's send-start is stamped before the blocking
            # first move; the sample only becomes visible once the move
            # finishes, and further commands stay blocked for a while after.
            first_send_start = self._started_s + first_send_start_s
            while (
                self._elapsed() < first_sample_visible_s
                and not self._stop_event.is_set()
            ):
                self._stop_event.wait(0.005)
            if self._stop_event.is_set():
                return
            self._append(first_send_start)
            while (
                self._elapsed() < controller_quiet_end_s
                and not self._stop_event.is_set()
            ):
                self._stop_event.wait(0.005)
            while not self._stop_event.is_set():
                self._append(record_module.time.perf_counter())
                self._stop_event.wait(period_s)

        def stop(self):
            self._stop_event.set()
            self._thread.join(timeout=1.0)

        def latest_action_sample(self, after_action_index, *, not_before_s, wait_s):
            deadline = record_module.time.perf_counter() + wait_s
            while True:
                with self._lock:
                    eligible = [
                        sample
                        for sample in self._samples
                        if sample.action_index > after_action_index
                        and (not_before_s is None or sample.send_start_s >= not_before_s)
                    ]
                if eligible:
                    sample = eligible[-1]
                    calls.append(
                        {
                            "after": after_action_index,
                            "not_before_s": not_before_s,
                            "wait_s": wait_s,
                            "returned_send_start_s": sample.send_start_s,
                            "returned_at": record_module.time.perf_counter(),
                        }
                    )
                    return sample
                if record_module.time.perf_counter() >= deadline:
                    raise TimeoutError("no eligible action within wait_s")
                record_module.time.sleep(0.001)

    producer = Producer()

    class Controller:
        period_s = 1 / 25

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            producer.stop()

        def update_observation(self, obs):
            pass

        def latest_action_sample(self, *args, **kwargs):
            return producer.latest_action_sample(*args, **kwargs)

    robot = SimpleNamespace(
        _control_space="joint",
        enable_logs=False,
        action_features={"J1.pos": float},
        get_observation=lambda: {"J1.pos": 0.0},
    )
    observed_anchors = []

    def observe(anchor_s=None, target_monotonic_ns=None):
        observed_anchors.append(anchor_s)
        robot._last_realtime_observation_monotonic_s = anchor_s
        return {"J1.pos": 0.0}

    robot.get_realtime_observation = observe
    monkeypatch.setattr(record_module, "UFBaseTeleop", Teleop)
    monkeypatch.setattr(record_module, "Teleoperator", Teleop)
    monkeypatch.setattr(record_module, "RealtimeTeleopController", Controller)
    pipelines = record_module.make_default_processors()
    try:
        record_module.record_loop(
            robot, {"exit_early": False}, 25, *pipelines,
            teleop=Teleop(), control_time_s=controller_quiet_end_s + 0.2,
        )
    finally:
        producer.stop()

    assert calls, "record loop never consumed a realtime action"
    first = calls[0]
    assert first["after"] == -1
    assert first["wait_s"] == 1.0
    assert first["not_before_s"] is not None
    # The scenario reproduced: the first wait returned the stale pre-move
    # sample whose anchor was stamped before the blocking first move.
    assert first["returned_send_start_s"] < producer._started_s + 0.1
    # Frame 0 skipped that stale anchor and is anchored on a command issued
    # after the controller resumed, fresh at consumption time.
    assert observed_anchors[0] > producer._started_s + controller_quiet_end_s - 0.05
    returned = [call["returned_send_start_s"] for call in calls]
    frame0_call = next(
        call for call, anchor in zip(calls, returned) if anchor == observed_anchors[0]
    )
    assert calls.index(frame0_call) >= 1, "stale first anchor was not skipped"
    # After frame 0, ticks consume the newest action without a causal bound.
    start_later = calls[calls.index(frame0_call) + 1:]
    assert start_later, "expected steady-state ticks after the first frame"
    for call in start_later:
        assert call["not_before_s"] is None


class FakeXArm:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.connected = True
        self.axis = 6
        self.error_code = 0
        self.mode = 0
        self.initial_point = [0.0, -30.0, 0.0, 0.0, 0.0, 30.0]
        self._arm = type("FakeArmTransport", (), {"_baud_checkset": False})()
        self.gripper_position = 800
        self.gripper_g2_position = 84
        self.calls = []

    def motion_enable(self, **kwargs):
        self.calls.append(("motion_enable", kwargs))

    def clean_error(self):
        self.calls.append(("clean_error",))

    def set_teach_sensitivity(self, sensitivity):
        self.calls.append(("set_teach_sensitivity", sensitivity))
        return 0

    def set_mode(self, mode):
        self.calls.append(("set_mode", mode))
        self.mode = mode
        return 0

    def set_state(self, state):
        self.calls.append(("set_state", state))
        return 0

    def get_initial_point(self):
        self.calls.append(("get_initial_point",))
        return 0, self.initial_point

    def set_servo_angle(self, **kwargs):
        self.calls.append(("set_servo_angle", kwargs))
        return 0

    def get_err_warn_code(self):
        return 0, [0, 0]

    def set_linear_spd_limit_factor(self, factor):
        self.calls.append(("set_linear_spd_limit_factor", factor))
        return 0

    def set_gripper_enable(self, enable):
        self.calls.append(("set_gripper_enable", enable))
        return 0

    def set_gripper_mode(self, mode):
        self.calls.append(("set_gripper_mode", mode))
        return 0

    def set_gripper_speed(self, speed):
        self.calls.append(("set_gripper_speed", speed))
        return 0

    def set_gripper_position(self, position, **kwargs):
        self.calls.append(("set_gripper_position", position, kwargs))
        self.gripper_position = position
        return 0

    def get_gripper_position(self):
        self.calls.append(("get_gripper_position",))
        return 0, self.gripper_position

    def set_gripper_g2_position(self, position, **kwargs):
        self.calls.append(("set_gripper_g2_position", position, kwargs))
        self.gripper_g2_position = position
        return 0

    def get_gripper_g2_position(self):
        self.calls.append(("get_gripper_g2_position",))
        return 0, self.gripper_g2_position

    def getset_tgpio_modbus_data(self, data):
        self.calls.append(("getset_tgpio_modbus_data", data))
        return 0, []

    def get_joint_states(self, is_radian=True, num=3):
        positions = np.arange(6, dtype=np.float64)
        velocities = np.zeros(6, dtype=np.float64)
        return 0, [positions, velocities, velocities]

    def disconnect(self):
        self.calls.append(("disconnect",))
        self.connected = False


def test_manual_mode_robot_enters_teaching_mode_without_sending_actions(monkeypatch, tmp_path):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)

    config = UFRobotConfig(
        id="test_manual_robot",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=0,
        manual_mode=True,
        teach_sensitivity=4,
    )
    robot = uf_robot_module.UFRobot(config)

    robot.connect()
    assert robot.is_connected
    assert arm.mode == 2
    assert ("set_teach_sensitivity", 4) in arm.calls
    assert robot._initial_point == arm.initial_point
    assert not any(call[0] == "set_servo_angle" for call in arm.calls)

    robot.reset_to_initial()
    reset_calls = [call for call in arm.calls if call[0] == "set_servo_angle"]
    assert reset_calls == [
        (
            "set_servo_angle",
            {
                "angle": arm.initial_point,
                "speed": 20,
                "is_radian": False,
                "wait": True,
            },
        )
    ]
    assert arm.mode == 2

    action = {"J1.pos": 1.0}
    assert robot.send_action(action) is action

    observation = robot.get_observation()
    assert observation["J1.pos"] == 0.0
    assert observation["J6.pos"] == 5.0

    robot.disconnect()
    assert not robot.is_connected
    assert arm.mode == 0
    assert ("disconnect",) in arm.calls

    call_count = len(arm.calls)
    robot.disconnect()
    assert len(arm.calls) == call_count


def test_robot_reset_uses_sdk_initial_point_in_normal_mode(monkeypatch, tmp_path):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)

    config = UFRobotConfig(
        id="test_normal_robot",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=0,
    )
    robot = uf_robot_module.UFRobot(config)
    assert not hasattr(config, "start_joints")
    assert not hasattr(config, "start_tcp_pose")
    robot.connect()

    reset_calls = [call for call in arm.calls if call[0] == "set_servo_angle"]
    assert reset_calls == [
        (
            "set_servo_angle",
            {
                "angle": arm.initial_point,
                "speed": 20,
                "is_radian": False,
                "wait": True,
            },
        )
    ]

    robot.disconnect()


def test_normal_mode_waits_for_gripper_to_open_before_control(monkeypatch, tmp_path):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    arm.gripper_position = 400
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)

    config = UFRobotConfig(
        id="test_wait_for_gripper",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=1,
    )
    robot = uf_robot_module.UFRobot(config)
    robot.connect()

    open_calls = [call for call in arm.calls if call[0] == "set_gripper_position"]
    assert open_calls == [("set_gripper_position", 800, {"wait": True})]
    assert robot._last_gripper_command == 0.0

    before_writes = len(
        [call for call in arm.calls if call[0] == "getset_tgpio_modbus_data"]
    )
    robot._send_gripper_action(0.0)
    after_writes = len(
        [call for call in arm.calls if call[0] == "getset_tgpio_modbus_data"]
    )
    assert after_writes == before_writes

    robot.disconnect()


@pytest.mark.parametrize("gripper_type, physical_position", [(1, 400), (2, 42)])
def test_keyboard_gripper_stop_reads_actual_position_without_monitor(monkeypatch, tmp_path,
                                                                    gripper_type, physical_position):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)
    robot = uf_robot_module.UFRobot(UFRobotConfig(
        calibration_dir=tmp_path, robot_ip=arm.robot_ip, robot_dof=6,
        gripper_type=gripper_type, gripper_current_monitor=False,
        gripper_command_threshold=1.0, gripper_command_interval_s=60.0,
        gripper_error_log_path=None,
    ))
    robot.connect()
    # The previous goal is fully closed, but the gripper is only halfway there.
    robot._last_gripper_command = 1.0
    robot._last_gripper_command_attempt_s = uf_robot_module.time.perf_counter()
    arm.gripper_position = physical_position
    arm.gripper_g2_position = physical_position
    api = "set_gripper_g2_position" if gripper_type == 2 else "set_gripper_position"
    before = len([call for call in arm.calls if call[0] == api])
    assert robot.stop_gripper_at_current_position() == pytest.approx(0.5)
    writes = [call for call in arm.calls if call[0] == api]
    assert len(writes) == before + 1
    assert writes[-1][1] == physical_position
    assert writes[-1][2]["wait"] is False
    assert robot._last_gripper_command == pytest.approx(0.5)
    robot.disconnect()


def test_gripper_rs485_commands_are_rate_limited(monkeypatch, tmp_path):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)
    config = UFRobotConfig(
        id="test_gripper_rate_limit",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=1,
        gripper_command_interval_s=0.1,
        gripper_error_log_path=None,
    )
    robot = uf_robot_module.UFRobot(config)
    robot.connect()

    robot._send_gripper_action(0.2)
    robot._send_gripper_action(0.4)
    writes = [call for call in arm.calls if call[0] == "set_gripper_position"]
    # One initialization/open write and one runtime write; the second runtime
    # target is coalesced by the RS485 rate limiter.
    assert len(writes) == 2
    assert writes[-1][2] == {
        "wait": False,
        "wait_motion": False,
        "check_baud": False,
        "check_err": False,
    }

    robot.disconnect()


@pytest.mark.parametrize("gripper_speed, expected_speed", [(-1, 50), (100, 100)])
def test_xarm_gripper_g2_uses_sdk_units_and_dedicated_api(monkeypatch, tmp_path, caplog, gripper_speed, expected_speed):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)
    config = UFRobotConfig(
        id="test_gripper_g2",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=2,
        gripper_speed=gripper_speed,
        gripper_force=50,
        gripper_command_interval_s=0.0,
        gripper_error_log_path=None,
    )
    robot = uf_robot_module.UFRobot(config)
    assert robot._gripper_g2_speed == expected_speed
    assert robot._gripper_param.speed == int(((expected_speed * 60) / 9.88235 + 140) / 0.4)
    robot.connect()

    g2_writes = [call for call in arm.calls if call[0] == "set_gripper_g2_position"]
    assert g2_writes == [
        (
            "set_gripper_g2_position",
            84,
            {
                "speed": expected_speed,
                "force": 50,
                "wait": True,
                "check_baud": False,
            },
        )
    ]

    robot._send_gripper_action(0.5)
    assert [call for call in arm.calls if call[0] == "set_gripper_g2_position"][-1] == (
        "set_gripper_g2_position",
        42,
        {
            "speed": expected_speed,
            "force": 50,
            "wait": False,
            "wait_motion": False,
            "check_baud": False,
            "check_err": False,
        },
    )
    assert not any(call[0] == "getset_tgpio_modbus_data" for call in arm.calls)

    observation = robot.get_observation()
    assert observation["gripper.pos"] == pytest.approx(0.5)

    arm.get_gripper_g2_position = lambda: (1, None)
    observation = robot.get_observation()
    assert observation["gripper.pos"] == pytest.approx(0.5)
    assert "get_gripper_g2_position" in caplog.text
    robot.disconnect()


def test_xarm_gripper_g2_rejects_legacy_speed_and_invalid_force(tmp_path):
    with pytest.raises(ValueError, match="15 and 225 mm/s"):
        UFRobotConfig(
            id="test_gripper_g2_speed",
            calibration_dir=tmp_path,
            robot_dof=6,
            gripper_type=2,
            gripper_speed=1500,
        )

    with pytest.raises(ValueError, match="between 1 and 100"):
        UFRobotConfig(
            id="test_gripper_g2_force",
            calibration_dir=tmp_path,
            robot_dof=6,
            gripper_type=2,
            gripper_force=0,
        )


def test_gello_configs_select_xarm_gripper_g2():
    config_dir = Path("config/gello")
    config_paths = sorted(config_dir.glob("*.yaml"))
    assert config_paths

    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["robot"]["gripper_type"] == 2, config_path
        assert 15 <= config["robot"]["gripper_speed"] <= 225, config_path
        gripper_force = config["robot"].get("gripper_force", -1)
        assert gripper_force == -1 or 1 <= gripper_force <= 100, config_path


def test_manual_mode_config_rejects_cartesian_control(tmp_path):
    with pytest.raises(ValueError, match="control_space='joint'"):
        UFRobotConfig(
            id="test_manual_robot",
            calibration_dir=tmp_path,
            robot_dof=6,
            control_space="cartesian",
            manual_mode=True,
        )


def test_manual_gripper_speed_is_configurable_and_non_negative(tmp_path):
    config = UFRobotConfig(
        id="test_manual_robot",
        calibration_dir=tmp_path,
        robot_dof=6,
        manual_mode=True,
        manual_gripper_speed=0.25,
    )
    assert config.manual_gripper_speed == 0.25

    with pytest.raises(ValueError, match="manual_gripper_speed"):
        UFRobotConfig(
            id="test_manual_robot",
            calibration_dir=tmp_path,
            robot_dof=6,
            manual_mode=True,
            manual_gripper_speed=-0.1,
        )


def test_manual_action_filters_non_action_observation_fields():
    observation = {
        "J1.pos": 1.0,
        "J1.vel": 2.0,
        "gripper.pos": 0.5,
        "camera": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    action_features = {"J1.pos": float, "gripper.pos": float}

    assert _manual_action_from_observation(observation, action_features) == {
        "J1.pos": 1.0,
        "gripper.pos": 0.5,
    }


def test_manual_gripper_keys_update_target_in_expected_direction_and_bounds():
    key_state = {"close": False, "open": False}

    _update_manual_gripper_key_state(type("Key", (), {"char": "C"})(), True, key_state)
    assert key_state == {"close": True, "open": False}
    assert _update_manual_gripper_target(0.5, key_state, speed=1.0, fps=10) == pytest.approx(0.6)

    _update_manual_gripper_key_state(type("Key", (), {"char": "C"})(), False, key_state)
    _update_manual_gripper_key_state(type("Key", (), {"char": "o"})(), True, key_state)
    assert _update_manual_gripper_target(0.05, key_state, speed=1.0, fps=10) == 0.0

    _update_manual_gripper_key_state(type("Key", (), {"char": "c"})(), True, key_state)
    assert _update_manual_gripper_target(0.99, key_state, speed=1.0, fps=10) == 0.99


def test_manual_mode_initializes_gripper_without_opening_and_sends_only_gripper(monkeypatch, tmp_path):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)

    config = UFRobotConfig(
        id="test_manual_gripper_robot",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=1,
        manual_mode=True,
        gripper_error_log_path=None,
    )
    robot = uf_robot_module.UFRobot(config)
    robot.connect()

    assert ("set_gripper_enable", True) in arm.calls
    assert ("set_gripper_mode", 0) in arm.calls
    assert ("set_gripper_speed", 5000) in arm.calls
    assert not any(call[0] == "set_gripper_position" for call in arm.calls)

    robot.send_action({"J1.pos": 1.0, "gripper.pos": 0.5})

    runtime_writes = [call for call in arm.calls if call[0] == "set_gripper_position"]
    assert len(runtime_writes) == 1
    assert runtime_writes[0][2]["wait_motion"] is False
    assert not any(call[0] == "set_servo_angle" for call in arm.calls)
    robot.disconnect()


def test_gripper_command_is_only_sent_after_target_changes(monkeypatch, tmp_path):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module

    arm = FakeXArm("192.168.1.245")
    monkeypatch.setattr(uf_robot_module, "XArmAPI", lambda robot_ip: arm)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)

    config = UFRobotConfig(
        id="test_gripper_command_threshold",
        calibration_dir=tmp_path,
        robot_ip=arm.robot_ip,
        robot_dof=6,
        control_space="joint",
        gripper_type=1,
        manual_mode=True,
        gripper_command_threshold=0.01,
        gripper_command_interval_s=0.0,
        gripper_error_log_path=None,
    )
    robot = uf_robot_module.UFRobot(config)
    robot.connect()

    robot.send_action({"gripper.pos": 0.5})
    robot.send_action({"gripper.pos": 0.505})
    robot.send_action({"gripper.pos": 0.52})

    writes = [call for call in arm.calls if call[0] == "set_gripper_position"]
    assert len(writes) == 2
    robot.disconnect()


def test_manual_record_config_has_no_teleop(monkeypatch):
    config_path = Path("config/manual_mode/xarm7_manual_record_config.yaml").resolve()
    monkeypatch.setattr(
        sys,
        "argv",
        ["record", "--config_path", str(config_path)],
    )

    config = get_cfg()

    assert config.robot.manual_mode is True
    assert config.robot.robot_dof == 7
    assert config.robot.manual_gripper_speed == 0.5
    assert config.teleop is None
    assert config.dataset.fps == 30


def test_prepare_dataset_root_leaves_new_root_for_lerobot_create(tmp_path):
    root = tmp_path / "nested" / "dataset"
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root), resume=False)

    _prepare_dataset_root(cfg)

    assert not root.exists()


def test_prepare_dataset_root_rejects_incomplete_resume(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root), resume=True)

    with pytest.raises(RuntimeError, match="meta/tasks.parquet"):
        _prepare_dataset_root(cfg)


def test_prepare_dataset_root_rejects_resume_when_root_is_missing(tmp_path):
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=tmp_path / "missing"), resume=True)

    with pytest.raises(RuntimeError, match="does not exist"):
        _prepare_dataset_root(cfg)


def test_prepare_dataset_root_resumes_complete_dataset_without_prompt(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    (root / "meta" / "tasks.parquet").write_bytes(b"tasks")
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"episodes")
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"data")
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root), resume=True)
    monkeypatch.setattr(record_module.sys.stdin, "isatty", lambda: True)

    _prepare_dataset_root(cfg)


def test_manual_record_loop_writes_actual_state_as_action(tmp_path):
    class FakeRobot:
        name = "fake_manual_robot"
        robot_type = name
        action_features = {"J1.pos": float, "J2.pos": float}
        observation_features = action_features

        def __init__(self):
            self.observation_count = 0
            self.sent_actions = []

        def get_observation(self):
            self.observation_count += 1
            value = float(self.observation_count)
            return {"J1.pos": value, "J2.pos": value + 1}

        def send_action(self, action):
            self.sent_actions.append(action.copy())
            return action

    robot = FakeRobot()
    action_pipeline, robot_pipeline, observation_pipeline = record_module.make_default_processors()
    features = record_module.combine_feature_dicts(
        record_module.aggregate_pipeline_dataset_features(
            pipeline=action_pipeline,
            initial_features=record_module.create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        record_module.aggregate_pipeline_dataset_features(
            pipeline=observation_pipeline,
            initial_features=record_module.create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=False,
        ),
    )
    dataset = LeRobotDataset.create(
        "test/manual-record",
        fps=30,
        features=features,
        root=tmp_path / "dataset",
        robot_type=robot.robot_type,
        use_videos=False,
    )

    record_module.record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=30,
        teleop_action_processor=action_pipeline,
        robot_action_processor=robot_pipeline,
        robot_observation_processor=observation_pipeline,
        dataset=dataset,
        control_time_s=0.001,
        single_task="test task",
        manual_mode=True,
    )

    assert dataset.episode_buffer["size"] == 1
    assert robot.sent_actions == [{"J1.pos": 2.0, "J2.pos": 3.0}]
    assert dataset.episode_buffer["action"][0].tolist() == [2.0, 3.0]

    dataset.save_episode()
    dataset.finalize()


def test_manual_recording_episode_resets_before_recording():
    class FakeRobot:
        def __init__(self):
            self.calls = []

        def reset_to_initial(self):
            self.calls.append("reset_to_initial")

    robot = FakeRobot()

    _prepare_recording_episode(robot, teleop=None, is_uf_teleop=False, manual_mode=True)

    assert robot.calls == ["reset_to_initial"]


def test_manual_record_loop_applies_keyboard_gripper_target():
    class FakeRobot:
        name = "fake_manual_robot"
        robot_type = name
        action_features = {"J1.pos": float, "gripper.pos": float}

        def __init__(self):
            self.sent_actions = []

        def get_observation(self):
            return {"J1.pos": 1.0, "gripper.pos": 0.5}

        def send_action(self, action):
            self.sent_actions.append(action.copy())
            return action

    robot = FakeRobot()
    action_pipeline, robot_pipeline, observation_pipeline = record_module.make_default_processors()

    record_module.record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=10,
        teleop_action_processor=action_pipeline,
        robot_action_processor=robot_pipeline,
        robot_observation_processor=observation_pipeline,
        control_time_s=0.001,
        manual_mode=True,
        manual_gripper_keys={"close": True, "open": False},
        manual_gripper_speed=1.0,
    )

    assert robot.sent_actions == [{"J1.pos": 1.0, "gripper.pos": 0.6}]
