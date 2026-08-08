import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig
from lerobot_robot_ufactory.scripts import uf_lerobot_record as record_module
from lerobot_robot_ufactory.scripts.uf_lerobot_record import (
    _manual_action_from_observation,
    _update_manual_gripper_key_state,
    _update_manual_gripper_target,
    _prepare_dataset_root,
    _prepare_recording_episode,
    get_cfg,
)


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
                "speed": 60,
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
    assert arm.mode == 0
    assert ("disconnect",) in arm.calls


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
                "speed": 60,
                "is_radian": False,
                "wait": True,
            },
        )
    ]

    robot.disconnect()


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
    )
    robot = uf_robot_module.UFRobot(config)
    robot.connect()

    assert ("set_gripper_enable", True) in arm.calls
    assert ("set_gripper_mode", 0) in arm.calls
    assert ("set_gripper_speed", 5000) in arm.calls
    assert not any(call[0] == "set_gripper_position" for call in arm.calls)

    robot.send_action({"J1.pos": 1.0, "gripper.pos": 0.5})

    assert any(call[0] == "getset_tgpio_modbus_data" for call in arm.calls)
    assert not any(call[0] == "set_servo_angle" for call in arm.calls)
    robot.disconnect()


def test_manual_record_config_has_no_teleop(monkeypatch):
    config_path = Path("config/manual_mode/xarm7_manual_record_config.yaml").resolve()
    monkeypatch.setattr(
        sys,
        "argv",
        ["uf-lerobot-record", "--config_path", str(config_path)],
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
