import sys
from pathlib import Path

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig
from lerobot_robot_ufactory.scripts import uf_lerobot_record as record_module
from lerobot_robot_ufactory.scripts.uf_lerobot_record import (
    _manual_action_from_observation,
    get_cfg,
)


class FakeXArm:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.connected = True
        self.axis = 6
        self.error_code = 0
        self.mode = 0
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

    def get_err_warn_code(self):
        return 0, [0, 0]

    def set_linear_spd_limit_factor(self, factor):
        self.calls.append(("set_linear_spd_limit_factor", factor))
        return 0

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
        start_joints=(),
        manual_mode=True,
        teach_sensitivity=4,
    )
    robot = uf_robot_module.UFRobot(config)

    robot.connect()
    assert arm.mode == 2
    assert ("set_teach_sensitivity", 4) in arm.calls

    action = {"J1.pos": 1.0}
    assert robot.send_action(action) is action
    assert not any(call[0] == "set_servo_angle" for call in arm.calls)

    observation = robot.get_observation()
    assert observation["J1.pos"] == 0.0
    assert observation["J6.pos"] == 5.0

    robot.disconnect()
    assert arm.mode == 0
    assert ("disconnect",) in arm.calls


def test_manual_mode_config_rejects_cartesian_control(tmp_path):
    with pytest.raises(ValueError, match="control_space='joint'"):
        UFRobotConfig(
            id="test_manual_robot",
            calibration_dir=tmp_path,
            robot_dof=6,
            control_space="cartesian",
            manual_mode=True,
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
    assert config.teleop is None
    assert config.dataset.fps == 30


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
