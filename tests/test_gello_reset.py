import numpy as np
import pytest

from lerobot_robot_ufactory.teleoperators.gello_teleop import gello_teleop as gello_module
from lerobot_robot_ufactory.scripts.uf_lerobot_record import _prepare_recording_episode
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_adapter import (
    ContinuousDynamixelRobot,
)


class FakeDriver:
    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype=float)
        self.commands = []

    def get_joints(self):
        return self.positions.copy()

    def set_joints(self, positions):
        self.commands.append(np.asarray(positions, dtype=float))

    def close(self):
        pass


class FakeGelloRobot:
    def __init__(self):
        self._driver = FakeDriver([0.7, -0.2, 1.5])
        self._joint_signs = np.array([1.0, -1.0, 1.0])
        self._joint_offsets = np.array([0.1, 0.2, 0.0])
        self.gripper_open_close = (0.0, 1.0)
        self._last_pos = object()
        self.torque_calls = []

    def set_torque_mode(self, enabled):
        self.torque_calls.append(enabled)


def make_teleop(robot, align_gripper_to_current=True):
    teleop = gello_module.GelloTeleop.__new__(gello_module.GelloTeleop)
    teleop.id = "test_gello"
    teleop._is_connected = True
    teleop._teleop_enabled = False
    teleop._needs_alignment = True
    teleop._align_gripper_to_current = align_gripper_to_current
    teleop.dof = 2
    teleop.gello_agent = type("FakeAgent", (), {"_robot": robot})()
    return teleop


def test_gello_alignment_maps_current_pose_without_moving():
    robot = FakeGelloRobot()
    teleop = make_teleop(robot)

    teleop.reset_to_robot_observation(
        {"J1.pos": 0.3, "J2.pos": -0.4, "gripper.pos": 0.5}
    )

    assert robot.torque_calls == [False]
    assert robot._driver.commands == []
    assert np.allclose(robot._joint_offsets[:2], [0.4, -0.6])
    assert np.allclose(robot.gripper_open_close, [1.0, 2.0])
    assert robot._last_pos is None
    assert teleop._teleop_enabled is False
    assert teleop._needs_alignment is False


def test_gello_enable_requires_robot_observation():
    robot = FakeGelloRobot()
    teleop = make_teleop(robot)

    with pytest.raises(ValueError, match="Robot observation"):
        teleop.set_teleop_enabled(True)

    assert robot.torque_calls == []
    assert teleop._teleop_enabled is False


def test_fixed_gripper_endpoints_are_not_shifted_during_arm_alignment():
    robot = FakeGelloRobot()
    robot.gripper_open_close = (3.45, 2.72)
    teleop = make_teleop(robot, align_gripper_to_current=False)

    teleop.reset_to_robot_observation(
        {"J1.pos": 0.3, "J2.pos": -0.4, "gripper.pos": 0.5}
    )

    assert robot.gripper_open_close == (3.45, 2.72)
    assert robot._driver.commands == []


def test_dynamixel_arm_joint_is_continuous_across_encoder_wrap():
    robot = ContinuousDynamixelRobot(
        joint_ids=[1],
        joint_offsets=[0.0],
        joint_signs=[1],
        real=False,
    )
    robot._alpha = 1.0
    robot._driver._joint_angles = np.array([2 * np.pi - 0.05])
    before_wrap = robot.get_joint_state()[0]

    robot._driver._joint_angles = np.array([0.05])
    after_wrap = robot.get_joint_state()[0]

    assert after_wrap > before_wrap
    assert after_wrap - before_wrap == pytest.approx(0.1)


def test_gello_enable_after_pause_realigns_current_pose_before_output():
    robot = FakeGelloRobot()
    teleop = make_teleop(robot)

    teleop.set_teleop_enabled(
        True,
        {"J1.pos": 0.3, "J2.pos": -0.4, "gripper.pos": 0.5},
    )

    assert robot.torque_calls == [False]
    assert robot._driver.commands == []
    assert teleop._teleop_enabled is True

    teleop.set_teleop_enabled(False)
    robot._driver.positions = np.array([1.0, 0.5, 1.8])
    teleop.set_teleop_enabled(
        True,
        {"J1.pos": 0.1, "J2.pos": 0.2, "gripper.pos": 0.25},
    )

    assert np.allclose(robot._joint_offsets[:2], [0.9, 0.7])
    assert np.allclose(robot.gripper_open_close, [1.55, 2.55])
    assert robot._driver.commands == []


def test_gello_disconnect_closes_driver():
    robot = FakeGelloRobot()
    closed = []
    robot._driver.close = lambda: closed.append(True)
    teleop = make_teleop(robot)
    teleop.disconnect()

    assert closed == [True]
    assert robot.torque_calls == [False]
    assert teleop._is_connected is False


def test_recording_reset_disables_before_robot_and_enables_after_alignment():
    calls = []

    class FakeRobot:
        def reset_to_initial(self):
            calls.append("robot_reset")

        def get_observation(self):
            calls.append("observation")
            return {"J1.pos": 0.0}

    class FakeTeleop:
        def set_teleop_enabled(self, enabled, obs=None):
            calls.append(f"teleop_{enabled}")

    _prepare_recording_episode(FakeRobot(), FakeTeleop(), True, False)

    assert calls == [
        "teleop_False",
        "robot_reset",
        "observation",
        "teleop_True",
    ]
