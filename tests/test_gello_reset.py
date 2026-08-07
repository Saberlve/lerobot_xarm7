import numpy as np
import pytest

from lerobot_robot_ufactory.teleoperators.gello_teleop import gello_teleop as gello_module
from lerobot_robot_ufactory.scripts.uf_lerobot_record import _prepare_recording_episode


class FakeDriver:
    def __init__(self, positions, follow_commands=True):
        self.positions = np.asarray(positions, dtype=float)
        self.follow_commands = follow_commands
        self.commands = []

    def get_joints(self):
        return self.positions.copy()

    def set_joints(self, positions):
        self.commands.append(np.asarray(positions, dtype=float))
        if self.follow_commands:
            self.positions = self.commands[-1].copy()


class FakeGelloRobot:
    def __init__(self, follow_commands=True):
        self._driver = FakeDriver([0.0, 0.0, 0.0], follow_commands=follow_commands)
        self._joint_signs = np.array([1.0, -1.0, 1.0])
        self._joint_offsets = np.array([0.1, 0.2, 0.0])
        self.gripper_open_close = (0.0, 1.0)
        self._last_pos = object()
        self.torque_calls = []

    def set_torque_mode(self, enabled):
        self.torque_calls.append(enabled)


def make_teleop(robot):
    teleop = gello_module.GelloTeleop.__new__(gello_module.GelloTeleop)
    teleop._is_connected = True
    teleop._teleop_enabled = False
    teleop.dof = 2
    teleop.gello_agent = type("FakeAgent", (), {"_robot": robot})()
    return teleop


def patch_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(gello_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        gello_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )


def test_gello_reset_moves_to_robot_observation_and_disables_torque(monkeypatch):
    patch_clock(monkeypatch)
    robot = FakeGelloRobot()
    teleop = make_teleop(robot)

    teleop.reset_to_robot_observation(
        {"J1.pos": 0.3, "J2.pos": -0.4, "gripper.pos": 0.5}
    )

    assert robot.torque_calls == [True, False]
    assert np.allclose(robot._driver.positions, [0.4, 0.6, 0.5])
    assert robot._last_pos is None
    assert teleop._teleop_enabled is False


def test_gello_reset_failure_leaves_torque_off_and_teleop_disabled(monkeypatch):
    patch_clock(monkeypatch)
    robot = FakeGelloRobot(follow_commands=False)
    teleop = make_teleop(robot)

    with pytest.raises(RuntimeError, match="did not reach"):
        teleop.reset_to_robot_observation(
            {"J1.pos": 0.3, "J2.pos": -0.4, "gripper.pos": 0.5}
        )

    assert robot.torque_calls == [True, False]
    assert teleop._teleop_enabled is False


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

        def reset_to_robot_observation(self, obs):
            calls.append("gello_alignment")

    _prepare_recording_episode(FakeRobot(), FakeTeleop(), True, False)

    assert calls == [
        "teleop_False",
        "robot_reset",
        "observation",
        "gello_alignment",
        "teleop_True",
    ]
