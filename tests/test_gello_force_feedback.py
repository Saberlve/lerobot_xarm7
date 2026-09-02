import threading
from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import (
    GRIPPER_CURRENT_FEEDBACK_KEY,
    GelloTeleop,
)
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
    GelloTeleopConfig,
)


class FakeGripperCurrentRobot:
    def __init__(self):
        self.calls = []
        self.write_seen = threading.Event()
        self.fail_write = False

    def enable_gripper_current_mode(self, limit_ma):
        self.calls.append(("enable", limit_ma))
        return "ID8-info"

    def write_gripper_current_ma(self, current_ma):
        self.calls.append(("write", current_ma))
        self.write_seen.set()
        if self.fail_write:
            raise RuntimeError("simulated ID8 write failure")
        return current_ma

    def zero_gripper_current(self):
        self.calls.append(("zero",))

    def disable_gripper_current_mode(self):
        self.calls.append(("disable",))


def make_feedback_teleop(*, enabled=True, timeout_s=0.1):
    config = GelloTeleopConfig(
        gripper_current_control_enabled=enabled,
        gripper_current_limit_ma=20.0 if enabled else None,
        gripper_force_feedback_enabled=enabled,
        gripper_feedback_bias_ma=0.0 if enabled else None,
        gripper_feedback_deadzone_ma=0.0 if enabled else None,
        gripper_feedback_input_limit_ma=1000.0 if enabled else None,
        gripper_feedback_ema_beta=0.0 if enabled else None,
        gripper_feedback_gain=0.01 if enabled else None,
        gripper_feedback_output_sign=1 if enabled else None,
        gripper_feedback_output_limit_ma=20.0 if enabled else None,
        gripper_feedback_slew_rate_ma_s=100.0 if enabled else None,
        gripper_feedback_timeout_s=timeout_s if enabled else None,
    )
    teleop = GelloTeleop(config)
    robot = FakeGripperCurrentRobot()
    teleop.gello_agent = SimpleNamespace(_robot=robot)
    teleop._is_connected = True
    return teleop, robot


def test_send_feedback_is_async_clamped_and_stop_zeros_before_disable():
    teleop, robot = make_feedback_teleop()

    assert teleop.start_feedback() == "ID8-info"
    teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 1000.0})
    assert robot.write_seen.wait(timeout=1.0)
    teleop.stop_feedback()

    assert ("enable", 20.0) in robot.calls
    assert ("write", 20.0) in robot.calls
    assert robot.calls[-2:] == [("zero",), ("disable",)]


@pytest.mark.parametrize("unsafe_value", [None, float("nan"), float("inf")])
def test_send_feedback_converts_invalid_target_to_zero(unsafe_value):
    teleop, robot = make_feedback_teleop()
    teleop.start_feedback()

    teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: unsafe_value})
    assert robot.write_seen.wait(timeout=1.0)
    teleop.stop_feedback()

    assert ("write", 0.0) in robot.calls


def test_send_feedback_disabled_never_enters_current_output():
    teleop, robot = make_feedback_teleop(enabled=False)

    assert teleop.start_feedback() is None
    teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 10.0})
    teleop.stop_feedback()

    assert not [call for call in robot.calls if call[0] in ("enable", "write")]


def test_feedback_write_failure_zeros_and_disables_id8():
    teleop, robot = make_feedback_teleop()
    robot.fail_write = True
    teleop.start_feedback()

    teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 10.0})
    assert robot.write_seen.wait(timeout=1.0)
    assert teleop._feedback_thread is not None
    teleop._feedback_thread.join(timeout=1.0)

    assert teleop._feedback_output_active is False
    assert robot.calls[-2:] == [("zero",), ("disable",)]


def test_feedback_command_watchdog_does_not_hold_last_nonzero():
    teleop, robot = make_feedback_teleop(timeout_s=0.02)
    teleop.start_feedback()
    teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 10.0})
    assert robot.write_seen.wait(timeout=1.0)

    deadline = threading.Event()
    deadline.wait(timeout=0.2)
    teleop.stop_feedback()

    assert ("write", 10.0) in robot.calls
    assert ("write", 0.0) in robot.calls
