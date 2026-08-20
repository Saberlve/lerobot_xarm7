import time

import pytest

from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
    GelloTeleopConfig,
)
from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController


class FakeTeleop:
    def __init__(self):
        self.count = 0

    def get_action(self):
        self.count += 1
        return {"J1.pos": float(self.count)}


class FakeRobot:
    def __init__(self):
        self.actions = []

    def send_action(self, action):
        self.actions.append(dict(action))
        return action


def identity_action_processor(value):
    return value[0]


def test_gello_realtime_control_fps_defaults_to_30():
    assert GelloTeleopConfig().realtime_control_fps == 30


def test_gello_realtime_control_fps_must_be_positive():
    with pytest.raises(ValueError, match="realtime_control_fps"):
        GelloTeleopConfig(realtime_control_fps=0)


def test_gello_gripper_control_mode_is_validated():
    assert GelloTeleopConfig(gripper_control_mode="keyboard").gripper_control_mode == "keyboard"
    with pytest.raises(ValueError, match="gripper_control_mode"):
        GelloTeleopConfig(gripper_control_mode="invalid")


def test_realtime_controller_sends_without_waiting_for_observation_owner():
    robot = FakeRobot()
    controller = RealtimeTeleopController(
        robot,
        FakeTeleop(),
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
    )

    controller.start()
    time.sleep(0.06)
    controller.heartbeat()
    controller.stop()

    assert len(robot.actions) >= 4
    assert controller.latest_action() == robot.actions[-1]


def test_action_at_never_selects_a_future_command():
    controller = RealtimeTeleopController(
        FakeRobot(),
        FakeTeleop(),
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
    )
    controller.start()
    time.sleep(0.035)
    controller.stop()

    with controller._lock:
        history = list(controller._action_history)
    assert len(history) >= 2
    sample_time = (history[0][0] + history[1][0]) / 2
    assert controller.action_at(sample_time) == history[0][1]
    action, sent_at = controller.action_sample_at(sample_time)
    assert action == history[0][1]
    assert sent_at == history[0][0]


def test_realtime_controller_records_action_timing_when_enabled():
    controller = RealtimeTeleopController(
        FakeRobot(),
        FakeTeleop(),
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
        record_timing=True,
    )
    controller.start()
    time.sleep(0.025)
    controller.stop()

    timings = controller.action_timings()
    assert timings
    for index, timing in enumerate(timings):
        assert timing["action_index"] == index
        assert timing["gello_read_start_ns"] <= timing["gello_read_end_ns"]
        assert timing["gello_read_end_ns"] <= timing["command_send_end_ns"]


def test_realtime_controller_propagates_send_failures():
    class FailingRobot:
        def send_action(self, action):
            raise ValueError("servo failed")

    controller = RealtimeTeleopController(
        FailingRobot(),
        FakeTeleop(),
        identity_action_processor,
        identity_action_processor,
        fps=60,
        initial_observation={},
    )

    with pytest.raises(RuntimeError, match="Realtime joint control thread failed"):
        controller.start()
