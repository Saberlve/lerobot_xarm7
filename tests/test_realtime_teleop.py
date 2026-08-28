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


def test_gello_current_control_is_default_off_and_requires_a_safe_explicit_limit():
    config = GelloTeleopConfig()
    assert config.gripper_current_control_enabled is False
    assert config.gripper_current_limit_ma is None

    with pytest.raises(ValueError, match="explicitly configured"):
        GelloTeleopConfig(gripper_current_control_enabled=True)
    with pytest.raises(ValueError, match="no greater than 100"):
        GelloTeleopConfig(gripper_current_limit_ma=101.0)
    with pytest.raises(ValueError, match="gripper_id 8"):
        GelloTeleopConfig(
            gripper_id=7,
            gripper_current_control_enabled=True,
            gripper_current_limit_ma=20.0,
        )


def test_gello_keyboard_gripper_config_is_validated():
    with pytest.raises(ValueError, match="gripper_keyboard_step_mm"):
        GelloTeleopConfig(gripper_keyboard_step_mm=0.0)
    with pytest.raises(ValueError, match="gripper_keyboard_step_mm"):
        GelloTeleopConfig(gripper_keyboard_step_mm=-1.0)
    with pytest.raises(ValueError, match="gripper_keyboard_hold_delay_s"):
        GelloTeleopConfig(gripper_keyboard_hold_delay_s=-0.1)


def _make_keyboard_gripper_teleop(monkeypatch, step_mm=10.0, hold_delay_s=0.5):
    from types import SimpleNamespace

    import lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop as gello_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(
        gello_module, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    teleop = gello_module.GelloTeleop(
        GelloTeleopConfig(
            gripper_control_mode="keyboard",
            gripper_keyboard_step_mm=step_mm,
            gripper_keyboard_hold_delay_s=hold_delay_s,
        )
    )
    # speed 50 mm/s over a 100 mm stroke -> 0.5 normalized units per second.
    teleop.set_gripper_motion_parameters(speed_mm_s=50.0, stroke_mm=100.0)
    return teleop, clock


def test_keyboard_gripper_tap_applies_one_step(monkeypatch):
    teleop, clock = _make_keyboard_gripper_teleop(monkeypatch)

    teleop.set_gripper_keyboard_state(close=True, open=False)
    clock["now"] += 0.1
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(0.6)

    # Held but still within the hold delay: no further motion.
    clock["now"] += 0.2
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(0.6)


def test_keyboard_gripper_hold_continues_after_delay(monkeypatch):
    teleop, clock = _make_keyboard_gripper_teleop(monkeypatch)

    teleop.set_gripper_keyboard_state(close=True, open=False)
    clock["now"] += 0.1
    teleop._keyboard_gripper_action(0.5)  # -> 0.6 (step only)

    # 0.6 s after the press (> 0.5 s delay): continuous at 0.5/s, dt=0.5 s.
    clock["now"] += 0.5
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(0.85)

    # Release stops the motion.
    teleop.set_gripper_keyboard_state(close=False, open=False)
    clock["now"] += 0.5
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(0.85)


def test_keyboard_gripper_quick_tap_within_one_cycle_still_steps(monkeypatch):
    teleop, clock = _make_keyboard_gripper_teleop(monkeypatch)

    teleop._keyboard_gripper_action(0.5)  # initialize target
    # Press and release both before the next control cycle.
    teleop.set_gripper_keyboard_state(close=True, open=False)
    teleop.set_gripper_keyboard_state(close=False, open=False)
    clock["now"] += 0.05
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(0.6)


def test_keyboard_gripper_open_direction_and_clamp(monkeypatch):
    teleop, clock = _make_keyboard_gripper_teleop(monkeypatch)

    teleop.set_gripper_keyboard_state(close=False, open=True)
    clock["now"] += 0.1
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(0.4)

    # Holding close for a long time clamps at 1.0.
    teleop.set_gripper_keyboard_state(close=True, open=False)
    clock["now"] += 5.0
    assert teleop._keyboard_gripper_action(0.5) == pytest.approx(1.0)


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
