import time
from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
    GelloTeleopConfig,
)
from lerobot_robot_ufactory.utils.realtime_teleop import (
    GRIPPER_CURRENT_FEEDBACK_KEY,
    RealtimeTeleopController,
    map_gripper_current_feedback,
)


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


def test_gello_force_feedback_is_default_off_and_requires_explicit_mapping():
    config = GelloTeleopConfig()
    assert config.gripper_force_feedback_enabled is False
    assert config.gripper_feedback_gain is None
    assert config.gripper_feedback_output_sign is None
    enabled = GelloTeleopConfig(
        gripper_current_control_enabled=True,
        gripper_current_limit_ma=20.0,
        gripper_force_feedback_enabled=True,
        gripper_feedback_bias_ma=0.0,
        gripper_feedback_deadzone_ma=10.0,
        gripper_feedback_input_limit_ma=1000.0,
        gripper_feedback_ema_beta=0.5,
        gripper_feedback_gain=0.01,
        gripper_feedback_output_sign=-1,
        gripper_feedback_output_limit_ma=10.0,
        gripper_feedback_slew_rate_ma_s=50.0,
        gripper_feedback_timeout_s=0.2,
    )
    assert enabled.gripper_force_feedback_enabled is True

    with pytest.raises(ValueError, match="current_control_enabled"):
        GelloTeleopConfig(
            gripper_force_feedback_enabled=True,
        )
    with pytest.raises(ValueError, match="requires explicit configuration"):
        GelloTeleopConfig(
            gripper_current_control_enabled=True,
            gripper_current_limit_ma=20.0,
            gripper_force_feedback_enabled=True,
        )
    with pytest.raises(ValueError, match="either -1 or 1"):
        GelloTeleopConfig(gripper_feedback_output_sign=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gripper_feedback_deadzone_ma", -1.0, "non-negative"),
        ("gripper_feedback_input_limit_ma", 0.0, "positive"),
        ("gripper_feedback_ema_beta", 1.0, r"\[0, 1\)"),
        ("gripper_feedback_gain", -0.1, "non-negative"),
        ("gripper_feedback_output_limit_ma", 0.0, "positive"),
        ("gripper_feedback_slew_rate_ma_s", 0.0, "positive"),
        ("gripper_feedback_timeout_s", 0.0, "positive"),
    ],
)
def test_gello_force_feedback_conditioning_config_is_validated(field, value, message):
    with pytest.raises(ValueError, match=message):
        GelloTeleopConfig(**{field: value})


def test_feedback_output_limit_cannot_exceed_phase2_limit():
    with pytest.raises(ValueError, match="cannot exceed"):
        GelloTeleopConfig(
            gripper_current_limit_ma=20.0,
            gripper_feedback_output_limit_ma=21.0,
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


def _current_sample(
    current_ma=100.0,
    *,
    available=True,
    stale=False,
    reason="fresh",
    error=None,
    age_s=0.01,
    gripper_state=2,
):
    return SimpleNamespace(
        current_ma=current_ma,
        available=available,
        stale=stale,
        reason=reason,
        error=error,
        age_s=age_s,
        gripper_state=gripper_state,
    )


@pytest.mark.parametrize(
    ("sample", "gain", "output_sign", "limit_ma", "expected_ma"),
    [
        (_current_sample(100.0), 0.1, 1, 50.0, 10.0),
        (_current_sample(-100.0), 0.1, 1, 50.0, -10.0),
        (_current_sample(100.0), 0.25, 1, 50.0, 25.0),
        (_current_sample(100.0), 0.1, -1, 50.0, -10.0),
        (_current_sample(1000.0), 0.5, 1, 20.0, 20.0),
        (_current_sample(-1000.0), 0.5, 1, 20.0, -20.0),
        (None, 0.1, 1, 20.0, 0.0),
        (_current_sample(stale=True), 0.1, 1, 20.0, 0.0),
        (_current_sample(available=False), 0.1, 1, 20.0, 0.0),
        (_current_sample(None), 0.1, 1, 20.0, 0.0),
        (_current_sample(error="monitor failed"), 0.1, 1, 20.0, 0.0),
        (
            _current_sample(available=False, reason="cache_busy"),
            0.1,
            1,
            20.0,
            0.0,
        ),
        (_current_sample(float("nan")), 0.1, 1, 20.0, 0.0),
        (_current_sample(float("inf")), 0.1, 1, 20.0, 0.0),
        (_current_sample(float("-inf")), 0.1, 1, 20.0, 0.0),
    ],
)
def test_gripper_current_feedback_mapping(
    sample, gain, output_sign, limit_ma, expected_ma
):
    assert map_gripper_current_feedback(
        sample,
        gain=gain,
        output_sign=output_sign,
        output_limit_ma=limit_ma,
    ) == pytest.approx(expected_ma)


class FeedbackTeleop(FakeTeleop):
    def __init__(self, *, enabled=True, fail_feedback=False, fail_start=False):
        super().__init__()
        self.config = SimpleNamespace(
            gripper_force_feedback_enabled=enabled,
            gripper_feedback_bias_ma=0.0,
            gripper_feedback_deadzone_ma=0.0,
            gripper_feedback_input_limit_ma=1000.0,
            gripper_feedback_ema_beta=0.0,
            gripper_feedback_gain=0.1,
            gripper_feedback_output_sign=-1,
            gripper_current_limit_ma=20.0,
            gripper_feedback_output_limit_ma=20.0,
            gripper_feedback_slew_rate_ma_s=10000.0,
            gripper_feedback_timeout_s=0.25,
        )
        self.fail_feedback = fail_feedback
        self.fail_start = fail_start
        self.feedback = []
        self.feedback_starts = 0
        self.feedback_stops = 0

    def start_feedback(self):
        self.feedback_starts += 1
        if self.fail_start:
            raise RuntimeError("simulated current-mode setup failure")

    def send_feedback(self, feedback):
        if self.fail_feedback:
            raise RuntimeError("simulated feedback failure")
        self.feedback.append(dict(feedback))

    def stop_feedback(self):
        self.feedback_stops += 1


class FeedbackRobot(FakeRobot):
    def __init__(self, sample):
        super().__init__()
        self.sample = sample
        self.sample_reads = 0

    def get_gripper_current_sample(self):
        self.sample_reads += 1
        return self.sample


def test_feedback_disabled_does_not_enter_current_output_or_read_cache():
    robot = FeedbackRobot(_current_sample(100.0))
    teleop = FeedbackTeleop(enabled=False)
    controller = RealtimeTeleopController(
        robot,
        teleop,
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
    )

    controller.start()
    time.sleep(0.02)
    controller.stop()

    assert robot.actions
    assert robot.sample_reads == 0
    assert teleop.feedback_starts == 0
    assert teleop.feedback == []
    assert teleop.feedback_stops == 0


def test_realtime_controller_connects_cache_mapping_to_feedback_and_stops_at_zero():
    robot = FeedbackRobot(_current_sample(100.0))
    teleop = FeedbackTeleop()
    controller = RealtimeTeleopController(
        robot,
        teleop,
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
    )

    controller.start()
    time.sleep(0.02)
    controller.stop()

    assert robot.actions
    assert robot.sample_reads > 0
    assert teleop.feedback_starts == 1
    assert any(
        item[GRIPPER_CURRENT_FEEDBACK_KEY] == pytest.approx(-10.0)
        for item in teleop.feedback
    )
    assert teleop.feedback[-1][GRIPPER_CURRENT_FEEDBACK_KEY] == 0.0
    assert teleop.feedback_stops >= 1


def test_feedback_exception_does_not_break_normal_position_teleop():
    robot = FeedbackRobot(_current_sample(100.0))
    teleop = FeedbackTeleop(fail_feedback=True)
    controller = RealtimeTeleopController(
        robot,
        teleop,
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
    )

    controller.start()
    time.sleep(0.02)
    controller.stop()

    assert robot.actions
    assert controller.latest_action() == robot.actions[-1]


def test_feedback_start_failure_does_not_break_normal_position_teleop():
    robot = FeedbackRobot(_current_sample(100.0))
    teleop = FeedbackTeleop(fail_start=True)
    controller = RealtimeTeleopController(
        robot,
        teleop,
        identity_action_processor,
        identity_action_processor,
        fps=100,
        initial_observation={"J1.pos": 0.0},
    )

    controller.start()
    time.sleep(0.02)
    controller.stop()

    assert robot.actions
    assert teleop.feedback_starts == 1
    assert controller.latest_action() == robot.actions[-1]
