from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.utils.realtime_teleop import GripperFeedbackProcessor


def make_sample(
    current_ma=100.0,
    *,
    age_s=0.01,
    available=True,
    stale=False,
    reason=None,
    error=None,
):
    return SimpleNamespace(
        current_ma=current_ma,
        sample_monotonic_s=10.0,
        gripper_state=2,
        age_s=age_s,
        available=available,
        stale=stale,
        reason=reason,
        error=error,
    )


def make_processor(**overrides):
    parameters = {
        "bias_ma": 0.0,
        "deadzone_ma": 0.0,
        "input_limit_ma": 1000.0,
        "ema_beta": 0.0,
        "gain": 1.0,
        "output_sign": 1,
        "output_limit_ma": 100.0,
        "slew_rate_ma_s": 10000.0,
        "timeout_s": 1.0,
    }
    parameters.update(overrides)
    return GripperFeedbackProcessor(**parameters)


def test_bias_compensation():
    diagnostic = make_processor(bias_ma=25.0).process(
        make_sample(125.0), now_monotonic_s=1.0
    )
    assert diagnostic.bias_corrected_ma == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("current_ma", "expected_ma"),
    [(20.0, 15.0), (-20.0, -15.0), (5.0, 0.0), (-5.0, 0.0)],
)
def test_symmetric_deadzone_preserves_sign(current_ma, expected_ma):
    diagnostic = make_processor(deadzone_ma=5.0).process(
        make_sample(current_ma), now_monotonic_s=1.0
    )
    assert diagnostic.deadzone_output_ma == pytest.approx(expected_ma)


def test_input_clamp_happens_before_deadzone():
    diagnostic = make_processor(input_limit_ma=50.0, deadzone_ma=10.0).process(
        make_sample(100.0), now_monotonic_s=1.0
    )
    assert diagnostic.input_clamped_ma == pytest.approx(50.0)
    assert diagnostic.deadzone_output_ma == pytest.approx(40.0)


def test_ema_uses_only_valid_samples():
    processor = make_processor(ema_beta=0.5)
    first = processor.process(make_sample(100.0), now_monotonic_s=1.0)
    second = processor.process(make_sample(100.0), now_monotonic_s=1.1)
    assert first.filtered_ma == pytest.approx(50.0)
    assert second.filtered_ma == pytest.approx(75.0)


def test_gain_and_output_sign():
    diagnostic = make_processor(gain=0.2, output_sign=-1).process(
        make_sample(100.0), now_monotonic_s=1.0
    )
    assert diagnostic.target_ma == pytest.approx(-20.0)


def test_slew_rate_uses_monotonic_dt_and_starts_from_zero():
    processor = make_processor(slew_rate_ma_s=10.0)
    first = processor.process(make_sample(100.0), now_monotonic_s=1.0)
    second = processor.process(make_sample(100.0), now_monotonic_s=1.1)
    assert first.command_ma == 0.0
    assert second.command_ma == pytest.approx(1.0)


def test_final_output_clamp():
    processor = make_processor(output_limit_ma=5.0, slew_rate_ma_s=10000.0)
    processor.process(make_sample(100.0), now_monotonic_s=1.0)
    diagnostic = processor.process(make_sample(100.0), now_monotonic_s=1.1)
    assert diagnostic.command_ma == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        (make_sample(stale=True, available=False, reason="stale"), "stale"),
        (make_sample(available=False, reason="unavailable"), "unavailable"),
        (make_sample(available=False, reason="cache_busy"), "cache_busy"),
        (make_sample(error="monitor failed", available=False), "monitor_error"),
        (make_sample(float("nan")), "current_not_finite"),
        (make_sample(float("inf")), "current_not_finite"),
        (make_sample(age_s=0.5), "sample_timeout"),
    ],
)
def test_invalid_sample_immediately_zeros_and_clears_state(sample, reason):
    processor = make_processor(timeout_s=0.25)
    processor.process(make_sample(100.0), now_monotonic_s=1.0)
    processor.process(make_sample(100.0), now_monotonic_s=1.1)

    diagnostic = processor.process(sample, now_monotonic_s=1.2)

    assert diagnostic.command_ma == 0.0
    assert diagnostic.filtered_ma is None
    assert diagnostic.feedback_active is False
    assert diagnostic.reason == reason


def test_recovery_after_stale_does_not_reuse_old_filter_or_command():
    processor = make_processor(ema_beta=0.5, slew_rate_ma_s=10.0)
    processor.process(make_sample(100.0), now_monotonic_s=1.0)
    processor.process(make_sample(100.0), now_monotonic_s=1.1)
    processor.process(
        make_sample(stale=True, available=False, reason="stale"),
        now_monotonic_s=1.2,
    )

    recovered = processor.process(make_sample(100.0), now_monotonic_s=1.3)

    assert recovered.filtered_ma == pytest.approx(50.0)
    assert recovered.command_ma == 0.0


def test_abnormally_large_dt_restarts_from_zero():
    processor = make_processor(timeout_s=0.25, ema_beta=0.5)
    processor.process(make_sample(100.0), now_monotonic_s=1.0)
    processor.process(make_sample(100.0), now_monotonic_s=1.1)

    recovered = processor.process(make_sample(100.0), now_monotonic_s=2.0)

    assert recovered.reason == "valid_after_gap"
    assert recovered.filtered_ma == pytest.approx(50.0)
    assert recovered.command_ma == 0.0


def test_reset_and_reconnect_start_with_cleared_state():
    processor = make_processor(ema_beta=0.5)
    processor.process(make_sample(100.0), now_monotonic_s=1.0)
    processor.process(make_sample(100.0), now_monotonic_s=1.1)
    processor.reset("disconnect")

    reset_state = processor.get_diagnostic()
    assert reset_state.command_ma == 0.0
    assert reset_state.filtered_ma is None
    assert reset_state.reason == "disconnect"

    reconnected = processor.process(make_sample(100.0), now_monotonic_s=2.0)
    assert reconnected.filtered_ma == pytest.approx(50.0)
    assert reconnected.command_ma == 0.0


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("deadzone_ma", -1.0),
        ("input_limit_ma", 0.0),
        ("ema_beta", 1.0),
        ("gain", -1.0),
        ("output_sign", 0),
        ("output_limit_ma", 0.0),
        ("slew_rate_ma_s", 0.0),
        ("timeout_s", 0.0),
    ],
)
def test_processor_rejects_invalid_configuration(parameter, value):
    with pytest.raises(ValueError):
        make_processor(**{parameter: value})
