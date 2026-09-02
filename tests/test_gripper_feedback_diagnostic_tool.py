import pytest

from lerobot_robot_ufactory.scripts.uf_test_gripper_force_feedback import (
    calculate_stage_statistics,
    suggest_idle_calibration,
)


def test_stage_statistics_include_requested_metrics():
    stats = calculate_stage_statistics([-2.0, 0.0, 4.0])
    assert stats.count == 3
    assert stats.mean_ma == pytest.approx(2.0 / 3.0)
    assert stats.median_ma == 0.0
    assert stats.minimum_ma == -2.0
    assert stats.maximum_ma == 4.0
    assert stats.peak_abs_ma == 4.0
    assert stats.std_ma > 0.0


def test_idle_suggestion_uses_only_open_idle_and_does_not_mutate_samples():
    stage_values = {
        "open_idle": [9.0, 10.0, 11.0],
        "holding_soft_object": [500.0, 600.0],
    }
    original = {name: list(values) for name, values in stage_values.items()}

    bias_ma, deadzone_ma = suggest_idle_calibration(stage_values)

    assert bias_ma == 10.0
    assert deadzone_ma >= 1.0
    assert stage_values == original


def test_idle_suggestion_requires_explicit_open_idle_samples():
    assert suggest_idle_calibration({"holding_soft_object": [500.0]}) is None
