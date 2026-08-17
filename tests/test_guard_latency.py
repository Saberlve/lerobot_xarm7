import csv
import math
from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.scripts.uf_robot_teleop import (
    GuardLatencyTiming,
    TeleopConfig,
    _percentile,
    _validate_guard_latency_config,
    _write_guard_latency_timings,
)


def _config(**robot_overrides):
    robot_values = {
        "cameras": {"camera": object()},
        "control_space": "joint",
        "joint_command_mode": 1,
        "min_tcp_z_mm": 100.0,
    }
    robot_values.update(robot_overrides)
    return TeleopConfig(
        robot=SimpleNamespace(**robot_values),
        teleop=SimpleNamespace(),
        fps=60,
        guard_latency_experiment=True,
    )


def test_guard_latency_requires_guarded_servoj():
    _validate_guard_latency_config(_config())

    with pytest.raises(ValueError, match="control_space='joint'"):
        _validate_guard_latency_config(_config(control_space="cartesian"))
    with pytest.raises(ValueError, match="joint_command_mode=1"):
        _validate_guard_latency_config(_config(joint_command_mode=6))
    with pytest.raises(ValueError, match="min_tcp_z_mm"):
        _validate_guard_latency_config(_config(min_tcp_z_mm=None))


def test_guard_latency_config_rejects_invalid_rates():
    with pytest.raises(ValueError, match="fps"):
        TeleopConfig(robot=SimpleNamespace(cameras={}), teleop=SimpleNamespace(), fps=0)
    with pytest.raises(ValueError, match="experiment_duration_s"):
        TeleopConfig(
            robot=SimpleNamespace(cameras={}),
            teleop=SimpleNamespace(),
            experiment_duration_s=float("nan"),
        )


def test_percentile_ignores_non_finite_values():
    assert _percentile([1.0, 2.0, float("nan"), 3.0, 4.0], 50) == pytest.approx(2.5)
    assert math.isnan(_percentile([], 95))


def test_write_guard_latency_timings_creates_parseable_csv(tmp_path):
    sample = GuardLatencyTiming(
        iteration=0,
        elapsed_s=0.0,
        period_ms=None,
        gello_read_ms=1.0,
        safety_guard_ms=2.0,
        guard_path="fk_safe",
        servo_j_ms=3.0,
        send_action_ms=6.0,
        work_ms=7.0,
        cycle_ms=16.7,
    )

    output_path = _write_guard_latency_timings([sample], str(tmp_path), fps=60)

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["period_ms"] == ""
    assert rows[0]["guard_path"] == "fk_safe"
    assert float(rows[0]["safety_guard_ms"]) == 2.0
    assert float(rows[0]["servo_j_ms"]) == 3.0
