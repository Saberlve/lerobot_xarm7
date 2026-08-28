from io import StringIO

import pytest

from lerobot_robot_ufactory.robots.uf_robot.uf_robot import G2CurrentSample
from lerobot_robot_ufactory.scripts.uf_read_g2_current import (
    format_dashboard,
    load_probe_settings,
    monitor_g2_current,
    render_ascii_chart,
)


def _sample(
    current_ma,
    timestamp_s,
    *,
    available=True,
    stale=False,
    state=2,
    reason=None,
    error=None,
):
    return G2CurrentSample(
        current_ma=current_ma,
        sample_monotonic_s=timestamp_s,
        gripper_state=state,
        age_s=0.01 if timestamp_s is not None else None,
        available=available,
        stale=stale,
        reason=reason,
        error=error,
    )


def test_load_probe_settings_reads_g2_monitor_configuration(tmp_path):
    config_path = tmp_path / "g2.yaml"
    config_path.write_text(
        """
robot:
  robot_ip: 192.168.1.245
  robot_dof: 7
  gripper_type: 2
  gripper_current_monitor_frequency_hz: 125
  gripper_current_stale_timeout_s: 0.4
""".strip(),
        encoding="utf-8",
    )

    settings = load_probe_settings(config_path)

    assert settings.robot_ip == "192.168.1.245"
    assert settings.robot_dof == 7
    assert settings.monitor_frequency_hz == 125
    assert settings.stale_timeout_s == 0.4


def test_load_probe_settings_requires_g2(tmp_path):
    config_path = tmp_path / "not_g2.yaml"
    config_path.write_text(
        "robot:\n  robot_ip: 192.168.1.245\n  robot_dof: 7\n  gripper_type: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gripper_type: 2"):
        load_probe_settings(config_path)


def test_ascii_chart_shows_signed_samples_and_zero_axis():
    chart = render_ascii_chart([-100, -50, 0, 50, 100], width=10, height=5)

    assert chart.count("*") == 5
    assert "      0 |" in chart
    assert "+----------" in chart


def test_dashboard_exposes_freshness_state_and_current():
    output = format_dashboard(
        _sample(-321, 10.0),
        [-100, -200, -321],
        unique_sample_rate_hz=20,
        chart_width=10,
    )

    assert "status=FRESH" in output
    assert "current=-321 mA" in output
    assert "state=2 (object detected while closing)" in output
    assert "fresh snapshots observed=20/s" in output


class _FakeRobot:
    def __init__(self, samples):
        self.samples = iter(samples)

    def get_gripper_current_sample(self):
        return next(self.samples)


class _FakeClock:
    def __init__(self):
        self.now_s = 0.0

    def __call__(self):
        return self.now_s

    def sleep(self, duration_s):
        self.now_s += duration_s


def test_monitor_collects_fresh_unique_samples_and_plain_output():
    clock = _FakeClock()
    robot = _FakeRobot(
        [
            _sample(-20, 1.0),
            _sample(0, 1.1, state=3),
            _sample(30, 1.2, state=3),
        ]
    )
    output = StringIO()

    stats = monitor_g2_current(
        robot,
        duration_s=0.3,
        display_hz=10,
        chart_width=10,
        dashboard=False,
        stream=output,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert stats.display_cycles == 3
    assert stats.fresh_cycles == 3
    assert stats.unique_samples == 3
    assert stats.minimum_ma == -20
    assert stats.maximum_ma == 30
    assert output.getvalue().count("available=True") == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"duration_s": -1}, "duration_s"),
        ({"display_hz": 0}, "display_hz"),
        ({"chart_width": 9}, "chart_width"),
    ],
)
def test_monitor_rejects_invalid_display_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        monitor_g2_current(_FakeRobot([]), **kwargs)
