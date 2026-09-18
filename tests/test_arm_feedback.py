import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.utils.arm_feedback import (
    ArmFeedbackConfig,
    ArmFeedbackProcessor,
    ArmFeedbackSample,
    ft_wrench_to_joint_torque,
)
from lerobot_robot_ufactory.utils.arm_feedback_runtime import ArmFeedbackWorker, XArmFeedbackSource


def config(**kwargs):
    values = dict(
        enabled=True,
        enabled_joints=(True,) * 7,
        ema_alpha=(1,) * 7,
        gain_ma_per_unit=(1,) * 7,
        input_limit=(100,) * 7,
        current_limit_ma=(50,) * 7,
        slew_rate_ma_s=(10000,) * 7,
        stale_timeout_ms=1000,
    )
    values.update(kwargs)
    return ArmFeedbackConfig(**values)


def sample(values, timestamp=1_000_000_000):
    return ArmFeedbackSample(timestamp, np.asarray(values), np.asarray(values))


def process_twice(c, values, velocity=None):
    p = ArmFeedbackProcessor(c)
    velocity = np.zeros(7) if velocity is None else velocity
    assert np.all(p.process(sample(values), velocity, 1_000_000_000).command_current_ma == 0)
    return p.process(sample(values, 1_100_000_000), velocity, 1_100_000_000)


def test_zero_and_independent_gains_signs():
    assert not process_twice(config(), np.zeros(7)).command_current_ma.any()
    c = config(gain_ma_per_unit=tuple(range(1, 8)), sign=(-1, 1, -1, 1, -1, 1, -1))
    values = np.array([1, -1, 1, -1, 1, -1, 1])
    np.testing.assert_allclose(process_twice(c, values).command_current_ma, -np.arange(1, 8))


def test_deadzone_preserves_sign_and_bias():
    c = config(bias=(1,) * 7, deadzone=(2,) * 7)
    r = process_twice(c, [1, 2, 3, 4, -2, -3, 0])
    np.testing.assert_allclose(r.command_current_ma, [0, 0, 0, 1, -1, -2, 0])


def test_input_output_clamp_and_mask():
    c = config(
        input_limit=(2,) * 7,
        gain_ma_per_unit=(10,) * 7,
        current_limit_ma=(3,) * 7,
        enabled_joints=(True, False, True, False, True, False, True),
    )
    r = process_twice(c, [100, 100, -100, 100, 100, 100, 100])
    assert r.clamped
    np.testing.assert_allclose(r.command_current_ma, [3, 0, -3, 0, 3, 0, 3])


def test_ema_only_on_new_sample():
    p = ArmFeedbackProcessor(config(ema_alpha=(0.5,) * 7))
    s = sample(np.ones(7) * 8)
    p.process(s, np.zeros(7), s.timestamp_ns)
    r = p.process(s, np.zeros(7), s.timestamp_ns + 100_000_000)
    np.testing.assert_allclose(r.command_current_ma, 4)
    r = p.process(
        replace(s, timestamp_ns=s.timestamp_ns + 200_000_000),
        np.zeros(7),
        s.timestamp_ns + 200_000_000,
    )
    np.testing.assert_allclose(r.command_current_ma, 6)


def test_damping_opposes_motor_velocity_independently_of_feedback_sign():
    p = ArmFeedbackProcessor(config(damping_ma_per_rad_s=(2,) * 7, sign=(-1,) * 7))
    p.process(sample(np.zeros(7)), np.ones(7), 1_000_000_000, motor_signs=(-1,) * 7)
    r = p.process(sample(np.zeros(7)), np.ones(7), 1_100_000_000, motor_signs=(-1,) * 7)
    np.testing.assert_allclose(r.command_current_ma, 2)


def test_slew_uses_elapsed_time():
    r = process_twice(config(slew_rate_ma_s=(2,) * 7), np.ones(7) * 20)
    np.testing.assert_allclose(r.command_current_ma, 0.2)


@pytest.mark.parametrize("case", ["stale", "timeout", "nan", "inf", "future", "error", "shape"])
def test_fail_closed_and_reset(case):
    p = ArmFeedbackProcessor(config(stale_timeout_ms=200))
    p.process(sample(np.ones(7)), np.zeros(7), 1_000_000_000)
    s, now = sample(np.ones(7), 1_100_000_000), 1_100_000_000
    if case == "stale":
        now = 2_000_000_000
    elif case == "timeout":
        s, now = sample(np.ones(7), 2_000_000_000), 2_000_000_000
    elif case in ("nan", "inf"):
        s = sample([float(case)] * 7)
    elif case == "future":
        s = sample(np.ones(7), 3_000_000_000)
    elif case == "error":
        s = replace(s, error="read failed")
    elif case == "shape":
        s = sample([1, 2])
    r = p.process(s, np.zeros(7), now)
    assert r.fault and not r.command_current_ma.any()
    assert p.last_ns is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gain_ma_per_unit": [1]},
        {"sign": [0] * 7},
        {"ema_alpha": [1.1] * 7},
        {"bias": [float("nan")] * 7},
        {"current_limit_ma": [101] * 7},
        {"enabled_joints": [1] * 7},
        {"sampling_hz": 0},
        {"observe_only": "false"},
        {"damping_ma_per_rad_s": [-1] * 7},
        {"baseline": [True] * 7},
        {"enabled": True, "observe_only": False},
        {"source": "unknown"},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ArmFeedbackConfig(**kwargs)


def test_ft_jacobian_mm_conversion_and_sensor_rotation():
    class Model:
        def forward_matrix(self, q):
            t = np.eye(4)
            t[2, 3] = q[0] * 1000  # 1 m/rad in world z
            return t

    t = np.eye(4)
    t[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    result = ft_wrench_to_joint_torque(Model(), np.zeros(7), [2, 0, 0, 0, 0, 0], t, True)
    np.testing.assert_allclose(result, [-2, 0, 0, 0, 0, 0, 0], atol=1e-9)


def test_ft_moment_and_sensor_offset():
    class Model:
        def forward_matrix(self, q):
            c, s = np.cos(q[0]), np.sin(q[0])
            return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    t = np.eye(4)
    t[0, 3] = 1000
    r = ft_wrench_to_joint_torque(Model(), np.zeros(7), [0, 2, 0, 0, 0, 3], t)
    np.testing.assert_allclose(r, [5, 0, 0, 0, 0, 0, 0], atol=1e-8)


def test_source_fixed_baseline_and_request_age():
    api = SimpleNamespace(get_joint_states=lambda **kw: (0, [[0] * 7, [0] * 7, [5] * 7]))
    source = XArmFeedbackSource("unused", config(baseline=(3,) * 7), api=api)
    a, b = source.read_once(), source.read_once()
    np.testing.assert_allclose(a.raw_joint_effort, 5)
    np.testing.assert_allclose(a.estimated_contact_torque, 2)
    np.testing.assert_allclose(b.estimated_contact_torque, 2)
    assert a.unit == "sdk_effort_unit" and a.read_latency_ms >= 0


def test_source_nonzero_sdk_status_rejected():
    api = SimpleNamespace(get_joint_states=lambda **kw: (9, []))
    with pytest.raises(RuntimeError):
        XArmFeedbackSource("unused", config(), api=api).read_once()


class MockAdapter:
    def __init__(self):
        self.commands = []
        self.infos = []
        self.disabled = False
        self.enables = 0

    def discover(self):
        return []

    def enable(self):
        self.enables += 1

    def write(self, values, **kwargs):
        self.commands.append(values.copy())
        return values.copy()

    def disable(self):
        self.disabled = True


@pytest.mark.parametrize("observe", [True, False])
def test_source_processor_adapter_integration(observe):
    c = config(observe_only=observe, baseline_verified=True, sign_verified=True, sign=(-1,) * 7)
    adapter = MockAdapter()
    source = SimpleNamespace(latest=sample(np.ones(7), time.monotonic_ns()))

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.0

    worker = ArmFeedbackWorker(c, source, adapter, state)
    worker.tick()
    time.sleep(0.005)
    source.latest = sample(np.ones(7), time.monotonic_ns())
    row = worker.tick()
    assert row["hypothetical_current_ma_1"] < 0
    if observe:
        assert not adapter.commands and row["command_current_ma_1"] == 0
    else:
        assert adapter.commands[-1].shape == (7,)
        assert np.all(adapter.commands[-1] < 0)


def test_worker_stale_latches_fault_disables_without_slew():
    source = SimpleNamespace(latest=sample(np.ones(7), 1))
    adapter = MockAdapter()

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.0

    worker = ArmFeedbackWorker(config(), source, adapter, state)
    row = worker.tick()
    assert row["stale"] and worker.fault and worker.stop_event.is_set()
    assert adapter.disabled and not adapter.commands
