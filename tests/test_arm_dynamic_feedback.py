import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.utils.arm_dynamic_feedback import (
    DynamicExternalTorqueEstimator,
)
from lerobot_robot_ufactory.utils.arm_feedback import ArmFeedbackConfig, ArmFeedbackSample
from lerobot_robot_ufactory.utils.arm_feedback_runtime import (
    ArmFeedbackWorker,
    XArmFeedbackSource,
)


def test_estimator_c1_math_is_effort_minus_baseline():
    baseline = np.arange(7, dtype=float) * 0.5
    estimator = DynamicExternalTorqueEstimator(baseline)
    effort = np.array([1.0, -2.0, 3.5, 0.0, 0.25, -0.75, 2.0])
    tau = estimator.estimate(np.zeros(7), np.zeros(7), effort)
    assert np.allclose(tau, effort - baseline)


def test_estimator_is_pure_and_validates_shapes():
    estimator = DynamicExternalTorqueEstimator(np.zeros(7))
    args = (np.zeros(7), np.zeros(7), np.ones(7))
    assert np.array_equal(estimator.estimate(*args), estimator.estimate(*args))
    with pytest.raises(ValueError):
        estimator.estimate(np.zeros(6), np.zeros(7), np.ones(7))
    with pytest.raises(ValueError):
        estimator.estimate(np.zeros(7), np.zeros(7), np.ones(6))
    with pytest.raises(ValueError):
        DynamicExternalTorqueEstimator(np.zeros(6))


def test_dynamic_mode_config_field():
    assert ArmFeedbackConfig(enabled=False).dynamic_mode is False
    config = ArmFeedbackConfig(enabled=False, dynamic_mode=True)
    assert config.dynamic_mode is True
    with pytest.raises(ValueError):
        ArmFeedbackConfig(dynamic_mode=1)


class FakeDynamicArm:
    """XArmAPI stand-in with report-cache angles/speeds/torque."""

    def __init__(self):
        self._arm = SimpleNamespace(_last_update_cmdnum_time=time.monotonic())
        self.rpc_calls = 0

    @property
    def joints_torque(self):
        return list(np.arange(7, dtype=float) + 10)

    @property
    def angles(self):
        return list(np.arange(7, dtype=float) * 0.1)

    @property
    def realtime_joint_speeds(self):
        return list(np.arange(7, dtype=float) * 0.01)

    def get_joint_states(self, is_radian=True, num=1):
        self.rpc_calls += 1
        return 0, [list(np.zeros(7))]

    def disconnect(self):
        pass


def make_dynamic_source(api, **kwargs):
    values = dict(baseline_verified=True, sign_verified=True)
    values.update(kwargs)
    return XArmFeedbackSource("test-ip", ArmFeedbackConfig(**values), api=api)


def test_dynamic_read_uses_report_synchronized_q_qd_and_skips_rpc():
    api = FakeDynamicArm()
    source = make_dynamic_source(api, dynamic_mode=True)
    sample = source.read_once()
    assert np.allclose(sample.raw_joint_effort, np.arange(7) + 10)
    assert np.allclose(sample.position, np.arange(7) * 0.1)
    assert np.allclose(sample.velocity, np.arange(7) * 0.01)
    assert api.rpc_calls == 0  # no RPC position read in dynamic mode


def test_static_read_path_unchanged():
    api = FakeDynamicArm()
    source = make_dynamic_source(api)
    sample = source.read_once()
    assert api.rpc_calls == 1
    assert sample.position is None and sample.velocity is None


class Source:
    def __init__(self, dynamic=False):
        self.dynamic = dynamic
        self.stopped = False

    @property
    def latest(self):
        kwargs = {}
        if self.dynamic:
            kwargs = dict(position=np.zeros(7), velocity=np.zeros(7))
        return ArmFeedbackSample(time.monotonic_ns(), np.ones(7), np.ones(7), sequence=1, **kwargs)

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class Adapter:
    def __init__(self):
        self.infos = []
        self.commands = []
        self.disabled = False
        self.enables = 0

    def enable(self):
        self.enables += 1

    def discover(self):
        return []

    def write(self, v, **kwargs):
        self.commands.append(v.copy())
        return v.copy()

    def disable(self):
        self.disabled = True


def make_worker(tmp_path, dynamic):
    values = dict(
        enabled=True,
        observe_only=True,
        update_hz=100,
        stale_timeout_ms=500,
        dynamic_mode=dynamic,
        baseline=(0.5,) * 7,
        log_path=str(tmp_path / "arm.csv"),
    )

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1, 1, 48.0

    return ArmFeedbackWorker(ArmFeedbackConfig(**values), Source(dynamic), Adapter(), state)


def test_dynamic_worker_logs_synchronized_state_and_tau_ext(tmp_path):
    w = make_worker(tmp_path, dynamic=True)
    w.start()
    deadline = time.monotonic() + 2
    while w.latest is None and time.monotonic() < deadline:
        time.sleep(0.005)
    w.stop()
    assert not w.fault
    for prefix in ("joint_position", "joint_velocity", "estimated_external_torque"):
        for j in range(1, 8):
            assert f"{prefix}_{j}" in w.latest
    # C1 identity: tau_ext == raw_effort - baseline == 1 - 0.5
    assert w.latest["estimated_external_torque_1"] == pytest.approx(0.5)
    assert w.adapter.enables == 0 and not w.adapter.commands  # observe-only


def test_static_worker_has_no_dynamic_columns(tmp_path):
    w = make_worker(tmp_path, dynamic=False)
    w.start()
    deadline = time.monotonic() + 2
    while w.latest is None and time.monotonic() < deadline:
        time.sleep(0.005)
    w.stop()
    assert not w.fault
    assert "estimated_external_torque_1" not in w.latest
    assert "joint_position_1" not in w.latest
