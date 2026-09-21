import json
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.scripts.uf_test_arm_force_feedback import summarize
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig
from lerobot_robot_ufactory.utils.arm_feedback import ArmFeedbackConfig, ArmFeedbackSample
from lerobot_robot_ufactory.utils.arm_feedback_runtime import ArmFeedbackWorker


class Source:
    def __init__(self):
        self.stopped = False

    @property
    def latest(self):
        return ArmFeedbackSample(time.monotonic_ns(), np.ones(7), np.ones(7), sequence=1)

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class Adapter:
    def __init__(self, fail=False):
        self.infos = []
        self.commands = []
        self.deadlines = []
        self.disabled = False
        self.fail = fail
        self.enables = 0

    def enable(self):
        self.enables += 1

    def discover(self):
        return []

    def write(self, v, **kwargs):
        if self.fail:
            raise RuntimeError("simulated disconnected serial port")
        self.commands.append(v.copy())
        self.deadlines.append(kwargs.get("deadline_ns"))
        return v.copy()

    def disable(self):
        self.disabled = True


def make_worker(tmp_path, **kwargs):
    values = dict(
        enabled=True,
        update_hz=100,
        stale_timeout_ms=500,
        baseline_verified=True,
        sign_verified=True,
        enabled_joints=(True,) * 7,
        gain_ma_per_unit=(5,) * 7,
        log_path=str(tmp_path / "arm.csv"),
    )
    values.update(kwargs)

    def state():
        # Same 7-tuple shape as TimedArmReaderMixin.arm_state_snapshot:
        # (ns, pos, vel, error, read_ms, sequence, period_ms); 48 ms is the
        # measured steady-state SyncRead period.
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1, 1, 48.0

    return ArmFeedbackWorker(ArmFeedbackConfig(**values), Source(), Adapter(), state)


@pytest.mark.parametrize("observe", [True, False])
def test_worker_lifecycle_csv_and_observe_no_output(tmp_path, observe):
    w = make_worker(tmp_path, observe_only=observe)
    w.start()
    deadline = time.monotonic() + 2
    while w.latest is None and time.monotonic() < deadline:
        time.sleep(0.005)
    w.stop()
    assert w.latest is not None and not w.fault
    assert w.source.stopped and not w.thread.is_alive() and not w.watchdog.is_alive()
    assert w.adapter.disabled
    if observe:
        assert w.adapter.enables == 0 and not w.adapter.commands
    else:
        assert w.adapter.enables == 1 and not w.adapter.commands[0].any()
    report = summarize(w.log_path)
    if observe:
        assert report["dynamixel_write_latency_ms"] == "NOT MEASURED"
    assert w.log_path.with_suffix(".metadata.json").exists()
    assert json.loads(w.log_path.with_suffix(".status.json").read_text())["fault"] is None


def test_start_waits_for_leader_cadence_to_stabilize(tmp_path):
    # Post-enable startup transient: SyncRead cycles of ~400 ms publish
    # snapshots whose timestamp predates the transaction. The worker must
    # not enter the write loop until the read period is back near its
    # ~48 ms steady state.
    w = make_worker(tmp_path, observe_only=False)
    periods = iter([400.0, 400.0, 400.0, 48.0])

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1, 1, next(
            periods, 48.0
        )

    w.leader_state = state
    w.start()
    w.stop()
    assert not w.fault and w.adapter.enables == 1


def test_start_rejects_persistent_slow_leader_cadence(tmp_path):
    w = make_worker(tmp_path, observe_only=False)

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1, 1, 400.0

    w.leader_state = state
    with pytest.raises(RuntimeError, match="leader_stale after adapter init"):
        w.start()
    assert w.adapter.disabled


def test_worker_write_exception_latches_fault_and_cleanup(tmp_path):
    w = make_worker(tmp_path, observe_only=False)
    w.adapter.fail = True
    w.start()
    w.thread.join(1)
    w.stop()
    assert "write_error" in w.fault and w.adapter.disabled
    assert w.counters["write_error_count"] == 1


def test_watchdog_disables_stalled_worker(tmp_path):
    w = make_worker(tmp_path, observe_only=False, stale_timeout_ms=40)
    w.heartbeat_ns = time.monotonic_ns() - 100_000_000
    t = threading.Thread(target=w._watchdog)
    t.start()
    t.join(1)
    assert w.fault == "worker_watchdog_timeout" and w.adapter.disabled


def test_processing_exception_cleans_up_and_is_recorded(tmp_path):
    w = make_worker(tmp_path, observe_only=False)
    original = w.processor.process
    calls = 0

    def fail_after_start(*a, **kw):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("injected processing failure")
        return original(*a, **kw)

    w.processor.process = fail_after_start
    w.start()
    w.thread.join(1)
    w.stop()
    assert "worker_exception" in w.fault
    assert w.adapter.disabled and w.source.stopped


def test_existing_log_preserved_on_new_session(tmp_path):
    path = tmp_path / "arm.csv"
    path.write_text("previous experiment")
    w = make_worker(tmp_path)
    w.start()
    w.stop()
    assert path.read_text() == "previous experiment"
    assert w.log_path != path


def test_nested_config_draccus_and_independent_id8():
    import draccus

    c = GelloTeleopConfig(arm_feedback=asdict(ArmFeedbackConfig(enabled=True)))
    assert c.arm_feedback.enabled and not c.gripper_force_feedback_enabled
    decoded = draccus.decode(GelloTeleopConfig, {"arm_feedback": {"enabled": True}})
    assert decoded.arm_feedback.enabled and decoded.arm_feedback.observe_only


def test_startup_failure_logged_and_sampler_closed(tmp_path):
    w = make_worker(tmp_path, observe_only=False)

    def fail():
        raise RuntimeError("unsupported motor")

    w.adapter.enable = fail
    with pytest.raises(RuntimeError, match="unsupported motor"):
        w.start()
    assert w.source.stopped and w.log_file.closed
    assert (
        "startup_error" in json.loads(w.log_path.with_suffix(".status.json").read_text())["fault"]
    )


def test_source_stop_failure_does_not_skip_log_cleanup(tmp_path):
    w = make_worker(tmp_path)
    w.start()

    def fail():
        raise RuntimeError("sampler disconnect error")

    w.source.stop = fail
    with pytest.raises(RuntimeError, match="sampler disconnect error"):
        w.stop()
    assert w.adapter.disabled and w.log_file.closed
    assert "cleanup_error" in w.fault


def test_runtime_starts_both_channels_and_stops_independently():
    # Exercise actual controller lifecycle hook without hardware threads.
    from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController

    calls = []
    controller = RealtimeTeleopController.__new__(RealtimeTeleopController)
    controller.teleop = SimpleNamespace(
        config=SimpleNamespace(arm_feedback=ArmFeedbackConfig(enabled=True)),
        start_feedback=lambda: calls.append("id8_start"),
        start_arm_feedback=lambda ip: calls.append(("arm_start", ip)),
        stop_arm_feedback=lambda: calls.append("arm_stop"),
        stop_feedback=lambda: calls.append("id8_stop"),
        send_feedback=lambda value: None,
    )
    controller.robot = SimpleNamespace(config=SimpleNamespace(robot_ip="test"))
    controller._gripper_feedback_enabled = True
    controller._thread = SimpleNamespace(start=lambda: None)
    controller._first_action = SimpleNamespace(wait=lambda **kw: True)
    controller.raise_if_failed = lambda: None
    controller.reset_gripper_feedback_state = lambda reason: None
    controller.start()
    controller._safe_stop_feedback_output()
    assert calls == ["id8_start", ("arm_start", "test"), "arm_stop", "id8_stop"]


def test_throttled_leader_cadence_accepted_at_startup(tmp_path):
    # leader_read_hz=5 makes a ~200 ms leader period the configured steady
    # state, not a fault; the startup gate must accept it (budget 300 ms).
    w = make_worker(tmp_path, observe_only=False, leader_read_hz=5)

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1, 1, 200.0

    w.leader_state = state
    w.start()
    w.stop()
    assert not w.fault and w.adapter.enables == 1


def test_throttled_leader_cadence_beyond_budget_rejected(tmp_path):
    # Even with throttling configured, a period far above 1.5x the configured
    # one still means the reader is unhealthy.
    w = make_worker(tmp_path, observe_only=False, leader_read_hz=5)

    def state():
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1, 1, 500.0

    w.leader_state = state
    with pytest.raises(RuntimeError, match="leader_stale after adapter init"):
        w.start()
    assert w.adapter.disabled


def test_zero_damping_fast_path_ignores_stale_leader(tmp_path):
    # damping=0 on every enabled joint: the feedback command must not wait on
    # or fail because of the (possibly throttled) leader reader, and the write
    # deadline must derive from the xArm sample, not the stale leader.
    w = make_worker(tmp_path, observe_only=False)
    assert not w.needs_leader_velocity
    w.start()
    stale_ns = time.monotonic_ns() - 10_000_000_000
    w.leader_state = lambda: (stale_ns, np.zeros(7), np.zeros(7), None, 0.1, 1, 48.0)
    deadline = time.monotonic() + 2
    while not w.adapter.commands and time.monotonic() < deadline:
        time.sleep(0.005)
    w.stop()
    assert w.adapter.commands and not w.fault
    # sample-based expiry: stale_ns + 10 s would be long expired if the
    # deadline were min(sample, leader) + stale_timeout.
    assert w.adapter.deadlines[-1] - stale_ns > 10_000_000_000
    assert w.latest["stale"] in ("False", False)


def test_damping_stale_leader_latches_fault(tmp_path):
    # With damping active on an enabled joint the leader velocity feeds the
    # command, so a stale leader must still disable output.
    w = make_worker(tmp_path, observe_only=False, damping_ma_per_rad_s=(1,) * 7)
    assert w.needs_leader_velocity
    w.start()
    stale_ns = time.monotonic_ns() - 10_000_000_000
    w.leader_state = lambda: (stale_ns, np.zeros(7), np.zeros(7), None, 0.1, 1, 48.0)
    w.thread.join(2)
    w.stop()
    assert "leader_stale" in (w.fault or "") and w.adapter.disabled


def test_csv_contains_lock_instrumentation_columns(tmp_path):
    w = make_worker(tmp_path)
    w.start()
    deadline = time.monotonic() + 2
    while w.latest is None and time.monotonic() < deadline:
        time.sleep(0.005)
    w.stop()
    header = w.log_path.open(encoding="utf-8").readline()
    for column in (
        "write_latency_ms",
        "serial_transaction_ms",
        "serial_lock_wait_ms_read",
        "serial_lock_hold_ms_read",
        "serial_lock_wait_ms_write",
        "serial_lock_hold_ms_write",
    ):
        assert column in header


class FakeArm:
    """Minimal XArmAPI stand-in: RPC position + report-cache effort."""

    def __init__(self, effort=None, rpc_effort=None, stamp=None):
        self._arm = SimpleNamespace(
            _last_update_cmdnum_time=time.monotonic() if stamp is None else stamp
        )
        self._effort = list(np.arange(7, dtype=float) if effort is None else effort)
        self._rpc_effort = list(self._effort if rpc_effort is None else rpc_effort)
        self.rpc_nums = []
        self.disconnected = False

    @property
    def joints_torque(self):
        return list(self._effort)

    def get_joint_states(self, is_radian=True, num=1):
        self.rpc_nums.append(num)
        if num == 1:
            return 0, [list(np.zeros(7))]
        return 0, [list(np.zeros(7)), [0.0] * 7, list(self._rpc_effort)]

    def disconnect(self):
        self.disconnected = True


def make_source(api, **kwargs):
    from lerobot_robot_ufactory.utils.arm_feedback_runtime import XArmFeedbackSource

    values = dict(baseline_verified=True, sign_verified=True)
    values.update(kwargs)
    return XArmFeedbackSource("test-ip", ArmFeedbackConfig(**values), api=api)


def test_source_reads_effort_from_report_cache_not_rpc():
    api = FakeArm(effort=np.arange(7, dtype=float), rpc_effort=np.full(7, 99.0))
    source = make_source(api)
    sample = source.read_once()
    # Effort must come from the report stream cache, not the RPC response.
    assert np.allclose(sample.raw_joint_effort, np.arange(7))
    assert np.allclose(sample.estimated_contact_torque, np.arange(7))
    assert source.report_age_ms is not None and source.report_age_ms >= 0
    assert api.rpc_nums == [1]  # position-only RPC; no frozen num=3 effort


def test_source_faults_on_stale_report_stream():
    api = FakeArm(stamp=time.monotonic() - 10)
    source = make_source(api)
    with pytest.raises(RuntimeError, match="report stream stale"):
        source.read_once()


def test_source_faults_when_report_stream_never_started():
    api = FakeArm(stamp=0)
    source = make_source(api)
    with pytest.raises(RuntimeError, match="report stream not started"):
        source.read_once()


def test_source_start_enables_report_stream(monkeypatch):
    captured = {}

    def factory(ip, **kwargs):
        captured.update(ip=ip, **kwargs)
        return FakeArm()

    monkeypatch.setattr("xarm.wrapper.XArmAPI", factory)
    source = make_source(None)
    source.start()
    source.stop()
    assert captured["ip"] == "test-ip"
    assert captured["enable_report"] is True
    assert captured["report_type"] == "rich"


def test_source_start_rejects_effort_scale_mismatch(monkeypatch):
    api = FakeArm(effort=np.zeros(7), rpc_effort=np.full(7, 50.0))
    monkeypatch.setattr("xarm.wrapper.XArmAPI", lambda ip, **kw: api)
    source = make_source(None)
    with pytest.raises(RuntimeError, match="re-verify the baseline"):
        source.start()
    assert api.disconnected  # startup failure must close the session


def test_source_start_rejects_missing_report_stream(monkeypatch):
    api = FakeArm(stamp=0)
    monkeypatch.setattr("xarm.wrapper.XArmAPI", lambda ip, **kw: api)
    source = make_source(None)
    with pytest.raises(RuntimeError, match="report stream did not start"):
        source.start()
    assert api.disconnected
