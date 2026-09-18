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
        return time.monotonic_ns(), np.zeros(7), np.zeros(7), None, 0.1

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
