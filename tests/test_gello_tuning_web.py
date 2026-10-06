"""Continuous tuning, online gains, cancellation, lease loss and local HTTP controls."""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_robot_ufactory.current_control.control import runtime as runtime_module
from lerobot_robot_ufactory.current_control.config import DeviceProfile, CurrentControlConfig, tuning_slew
from lerobot_robot_ufactory.current_control.monitoring.encoder_monitor import EncoderMonitor
from lerobot_robot_ufactory.current_control.control.current import CurrentController
from lerobot_robot_ufactory.current_control.control.runtime import CurrentRuntime
from lerobot_robot_ufactory.current_control.control.tuning import (
    INITIAL_SLEW_A_S, TuningSession,
)
from lerobot_robot_ufactory.current_control.web.tuning_web import make_tuning_server, tuning_page

ROOT = Path(__file__).resolve().parents[1]


class TuningTransport:
    def __init__(self, profile):
        self.profile = profile
        self.position = np.r_[profile.zeros, 0.0]
        self.calls = []
        self.owners = set()
        self.gripper_limit = None

    def call(self, name):
        self.calls.append(name)
        self.owners.add(threading.get_ident())

    def open(self):
        self.call("open")

    def enable(self, **kwargs):
        self.call("enable")

    def state(self):
        self.call("state")
        return {
            "stamp": time.monotonic(), "position": self.position.copy(),
            "velocity": np.zeros(8), "current_a": [0] * 8,
            "temperature_c": [25] * 8, "voltage_v": [5] * 8,
        }

    def currents(self, values, gripper=None):
        assert np.max(np.abs(values)) <= np.max(self.profile.limits)
        self.call("currents")

    def health(self):
        self.call("health")

    def disable(self):
        self.call("disable")

    def close(self):
        self.call("close")




@pytest.fixture
def profile():
    p = DeviceProfile(ROOT / "config/current_control/gello_A_working.yaml")
    return p


def wait_for(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Expected tuning state was not reached")


def test_tuning_continues_past_experiment_limit_and_updates_only_on_owner(profile, monkeypatch):
    monkeypatch.setattr(runtime_module, "EXPERIMENT_MAX_DURATION_S", 0.025)
    transport = TuningTransport(profile)
    runtime = CurrentRuntime(profile, live=True, experimental=True, tuning=True,
                             transport=transport)
    try:
        runtime.start()
        time.sleep(0.08)
        assert runtime.status()["state"] == "active"
        values = [0.1] * 7
        assert runtime.set_current_slew(values) == values
        wait_for(lambda: runtime.diagnostics()["record"]["running_slew_a_s"] == values)
    finally:
        runtime.stop()
    assert len(transport.owners) == 1
    assert threading.get_ident() not in transport.owners
    assert transport.calls[-2:] == ["disable", "close"]


def test_tuning_lease_loss_unloads(profile, monkeypatch):
    monkeypatch.setattr(runtime_module, "TUNING_HEARTBEAT_TIMEOUT_S", 0.07)
    transport = TuningTransport(profile)
    runtime = CurrentRuntime(profile, live=True, experimental=True, tuning=True,
                             transport=transport)
    runtime.start()
    wait_for(lambda: runtime.status()["state"] == "fault")
    with pytest.raises(RuntimeError, match="失联"):
        runtime.stop()
    assert transport.calls[-2:] == ["disable", "close"]


def test_tuning_heartbeat_keeps_session_live_across_wide_joint_motion(profile, monkeypatch):
    monkeypatch.setattr(runtime_module, "TUNING_HEARTBEAT_TIMEOUT_S", 0.05)
    transport = TuningTransport(profile)
    runtime = CurrentRuntime(profile, live=True, experimental=True, tuning=True,
                             transport=transport)
    runtime.start()
    for _ in range(6):
        runtime.tuning_heartbeat()
        time.sleep(0.02)
    assert runtime.status()["state"] == "active"
    transport.position[1] += np.deg2rad(100)
    wait_for(lambda: runtime.diagnostics()["record"]["q"][1] > np.deg2rad(99))
    assert runtime.status()["state"] == "active"
    runtime.tuning_heartbeat()
    # The same continuous mode still rejects excessive measured current.
    original_state = transport.state

    def over_current():
        state = original_state()
        state["current_a"][1] = 1.01
        return state

    transport.state = over_current
    wait_for(lambda: runtime.status()["state"] == "fault")
    with pytest.raises(RuntimeError, match="current exceeded"):
        runtime.stop()
    assert transport.calls[-2:] == ["disable", "close"]




@pytest.mark.parametrize("values", [[0.049] * 7, [0.201] * 7, [float("nan")] * 7,
                                    [float("inf")] * 7, [True] * 7, [0.1] * 6, [-0.1] * 7])
def test_slew_validation_rejects_out_of_range_or_invalid_values(values):
    with pytest.raises(ValueError):
        tuning_slew(values)


def test_running_slew_preserves_startup_ramp_limits_and_reverse_rate(profile):
    profile.constant_current_a = [2.0] * 7  # Controller saturation test only.
    profile.constant_damping_a = np.zeros(7)
    rates = [0.05, 0.1, 0.2, 0.15, 0.05, 0.1, 0.2]
    controller = CurrentController(profile, running_slew_a_s=rates)
    position = np.r_[profile.zeros, 0]
    dt = 0.01
    current, record = controller.compute(position, np.zeros(8), dt, 1)
    assert np.allclose(current, profile.slew * dt)
    assert record["current_slew_a_s"] == profile.slew.tolist()
    before = current.copy()
    current, record = controller.compute(position, np.zeros(8), dt, 2)
    assert np.allclose(current - before, np.asarray(rates) * dt)
    assert record["current_slew_a_s"] == rates
    # A high requested current still saturates at the original current ceiling.
    controller.previous = profile.limits.copy()
    current, record = controller.compute(position, np.zeros(8), dt, 3)
    assert np.allclose(current, profile.limits) and record["saturated"]
    profile.constant_current_a = [-2.0] * 7
    current, _ = controller.compute(position, np.zeros(8), dt, 3.01)
    assert np.allclose(current, profile.limits - np.asarray(rates) * dt)
    assert np.all(profile.slew == 0.05)


def test_online_slew_is_owned_by_runtime_and_is_scoped_to_web(profile, tmp_path):
    transports = []
    session = session_with_fake(profile, tmp_path, transports)
    rates = [0.05, 0.1, 0.1, 0.15, 0.05, 0.2, 0.1]
    assert session.set_current_slew(rates) == rates
    assert transports == []
    try:
        session.start()
        runtime = session._runtime
        wait_for(lambda: runtime.diagnostics()["record"]["running_slew_a_s"] == rates)
        assert runtime.diagnostics()["record"]["current_slew_a_s"] == [0.05] * 7
        updated = [0.2] * 7
        assert session.set_current_slew(updated) == updated
        wait_for(lambda: runtime.diagnostics()["record"]["running_slew_a_s"] == updated)
        with pytest.raises(ValueError):
            session.set_current_slew([0.3] * 7)
        assert session.snapshot()["tuning"]["current_slew_a_s"] == updated
        assert np.all(profile.slew == 0.05)
        assert runtime.temperature_limit_c == 45
        session.stop()
        session.start()
        wait_for(lambda: session.snapshot()["tuning"]["running_slew_a_s"] == updated)
    finally:
        session.stop()
    assert all(len(t.owners) == 1 for t in transports)
    with pytest.raises(ValueError, match="web tuning"):
        CurrentRuntime(profile, live=True, experimental=True, tuning_slew_a_s=updated)
    other = CurrentRuntime(profile, live=True, experimental=True, transport=TuningTransport(profile))
    with pytest.raises(RuntimeError, match="web tuning"):
        other.set_current_slew(updated)


@pytest.mark.parametrize("filename", ["xarm7_gello_teleop_current.yaml", "xarm7_gello_record_current_config.yaml"])
def test_saved_defaults_load_in_teleop_and_preserve_startup_and_short_test_guards(filename, tmp_path):
    settings = yaml.safe_load((ROOT / "config/gello" / filename).read_text())["teleop"]["current_control"]
    profile = CurrentControlConfig(**settings).load_profile()
    session = TuningSession(profile, tmp_path)
    assert session.snapshot()["tuning"]["constant_current_a"] == profile.constant_current_a
    assert session.snapshot()["tuning"]["current_slew_a_s"] == profile.running_current_slew_a_s
    transport = TuningTransport(profile)
    runtime = CurrentRuntime(profile, live=True, experimental=True, teleop=True,
                             transport=transport)
    try:
        runtime.start()
        record = runtime.diagnostics()["record"]
        assert "gravity_gains" not in record
        assert record["current_slew_a_s"] == [0.05] * 7
        assert record["running_slew_a_s"] == profile.running_current_slew_a_s
        assert runtime.temperature_limit_c == 45
        controller = CurrentController(profile, running_slew_a_s=profile.running_current_slew_a_s)
        _, running = controller.compute(np.r_[profile.zeros, 0], np.zeros(8), .01, 3)
        assert running["current_slew_a_s"] == profile.running_current_slew_a_s
    finally:
        runtime.stop()
    untouched = TuningTransport(profile)
    with pytest.raises(ValueError, match="continuous live"):
        CurrentRuntime(profile, live=True, experimental=True, transport=untouched)
    assert untouched.calls == []
    assert np.all(profile.slew == 0.05)


def test_running_slew_config_rejects_invalid_values_and_disabled_support():
    with pytest.raises(ValueError, match="0.05"):
        CurrentControlConfig(enabled=True, profile_path="unused", running_current_slew_a_s=[.3] * 7)
    with pytest.raises(ValueError, match="enabled"):
        CurrentControlConfig(running_current_slew_a_s=[.1] * 7)


def test_web_temperature_guard_unloads_at_45_degrees(profile):
    transport = TuningTransport(profile)
    original_state = transport.state

    def warm_state():
        state = original_state()
        state["temperature_c"][1] = warm_state.temperature
        return state

    warm_state.temperature = 44
    transport.state = warm_state
    runtime = CurrentRuntime(profile, live=True, experimental=True, tuning=True,
                             tuning_slew_a_s=[0.2] * 7, transport=transport)
    runtime.start()
    assert runtime.status()["state"] == "active"
    warm_state.temperature = 45
    wait_for(lambda: runtime.status()["state"] == "fault")
    with pytest.raises(RuntimeError, match=r"temperature.*45 C"):
        runtime.stop()
    assert transport.calls[-2:] == ["disable", "close"]
    assert profile.temperature_limit_c == 50


def session_with_fake(profile, tmp_path, transports):
    def factory(p, **kwargs):
        transport = TuningTransport(p)
        transports.append(transport)
        return CurrentRuntime(p, transport=transport, **kwargs)

    return TuningSession(profile, tmp_path / "logs", runtime_factory=factory)


def test_idle_does_not_open_hardware_and_session_logs_and_restarts(profile, tmp_path):
    transports = []
    session = session_with_fake(profile, tmp_path, transports)
    session.set_current_slew([0.1] * 7)
    assert transports == []
    assert session.snapshot()["tuning"]["state"] == "idle"
    try:
        session.start()
        session.heartbeat()
        session.set_current_slew([0.2] * 7)
        wait_for(lambda: session.snapshot()["tuning"]["running_slew_a_s"] == [0.2] * 7)
        assert session.snapshot()["sample"]["model_q_rad"] == [0] * 7
        session.stop()
        assert session.snapshot()["tuning"]["state"] == "stopped"
        session.start()
        wait_for(lambda: session.snapshot()["tuning"]["state"] == "active")
    finally:
        session.stop()
    assert len(transports) == 2
    assert all(t.calls[-2:] == ["disable", "close"] for t in transports)
    logs = list((tmp_path / "logs").glob("*.jsonl"))
    assert len(logs) == 2
    assert any(json.loads(line)["running_slew_a_s"] == [0.2] * 7 for line in logs[0].read_text().splitlines())
    assert "joint_gains" not in profile.data


def test_stop_during_startup_never_enables_torque(profile, tmp_path):
    entered, release = threading.Event(), threading.Event()
    transports = []

    def factory(p, **kwargs):
        t = TuningTransport(p)
        original = t.open

        def slow_open():
            original()
            entered.set()
            assert release.wait(2)

        t.open = slow_open
        transports.append(t)
        return CurrentRuntime(p, transport=t, **kwargs)

    session = TuningSession(profile, tmp_path, runtime_factory=factory)
    starter = threading.Thread(target=session.start)
    starter.start()
    assert entered.wait(1)
    stopper = threading.Thread(target=session.stop)
    stopper.start()
    wait_for(lambda: session._cancel.is_set())
    release.set()
    starter.join(2)
    stopper.join(2)
    assert not starter.is_alive() and not stopper.is_alive()
    assert "enable" not in transports[0].calls
    assert session.snapshot()["tuning"]["state"] == "stopped"


def test_http_controls_validate_origin_and_never_auto_start(profile, tmp_path):
    transports = []
    session = session_with_fake(profile, tmp_path, transports)
    token = "test-only-token"
    readers = []

    def monitor_factory(p):
        transport = TuningTransport(p)
        transport.info = []
        readers.append(transport)
        return EncoderMonitor(p, transport_factory=lambda _: transport)

    session._monitor_factory = monitor_factory
    page = tuning_page(profile, token)
    assert b"/*__LIVE_VIEW__*/" not in page
    server = make_tuning_server(0, page, session, token)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(path, body=None, **headers):
        request = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json", "X-Gello-Token": token, **headers})
        with opener.open(request, timeout=5) as response:
            return json.load(response)

    try:
        with opener.open(url) as response:
            assert response.read() == page
        assert transports == []
        for headers in ({"X-Gello-Token": "bad"}, {"Origin": "https://example.com"},
                        {"Host": "evil.example"}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post("/api/start", **headers)
            assert exc.value.code == 403
        assert transports == []
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/gains", {"gains": [True] * 7})
        assert exc.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/slew", {"current_slew_a_s": [0.3] * 7})
        assert exc.value.code == 400
        assert transports == []
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/view", {"mode": "read_only"}, **{"X-Gello-Token": "bad"})
        assert exc.value.code == 403
        assert readers == []
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/view", {"mode": "compensation"})
        assert exc.value.code == 400
        assert post("/api/view", {"mode": "read_only"})["tuning"]["state"] == "reading"
        wait_for(lambda: session.snapshot()["sample"] is not None)
        assert transports == []
        updated = post("/api/slew", {"current_slew_a_s": [0.1] * 7})
        assert updated["tuning"]["current_slew_a_s"] == [0.1] * 7
        post("/api/heartbeat")
        assert post("/api/start")["tuning"]["state"] == "active"
        assert readers[0].calls[-1] == "close"
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/view", {"mode": "offline"})
        assert exc.value.code == 503
        post("/api/slew", {"current_slew_a_s": [0.2] * 7})
        assert post("/api/stop")["tuning"]["state"] == "stopped"
    finally:
        session.stop()
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_read_only_handoff_closes_reader_before_torque_owner_opens(profile, tmp_path):
    events = []
    readers, controllers = [], []

    def monitor_factory(p):
        transport = TuningTransport(p)
        transport.info = []
        original = transport.call

        def call(name):
            events.append(("read", name))
            original(name)

        transport.call = call
        readers.append(transport)
        return EncoderMonitor(p, transport_factory=lambda _: transport)

    def runtime_factory(p, **kwargs):
        assert readers[-1].calls[-1] == "close"
        transport = TuningTransport(p)
        original = transport.call

        def call(name):
            events.append(("control", name))
            original(name)

        transport.call = call
        controllers.append(transport)
        return CurrentRuntime(p, transport=transport, **kwargs)

    session = TuningSession(profile, tmp_path, monitor_factory=monitor_factory,
                            runtime_factory=runtime_factory)
    try:
        assert session.snapshot()["view_mode"] == "offline"
        assert events == []
        session.set_view_mode("read_only")
        wait_for(lambda: session.snapshot()["sample"] is not None)
        packet = session.snapshot()
        assert packet["tuning"]["state"] == "reading"
        assert packet["tuning"]["record"] is None
        assert packet["status"] == "connected"
        assert set(readers[0].calls) == {"open", "state"}
        session.set_current_slew([0.1] * 7)
        assert controllers == []
        session.start()
        wait_for(lambda: session.snapshot()["tuning"]["record"] is not None)
        assert session.snapshot()["view_mode"] == "compensation"
        assert events.index(("read", "close")) < events.index(("control", "open"))
        with pytest.raises(RuntimeError, match="先立即卸力"):
            session.set_view_mode("offline")
        with pytest.raises(RuntimeError, match="先立即卸力"):
            session.set_view_mode("read_only")
        assert session.snapshot()["tuning"]["state"] == "active"
        session.stop()
        assert controllers[0].calls[-2:] == ["disable", "close"]
        session.set_view_mode("read_only")
        wait_for(lambda: session.snapshot()["sample"] is not None)
        session.set_view_mode("offline")
        assert readers[1].calls[-1] == "close"
        assert session.snapshot()["sample"] is None
        assert session.snapshot()["view_mode"] == "offline"
    finally:
        session.stop()
    assert all(set(t.calls) == {"open", "state", "close"} for t in readers)


def test_failed_reader_stop_blocks_compensation_owner(profile, tmp_path):
    class StuckMonitor:
        released = False

        def start(self):
            pass

        def snapshot(self):
            return {"status": "connecting", "error": None, "sample": None}

        def stop(self):
            if not self.released:
                raise RuntimeError("reader did not stop")

    monitor = StuckMonitor()
    created = []
    session = TuningSession(profile, tmp_path, monitor_factory=lambda _: monitor,
                            runtime_factory=lambda *args, **kwargs: created.append(True))
    session.set_view_mode("read_only")
    try:
        with pytest.raises(RuntimeError, match="reader did not stop"):
            session.start()
        assert created == []
        assert session.snapshot()["tuning"]["state"] == "fault"
    finally:
        monitor.released = True
        session.stop()


def test_stop_during_read_only_handoff_cancels_new_torque_owner(profile, tmp_path):
    closing, released = threading.Event(), threading.Event()
    transports = []

    class SlowMonitor:
        def start(self):
            pass

        def stop(self):
            closing.set()
            assert released.wait(2)

    def factory(p, **kwargs):
        transport = TuningTransport(p)
        transports.append(transport)
        return CurrentRuntime(p, transport=transport, **kwargs)

    session = TuningSession(profile, tmp_path, monitor_factory=lambda _: SlowMonitor(),
                            runtime_factory=factory)
    session.set_view_mode("read_only")
    starter = threading.Thread(target=session.start)
    starter.start()
    assert closing.wait(1)
    stopper = threading.Thread(target=session.stop)
    stopper.start()
    wait_for(lambda: session._cancel.is_set())
    released.set()
    starter.join(2)
    stopper.join(2)
    assert not starter.is_alive() and not stopper.is_alive()
    assert "enable" not in transports[0].calls
    assert session.snapshot()["tuning"]["state"] == "stopped"
