import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_robot_ufactory.current_control.config import DeviceProfile
from lerobot_robot_ufactory.current_control.control.current import CurrentController
from lerobot_robot_ufactory.current_control.control.runtime import CurrentRuntime, RuntimeRobot
from lerobot_robot_ufactory.current_control.hardware.transport import XL330Transport, signed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile(tmp_path):
    original = ROOT / "config/current_control/gello_A_working.yaml"
    data = yaml.safe_load(original.read_text())
    data["urdf"] = str((original.parent / "model/xarm7_gello.urdf").resolve())
    # Keep the unit-test device conservative and neutral without shipping a
    # second fixture profile under config/current_control.
    data["constant_current_a"] = [0.01] * 7
    data.pop("constant_damping_a", None)
    data["encoder_zero_rad"] = [0.0] * 7
    data["current_limit_a"] = [0.05] * 7
    data["baudrate"] = 57600
    path = tmp_path / "device.yaml"
    path.write_text(yaml.safe_dump(data))
    return DeviceProfile(path)


def commissioned(profile, tmp_path):
    data = profile.data.copy()
    # Test-only fixture, never shipped as a commissioned hardware profile.
    urdf = tmp_path / "test-only.urdf"
    urdf.write_text(profile.urdf.read_text().replace("UNVERIFIED_GEOMETRY", "TEST_ONLY"))
    data.update(
        urdf=str(urdf),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        baudrate=1000000,
        rate_hz=100,
        state_timeout_s=0.05,
        watchdog_ms=100,
    )
    data["commissioning"] = {k: True for k in data["commissioning"]}
    path = tmp_path / "commissioned.yaml"
    path.write_text(yaml.safe_dump(data))
    return DeviceProfile(path)




class FakeTransport:
    def __init__(self):
        self.calls = []
        self.stale = False
        self.fail_write = False
        self.gripper_limit = None

    def open(self):
        self.calls.append("open")

    def enable(self, *, experimental=False):
        self.calls.append("enable")

    def state(self):
        return {
            "stamp": time.monotonic() - (1 if self.stale else 0),
            "position": np.zeros(8),
            "velocity": np.zeros(8),
            "current_a": [0] * 8,
            "temperature_c": [25] * 8,
            "voltage_v": [5] * 8,
        }

    def currents(self, current, gripper=None):
        if self.fail_write:
            raise RuntimeError("injected write failure")
        self.calls.append("write")

    def health(self):
        pass

    def disable_gripper(self):
        self.gripper_limit = None

    def disable(self):
        self.calls.append("disable")

    def close(self):
        self.calls.append("close")


def test_starter_profiles_are_read_only(profile):
    with pytest.raises(ValueError, match="1 Mbps"):
        profile.validate_live()


@pytest.fixture
def experimental_profile(profile):
    profile.gain = 0.02
    profile.baudrate = 1000000
    profile.rate_hz = 100
    profile.state_timeout_s = 0.05
    profile.watchdog_ms = 100
    profile.ramp_s = 2
    profile.temperature_limit_c = 50
    profile.limits = np.ones(7)
    profile.slew = np.full(7, 0.05)
    digest = hashlib.sha256(profile.urdf.read_bytes()).hexdigest()
    profile.data.update(
        urdf_sha256=digest,
        alignment_user_accepted=True,
        direction_verification={
            "model_signs_verified": True,
            "urdf_sha256": digest,
            "model_signs": profile.signs.tolist(),
        },
        communication_verification={
            "baudrate_verified": 1000000,
            "target_100hz_verified": True,
        },
    )
    return profile




@pytest.mark.parametrize("field,value", [
    ("limits", np.full(7, 1.01)),
    ("slew", np.full(7, 0.051)), ("ramp_s", 1.9),
    ("temperature_limit_c", 51), ("state_timeout_s", 0.051),
    ("watchdog_ms", 120), ("baudrate", 57600),
])
def test_experiment_rejects_unsafe_parameters(experimental_profile, field, value):
    setattr(experimental_profile, field, value)
    with pytest.raises(ValueError):
        experimental_profile.validate_live(experimental=True)






















def test_experiment_runtime_has_independent_time_limit(experimental_profile, monkeypatch):
    from lerobot_robot_ufactory.current_control.control import runtime

    assert runtime.EXPERIMENT_MAX_DURATION_S == 10.0
    monkeypatch.setattr(runtime, "EXPERIMENT_MAX_DURATION_S", 0.05)
    fake = FakeTransport()
    r = CurrentRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake)
    r.start()
    r._thread.join(timeout=1)
    assert not r._thread.is_alive()
    r.raise_if_failed()
    assert "write" in fake.calls
    assert fake.calls[-2:] == ["disable", "close"]
    assert all(record["experimental"] for record in r.drain_records())


@pytest.mark.parametrize("displacement_deg", [-45, -44, 44, 45])
def test_experiment_allows_motion_up_to_45_degrees(experimental_profile, monkeypatch, displacement_deg):
    from lerobot_robot_ufactory.current_control.control import runtime

    assert runtime.EXPERIMENT_MAX_DISPLACEMENT_DEG == 45.0
    monkeypatch.setattr(runtime, "EXPERIMENT_MAX_DURATION_S", 0.03)
    fake = FakeTransport()
    original = fake.state
    reads = 0

    def state():
        nonlocal reads
        reads += 1
        result = original()
        if reads >= 3:
            result["position"][0] = np.deg2rad(displacement_deg)
        return result

    monkeypatch.setattr(fake, "state", state)
    r = CurrentRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake)
    r.start()
    r._thread.join(timeout=1)
    assert not r._thread.is_alive()
    r.raise_if_failed()
    assert "write" in fake.calls
    assert fake.calls[-2:] == ["disable", "close"]


@pytest.mark.parametrize("fault", ["motion", "negative_motion", "current", "mode_change"])
def test_experiment_fault_stops_and_cleans_up(experimental_profile, monkeypatch, fault):
    fake = FakeTransport()
    original = fake.state
    reads = 0

    def state():
        nonlocal reads
        reads += 1
        result = original()
        if fault == "mode_change" and reads >= 2:
            result["position"][0] = np.deg2rad(3)
        if fault == "motion" and reads >= 3:
            result["position"][0] = np.deg2rad(46)
        if fault == "negative_motion" and reads >= 3:
            result["position"][0] = np.deg2rad(-46)
        if fault == "current" and reads >= 3:
            result["current_a"][0] = 1.01
        return result

    monkeypatch.setattr(fake, "state", state)
    r = CurrentRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake)
    with pytest.raises(RuntimeError):
        r.start()
    r._thread.join(timeout=1)
    assert "write" not in fake.calls
    assert fake.calls[-2:] == ["disable", "close"]


@pytest.mark.parametrize("mode", ["teleop", "tuning"])
@pytest.mark.parametrize("displacement_deg", [-100, 100])
def test_continuous_support_allows_wide_motion(experimental_profile, monkeypatch, mode, displacement_deg):
    from lerobot_robot_ufactory.current_control.control import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "EXPERIMENT_MAX_DURATION_S", 0.02)
    fake = FakeTransport()
    original_state = fake.state
    reads = 0

    def state():
        nonlocal reads
        reads += 1
        result = original_state()
        if reads >= 3:
            result["position"][1] = np.deg2rad(displacement_deg)
        return result

    fake.state = state
    r = CurrentRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake, **{mode: True})
    try:
        r.start()
        time.sleep(0.06)
        assert r.status()["state"] == "active"
        assert r.diagnostics()["record"]["q"][1] == pytest.approx(np.deg2rad(displacement_deg))
        assert "write" in fake.calls
    finally:
        r.stop()
    assert fake.calls[-2:] == ["disable", "close"]








@pytest.mark.parametrize(
    "field,value",
    [
        ("model_signs", [0] * 7),
        ("constant_damping_a", [-1] * 7),
        ("current_limit_a", [2] * 7),
        ("encoder_zero_rad", [float("nan")] * 7),
        ("damping_deadband_rad_s", -1),
        ("constant_current_a", [True] * 7),
    ],
)
def test_invalid_profiles_fail_before_hardware(profile, field, value):
    data = profile.data.copy()
    data[field] = value
    profile.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        DeviceProfile(profile.path)




def test_ramp_slew_and_hard_limit(profile):
    controller = CurrentController(profile)
    first, _ = controller.compute(np.zeros(8), np.zeros(8), 0.05, 0)
    assert np.array_equal(first, np.zeros(7))
    profile.constant_current_a = [1.0] * 7
    profile.limits[:] = 0.002
    for _ in range(5):
        current, record = controller.compute(np.zeros(8), np.zeros(8), 0.05, 3)
        assert np.max(np.abs(current)) <= 0.002
    assert record["saturated"]


@pytest.mark.parametrize("dt", [0, -1, 1, float("nan")])
def test_bad_control_interval(profile, dt):
    with pytest.raises(RuntimeError):
        CurrentController(profile).compute(np.zeros(8), np.zeros(8), dt, 1)


def test_nan_state(profile):
    position = np.zeros(8)
    position[2] = float("nan")
    with pytest.raises(ValueError):
        CurrentController(profile).compute(position, np.zeros(8), 0.05, 1)


def test_observe_does_not_enable_or_write(profile):
    transport = FakeTransport()
    runtime = CurrentRuntime(profile, transport=transport)
    runtime.start()
    runtime.state()
    record = runtime.drain_records()[0]
    assert len(record["position"]) == len(record["velocity"]) == 8
    runtime.stop()
    runtime.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        runtime.state()
    assert "write" not in transport.calls and "enable" not in transport.calls
    assert transport.calls[-2:] == ["disable", "close"]


def test_read_only_probe_cleanup_has_no_register_writes(profile):
    transport = XL330Transport(profile)
    transport.packet = Packet(profile)
    transport.disable()
    assert transport.packet.writes == []


def test_stale_state_cleans_up(profile, tmp_path):
    p = commissioned(profile, tmp_path)
    transport = FakeTransport()
    transport.stale = True
    runtime = CurrentRuntime(p, live=True, transport=transport)
    with pytest.raises(RuntimeError, match="timeout"):
        runtime.start()
    assert transport.calls[-2:] == ["disable", "close"]
    assert "write" not in transport.calls


def test_write_failure_latches_fault(profile, tmp_path):
    p = commissioned(profile, tmp_path)
    transport = FakeTransport()
    transport.fail_write = True
    runtime = CurrentRuntime(p, live=True, transport=transport)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.start()
    assert runtime.status()["state"] == "fault"
    assert transport.calls[-2:] == ["disable", "close"]
    with pytest.raises(RuntimeError, match="injected write failure"):
        runtime.state()


@pytest.mark.parametrize("delay_phase", ["compute", "write"])
def test_follower_reads_fresh_encoders_during_current_processing(experimental_profile, monkeypatch, delay_phase):
    from types import SimpleNamespace
    from lerobot_robot_ufactory.current_control.control import runtime as runtime_module

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    observations = []

    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            clock.now += timeout

    class TimedTransport(FakeTransport):
        reads = 0
        writes = 0

        def state(self):
            self.reads += 1
            result = super().state()
            result["stamp"] = clock.now
            result["position"] = np.full(7, 0.01 * self.reads)
            result["velocity"] = np.zeros(7)
            clock.now += 0.010
            return result

        def currents(self, current, gripper=None):
            self.writes += 1
            if self.writes == 1:
                # Startup still waits for the first successful current write.
                assert not runtime._ready.is_set()
            super().currents(current, gripper)
            if self.writes == 2:
                if delay_phase == "write":
                    clock.now += 0.016
                    read_follower()
                clock.now += 0.009
            else:
                clock.now += 0.025
            if self.writes == 3:
                runtime.request_stop()

    transport = TimedTransport()
    runtime = CurrentRuntime(
        experimental_profile.arm_only(), live=True, teleop=True, transport=transport,
    )
    runtime._stop = StopEvent()
    follower = RuntimeRobot(runtime, [1] * 7, None)

    def read_follower():
        state = runtime.state()
        # The previous published frame would now be 61 ms old. The newly
        # read frame is only 26 ms old, even though this tick has not finished.
        observations.append((clock.now - state["stamp"], follower.get_joint_state().copy()))
        assert runtime.diagnostics()["record"]["position"][0] == pytest.approx(0.01)

    original_compute = runtime.controller.compute

    def compute(*args):
        if transport.reads == 2 and delay_phase == "compute":
            clock.now += 0.016
            read_follower()
        return original_compute(*args)

    monkeypatch.setattr(runtime.controller, "compute", compute)
    runtime._run()
    runtime.raise_if_failed()
    assert len(observations) == 1
    age, position = observations[0]
    assert age == pytest.approx(0.026)
    assert np.allclose(position, 0.02)
    records = runtime.drain_records()
    assert len(records) == 3
    assert all(record["work_s"] == pytest.approx(0.035) for record in records)
    assert all(record["interval_s"] <= 0.05 for record in records)
    assert transport.calls[-2:] == ["disable", "close"]


@pytest.mark.parametrize("fault, message", [
    ("position_shape", "Unexpected state shape"),
    ("position_nan", "Non-finite encoder state"),
    ("velocity_nan", "Non-finite encoder state"),
    ("temperature", "temperature exceeded"),
    ("voltage", "supply voltage outside"),
    ("stale", "state read exceeded timeout"),
])
def test_invalid_encoder_read_does_not_replace_valid_cache(experimental_profile, fault, message):
    class InvalidTransport(FakeTransport):
        reads = 0

        def state(self):
            self.reads += 1
            result = super().state()
            if self.reads == 2:
                if fault == "position_shape":
                    result["position"] = np.zeros(6)
                elif fault == "position_nan":
                    result["position"][0] = np.nan
                elif fault == "velocity_nan":
                    result["velocity"][0] = np.nan
                elif fault == "temperature":
                    result["temperature_c"][0] = 60
                elif fault == "voltage":
                    result["voltage_v"][0] = 7
                elif fault == "stale":
                    result["stamp"] -= 1
            else:
                self.valid_stamp = result["stamp"]
            return result

    transport = InvalidTransport()
    runtime = CurrentRuntime(experimental_profile, live=True, teleop=True, transport=transport)
    runtime._run()
    assert runtime._state["stamp"] == transport.valid_stamp
    assert transport.calls.count("write") == 1
    assert transport.calls[-2:] == ["disable", "close"]
    with pytest.raises(RuntimeError, match=message):
        runtime.state()


@pytest.mark.parametrize("delay_phase", ["idle", "read", "write"])
def test_delayed_control_tick_unloads(experimental_profile, monkeypatch, delay_phase):
    from types import SimpleNamespace
    from lerobot_robot_ufactory.current_control.control import runtime as runtime_module

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic=lambda: clock.now))

    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            clock.now += timeout + (0.055 if delay_phase == "idle" else 0)

    class TimedTransport(FakeTransport):
        reads = 0
        writes = 0

        def state(self):
            self.reads += 1
            stamp = clock.now
            state = super().state()
            state["stamp"] = stamp
            clock.now += 0.060 if delay_phase == "read" and self.reads == 2 else 0.004
            return state

        def currents(self, current, gripper=None):
            self.writes += 1
            super().currents(current, gripper)
            if delay_phase == "write" and self.writes == 2:
                clock.now += 0.060

    transport = TimedTransport()
    runtime = CurrentRuntime(experimental_profile, live=True, teleop=True, transport=transport)
    runtime._stop = StopEvent()
    # Run deterministically without a physical device or wall-clock sleeps.
    runtime._run()
    assert runtime.status()["state"] == "fault"
    assert transport.calls[-2:] == ["disable", "close"]
    if delay_phase == "idle":
        assert transport.reads == transport.writes == 1
        assert "Invalid or stale control interval" in runtime.status()["error"]
    elif delay_phase == "read":
        assert transport.writes == 1
        assert "state read exceeded timeout" in runtime.status()["error"]
    else:
        assert "control transaction exceeded timeout" in runtime.status()["error"]
    # Closing is idempotent, while reads and explicit fault checks still fail.
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError):
        runtime.state()


@pytest.mark.parametrize("failure_phase", ["disable", "close"])
def test_cleanup_failure_is_not_suppressed_by_close(profile, failure_phase):
    class BrokenCleanupTransport(FakeTransport):
        def disable(self):
            super().disable()
            if failure_phase == "disable":
                raise RuntimeError("injected unload failure")

        def close(self):
            super().close()
            if failure_phase == "close":
                raise RuntimeError("injected close failure")

    transport = BrokenCleanupTransport()
    runtime = CurrentRuntime(profile, transport=transport)
    runtime.start()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runtime.close()
    assert transport.calls[-2:] == ["disable", "close"]


def test_teleop_fault_cleanup_does_not_resubmit_gripper_requests(experimental_profile, caplog):
    from lerobot_robot_ufactory.current_control.control.runtime import RuntimeAgent
    from lerobot_robot_ufactory.scripts.uf_lerobot_record import _RecordingCleanup
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    transport = FakeTransport()
    transport.stale = True
    runtime = CurrentRuntime(experimental_profile, live=True, teleop=True, transport=transport)
    with pytest.raises(RuntimeError, match="timeout"):
        runtime.start()
    teleop = GelloTeleop(GelloTeleopConfig())
    teleop._current_runtime = runtime
    teleop.gello_agent = RuntimeAgent(RuntimeRobot(runtime, [1] * 7, [8, 0, -42]))
    teleop._is_connected = True
    caplog.clear()
    with pytest.raises(RuntimeError, match="state read exceeded timeout") as failure:
        with _RecordingCleanup(object(), teleop, None, None):
            runtime.raise_if_failed()
    assert failure.value.__cause__ is runtime._error
    # Repeated disconnects remain harmless after the context has closed it.
    teleop.disconnect()
    assert not teleop.is_connected
    assert teleop._current_runtime is None
    assert transport.calls.count("disable") == transport.calls.count("close") == 1
    assert not any(record.levelname == "ERROR" for record in caplog.records)




def test_device_fault_does_not_stop_other_worker(profile):
    ta, tb = FakeTransport(), FakeTransport()
    a = CurrentRuntime(profile, transport=ta)
    b = CurrentRuntime(profile, transport=tb)
    a.start()
    b.start()
    try:
        ta.stale = True
        time.sleep(0.12)
        assert a.status()["state"] == "fault"
        assert b.status()["state"] == "observe"
        assert np.isfinite(b.state()["position"]).all()
    finally:
        b.stop()


@pytest.mark.parametrize(
    "method,args,expected_request,mode",
    [
        ("probe_gripper_dynamixel", (), ("probe",), 3),
        ("enable_gripper_current_mode", (80,), ("enable", 80), 0),
    ],
)
def test_runtime_robot_gripper_adapter(method, args, expected_request, mode):
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_adapter import (
        GripperDynamixelInfo,
    )

    info = {
        "dxl_id": 8,
        "model_number": 1190,
        "model_name": "XL330-M077-T",
        "operating_mode": mode,
        "current_limit_raw": 80,
        "current_limit_ma": 80.0,
        "current_unit_ma": 1.0,
    }
    calls = []

    class OfflineRuntime:
        def request(self, *request):
            calls.append(request)
            return info.copy()

    robot = RuntimeRobot(OfflineRuntime(), [1] * 7, [8, 198.28125, 155.75])
    result = getattr(robot, method)(*args)

    assert calls == [expected_request]
    assert isinstance(result, GripperDynamixelInfo)
    assert result == GripperDynamixelInfo(**info)


def test_follower_rebase_and_pause_preserve_physical_state(profile):
    transport = FakeTransport()
    runtime = CurrentRuntime(profile, transport=transport)
    runtime.start()
    try:
        robot = RuntimeRobot(runtime, [1] * 7, [8, 0, -42])
        raw = runtime.get_joints()
        physical_before = profile.model_state(raw, np.zeros(8))[0]
        robot._joint_offsets[:7] = 1.3
        robot.set_torque_mode(False)
        assert np.allclose(robot.get_joint_state()[:7], -1.3)
        assert np.array_equal(
            profile.model_state(runtime.get_joints(), np.zeros(8))[0], physical_before
        )
        assert runtime.status()["state"] == "observe"
    finally:
        runtime.stop()


def test_teleop_uses_one_runtime_and_pause_keeps_support(profile, tmp_path, monkeypatch):
    import gello.agents.gello_agent as upstream

    from lerobot_robot_ufactory.current_control.control import runtime as runtime_module
    from lerobot_robot_ufactory.current_control.config import CurrentControlConfig
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
        GelloTeleopConfig,
    )

    p = commissioned(profile, tmp_path)
    transport = FakeTransport()
    actual = CurrentRuntime(p, live=True, transport=transport)
    monkeypatch.setattr(runtime_module, "CurrentRuntime", lambda *a, **kw: actual)

    def forbidden(*args, **kwargs):
        raise AssertionError("A second upstream serial driver must never be constructed")

    monkeypatch.setattr(upstream, "GelloAgent", forbidden)
    config = GelloTeleopConfig(
        port=p.port, current_control=CurrentControlConfig(True, str(p.path))
    )
    teleop = GelloTeleop(config)
    try:
        teleop.connect()
        obs = {f"J{i}.pos": 0.25 for i in range(1, 8)} | {"gripper.pos": 0.5}
        teleop.set_teleop_enabled(True, obs)
        assert teleop.get_action()["J1.pos"] == pytest.approx(0.25)
        teleop.set_teleop_enabled(False)
        assert actual.status()["state"] == "active"
        assert transport.calls.count("open") == 1
        assert "disable" not in transport.calls
        assert np.array_equal(p.zeros, np.zeros(7))
    finally:
        teleop.disconnect()
    assert actual.status()["state"] == "stopped"


def test_arm_only_profile_does_not_modify_shared_gripper_configuration(profile):
    selected = profile.arm_only()
    assert selected.gripper_id == -1
    assert selected.all_ids == tuple(range(1, 8))
    assert len(selected.model_numbers) == 7
    assert selected.data["gripper_id"] == -1
    assert profile.gripper_id == profile.data["gripper_id"] == 8
    assert profile.all_ids == tuple(range(1, 9))
    assert len(profile.data["model_numbers"]) == 8


def missing_id_transport_factory(missing_id, transports):
    """Use the actual sync-read parser, injecting an absent motor reply."""
    from types import SimpleNamespace
    from dynamixel_sdk import COMM_RX_TIMEOUT

    class MissingReplyTransport(XL330Transport):
        def __init__(self, profile):
            super().__init__(profile)
            self.read_ids = []
            self.calls = []
            transports.append(self)

        def open(self):
            self.calls.append("open")
            self.reader = SimpleNamespace(txPacket=lambda: 0)
            payload = bytearray(21)
            payload[18:20] = (50).to_bytes(2, "little")
            payload[20] = 25

            def read_reply(port, dxl_id, length):
                self.read_ids.append(dxl_id)
                if dxl_id == missing_id:
                    return [], COMM_RX_TIMEOUT, 0
                return payload, 0, 0

            self.packet = SimpleNamespace(readRx=read_reply)

        def enable(self, **kwargs):
            self.calls.append("enable")

        def currents(self, current, gripper=None):
            self.calls.append("write")

        def disable(self):
            self.calls.append("disable")

        def close(self):
            self.calls.append("close")

    return MissingReplyTransport


def test_keyboard_recording_can_discard_and_realign_without_reading_id8(profile, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lerobot_robot_ufactory.current_control.config import CurrentControlConfig
    from lerobot_robot_ufactory.current_control.control import runtime as runtime_module
    from lerobot_robot_ufactory.scripts.uf_lerobot_record import _discard_current_episode
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    p = commissioned(profile, tmp_path)
    transports = []
    monkeypatch.setattr(runtime_module, "XL330Transport", missing_id_transport_factory(8, transports))
    teleop = GelloTeleop(GelloTeleopConfig(
        port=p.port, current_control=CurrentControlConfig(True, str(p.path)),
        gripper_control_mode="keyboard",
    ))
    assert teleop._dynamixel_robo_config.gripper_config is None
    dataset = SimpleNamespace(
        root=tmp_path / "dataset",
        meta=SimpleNamespace(features={"wrist": {"dtype": "image"}}),
        episode_buffer={"episode_index": 6, "size": 1, "action": [0.0]},
        _wait_image_writer=lambda: None,
    )
    dataset._get_image_file_dir = lambda index, key: dataset.root / "images" / key / f"episode_{index:06d}"
    saved_dir = dataset._get_image_file_dir(5, "wrist")
    saved_dir.mkdir(parents=True)
    (saved_dir / "frame-000000.png").write_bytes(b"saved episode")
    try:
        teleop.connect()
        runtime = teleop._current_runtime
        assert runtime.profile.all_ids == tuple(range(1, 8))
        assert runtime.get_joints().shape == (7,)
        for attempt in range(3):
            pose = {f"J{i}.pos": attempt / 10 for i in range(1, 8)} | {"gripper.pos": 0.6}
            teleop.set_teleop_enabled(True, pose)
            action = teleop.get_action()
            assert action["J1.pos"] == pytest.approx(attempt / 10)
            assert action["gripper.pos"] == pytest.approx(0.6)
            current_dir = dataset._get_image_file_dir(6, "wrist")
            current_dir.mkdir(parents=True)
            (current_dir / "frame-000000.png").write_bytes(b"discarded episode")
            teleop.set_teleop_enabled(False)
            _discard_current_episode(dataset)
            assert not current_dir.exists()
            assert dataset.episode_buffer["episode_index"] == 6
            assert dataset.episode_buffer["size"] == 0
            assert (saved_dir / "frame-000000.png").read_bytes() == b"saved episode"
            runtime.raise_if_failed()
        assert transports[0].calls.count("open") == 1
        assert 8 not in transports[0].read_ids
    finally:
        teleop.disconnect()
    assert transports[0].calls.count("disable") == transports[0].calls.count("close") == 1


@pytest.mark.parametrize("gripper_mode,authorize_id8,missing_id", [
    ("gello", False, 8), ("keyboard", True, 8), ("keyboard", False, 2),
])
def test_needed_motor_timeouts_still_stop_compensation(profile, tmp_path, monkeypatch,
                                                     gripper_mode, authorize_id8, missing_id):
    from lerobot_robot_ufactory.current_control.config import CurrentControlConfig
    from lerobot_robot_ufactory.current_control.control import runtime as runtime_module
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    p = commissioned(profile, tmp_path)
    transports = []
    monkeypatch.setattr(runtime_module, "XL330Transport",
                        missing_id_transport_factory(missing_id, transports))
    teleop = GelloTeleop(GelloTeleopConfig(
        port=p.port, current_control=CurrentControlConfig(True, str(p.path)),
        gripper_control_mode=gripper_mode,
        gripper_current_control_enabled=authorize_id8,
        gripper_current_limit_ma=80 if authorize_id8 else None,
    ))
    with pytest.raises(RuntimeError, match=f"sync read reply ID{missing_id}.*-3001"):
        teleop.connect()
    assert not teleop.is_connected
    assert transports[0].calls[-2:] == ["disable", "close"]
    assert "write" not in transports[0].calls




def test_teleop_log_failure_requests_unload(experimental_profile, tmp_path, monkeypatch):
    from lerobot_robot_ufactory.current_control.monitoring.logging import CurrentLog

    fake = FakeTransport()
    runtime = CurrentRuntime(experimental_profile, live=True, experimental=True, teleop=True,
                             transport=fake)
    logger = CurrentLog(runtime, tmp_path)
    runtime.start()
    original = logger._drain

    def broken_drain():
        original()
        raise OSError("injected disk failure")

    monkeypatch.setattr(logger, "_drain", broken_drain)
    logger.start()
    runtime._thread.join(timeout=1)
    try:
        assert not runtime._thread.is_alive()
        assert fake.calls[-2:] == ["disable", "close"]
        with pytest.raises(RuntimeError, match="disk failure"):
            logger.raise_if_failed()
    finally:
        runtime.stop()
        with pytest.raises(RuntimeError, match="disk failure"):
            logger.stop()


def test_teleop_gravity_fault_blocks_follower_reads(experimental_profile, monkeypatch):
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    fake = FakeTransport()
    runtime = CurrentRuntime(experimental_profile, live=True, experimental=True, teleop=True,
                             transport=fake)
    teleop = GelloTeleop(GelloTeleopConfig())
    teleop._current_runtime = runtime
    teleop._teleop_enabled = True
    runtime.start()
    fake.stale = True
    runtime._thread.join(timeout=1)
    assert not runtime._thread.is_alive()
    with pytest.raises(RuntimeError, match="timeout"):
        teleop.get_action()
    assert fake.calls[-2:] == ["disable", "close"]






class Packet:
    def __init__(self, profile, fail_on=None):
        self.table = {}
        self.writes = []
        self.fail_on = fail_on
        for i in profile.all_ids:
            self.table.update({(i, 11): 3, (i, 64): 0, (i, 38): 1000, (i, 98): 0, (i, 70): 0})

    def write1ByteTxRx(self, port, i, addr, value):
        return self._write(i, addr, value)

    def write2ByteTxRx(self, port, i, addr, value):
        return self._write(i, addr, value)

    def _write(self, i, addr, value):
        self.writes.append((i, addr, value))
        if (i, addr, value) == self.fail_on:
            return -1, 0
        self.table[i, addr] = value
        if addr == 11:
            self.table[i, 102] = self.table[i, 38]  # Hardware resets goal on mode change.
        return 0, 0

    def read1ByteTxRx(self, port, i, addr):
        return self.table.get((i, addr), 0), 0, 0

    read2ByteTxRx = read1ByteTxRx


def setup_transport(profile, tmp_path, fail_on=None):
    p = commissioned(profile, tmp_path)
    t = XL330Transport(p)
    # Isolate host preflight from the actual USB adapter in hardware-free tests.
    t.latency_timer_path = tmp_path / "latency_timer"
    t.latency_timer_path.write_text("1\n")
    t.packet = Packet(p, fail_on)
    t.info = [
        {
            "id": i,
            "mode": 3,
            "watchdog": 0,
            "current_limit_raw": 1000,
            "hardware_error": 0,
            "torque_enabled": 0,
        }
        for i in p.all_ids
    ]
    return t


@pytest.mark.parametrize("latency", [2, 16])
def test_slow_usb_is_rejected_before_any_motor_write(profile, tmp_path, latency):
    t = setup_transport(profile, tmp_path)
    t.latency_timer_path.write_text(f"{latency}\n")
    with pytest.raises(RuntimeError, match=f"USB latency is {latency} ms"):
        t.enable()
    assert t.packet.writes == []
    assert t.touched == []


def test_missing_usb_latency_is_rejected_before_motor_write(profile, tmp_path):
    t = setup_transport(profile, tmp_path)
    t.latency_timer_path.unlink()
    with pytest.raises(RuntimeError, match="Cannot verify"):
        t.enable()
    assert t.packet.writes == []


def test_watchdog_cleanup_stops_all_motors_before_clearing_and_zeroing(profile, tmp_path):
    t = setup_transport(profile, tmp_path)
    t.enable()
    original_write = t.packet._write

    def locked_write(i, addr, value):
        # Reproduce the real Access Error when a goal register is locked.
        if addr == 102 and t.packet.table[i, 98] == 255:
            return 0, 7
        assert not (addr == 98 and value == 0) or t.packet.table[i, 64] == 0
        return original_write(i, addr, value)

    t.packet._write = locked_write
    for i in range(1, 8):
        t.packet.table[i, 98] = 255
    t.packet.writes.clear()
    t.disable()
    assert t.packet.writes[:7] == [(i, 64, 0) for i in range(1, 8)]
    assert all(t.packet.table[i, 64] == 0 for i in range(1, 8))
    assert all(t.packet.table[i, 98] == 0 for i in range(1, 8))
    assert all(t.packet.table[i, 102] == 0 for i in range(1, 8))
    assert all(t.packet.table[i, 11] == 3 for i in range(1, 8))


def test_watchdog_error_identifies_register_values(profile, tmp_path):
    t = setup_transport(profile, tmp_path)
    t.enable()
    t.packet.table[1, 98] = 255
    with pytest.raises(RuntimeError, match="hardware_error=0, watchdog=-1"):
        t.health()


def test_all_arm_zeroes_precede_any_torque_enable(profile, tmp_path):
    t = setup_transport(profile, tmp_path)
    t.enable()
    first_enable = next(i for i, row in enumerate(t.packet.writes) if row[1:] == (64, 1))
    for dxl_id in range(1, 8):
        assert (dxl_id, 102, 0) in t.packet.writes[:first_enable]
    assert not any(row[0] == 8 for row in t.packet.writes)
    t.disable()
    assert all(t.packet.table[i, 64] == 0 and t.packet.table[i, 11] == 3 for i in range(1, 8))


def test_partial_enable_failure_cleans_every_touched_motor(profile, tmp_path):
    t = setup_transport(profile, tmp_path, fail_on=(4, 64, 1))
    with pytest.raises(RuntimeError):
        t.enable()
    t.disable()
    assert all(t.packet.table[i, 64] == 0 for i in range(1, 8))


def test_gripper_disable_does_not_disable_arm(profile, tmp_path):
    t = setup_transport(profile, tmp_path)
    t.enable()
    t.enable_gripper(80)
    assert t.gripper_limit == 0.08
    t.disable_gripper()
    assert t.packet.table[8, 64] == 0
    assert all(t.packet.table[i, 64] == 1 for i in range(1, 8))
    t.disable()


def test_signed_registers():
    assert signed(65535, 16) == -1
    assert signed(0xFFFFFFFF, 32) == -1
    assert signed(32767, 16) == 32767


@pytest.mark.parametrize("error,payload", [(128, [0] * 21), (0, [0] * 20)])
def test_sync_read_rejects_hardware_alerts_and_short_packets(profile, error, payload):
    from types import SimpleNamespace

    transport = XL330Transport(profile)
    transport.reader = SimpleNamespace(txPacket=lambda: 0)
    transport.packet = SimpleNamespace(readRx=lambda *args: (payload, 0, error))
    with pytest.raises(RuntimeError):
        transport.state()


def test_sync_read_preserves_signed_units_and_all_eight_ids(profile):
    from types import SimpleNamespace

    payload = bytearray(21)
    payload[0:2] = (-50).to_bytes(2, "little", signed=True)
    payload[2:6] = (-10).to_bytes(4, "little", signed=True)
    payload[6:10] = (-2048).to_bytes(4, "little", signed=True)
    payload[18:20] = (50).to_bytes(2, "little")
    payload[20] = 25
    transport = XL330Transport(profile)
    transport.reader = SimpleNamespace(txPacket=lambda: 0)
    transport.packet = SimpleNamespace(readRx=lambda *args: (payload, 0, 0))
    state = transport.state()
    assert len(state["position"]) == 8
    assert np.allclose(state["position"], -np.pi)
    assert np.allclose(state["velocity"], -10 * 0.229 * 2 * np.pi / 60)
    assert state["current_a"] == [-0.05] * 8
    assert state["voltage_v"] == [5.0] * 8
    assert state["temperature_c"] == [25] * 8


def test_current_packets_are_signed_bounded_and_never_address_passive_gripper(profile):
    class Writer:
        def __init__(self):
            self.rows = []
            self.sent = []

        def addParam(self, dxl_id, data):
            self.rows.append((dxl_id, data))
            return True

        def txPacket(self):
            self.sent = self.rows.copy()
            return 0

        def clearParam(self):
            self.rows.clear()

    transport = XL330Transport(profile)
    transport.writer = Writer()
    transport.currents(np.array([-10, 10, 0.0019, 0, 0, 0, 0]))
    sent = transport.writer.sent
    assert [item[0] for item in sent] == list(range(1, 8))
    decoded = [signed(data[0] | (data[1] << 8), 16) for _, data in sent]
    assert decoded[:3] == [-50, 50, 1]
    assert transport.writer.rows == []


def test_recording_unloads_before_waiting_for_saver():
    from lerobot_robot_ufactory.scripts.uf_lerobot_record import _RecordingCleanup

    events = []

    class Teleop:
        is_connected = True

        def stop_current_control(self):
            events.append("unload")

        def disconnect(self):
            events.append("teleop_disconnect")

    class Robot:
        _is_connected = True

        def disconnect(self):
            events.append("robot_disconnect")

    class Saver:
        def close(self):
            events.append("save")

    with _RecordingCleanup(Robot(), Teleop(), None, Saver()):
        pass
    assert events.index("unload") < events.index("save") < events.index("robot_disconnect")




@pytest.mark.parametrize("mode", ["teleop", "tuning"])
def test_live_runtime_does_not_poll_health_registers(experimental_profile, mode):
    class NoHealthPollingTransport(FakeTransport):
        def health(self):
            raise AssertionError("Periodic health-register polling must remain disabled")

    transport = NoHealthPollingTransport()
    runtime = CurrentRuntime(
        experimental_profile, live=True, experimental=True,
        transport=transport, **{mode: True},
    )
    try:
        runtime.start()
        # Cross the old 0.5-second health polling interval.
        time.sleep(0.6)
        runtime.raise_if_failed()
        assert runtime.status()["state"] == "active"
        assert transport.calls.count("write") > 1
    finally:
        runtime.stop()
    assert transport.calls[-2:] == ["disable", "close"]




@pytest.mark.parametrize("values", [[None]*6, [True]*7, [float("nan")]*7, [0.1]*7])
def test_constant_current_profile_validation(profile, values):
    data = profile.data.copy()
    data["constant_current_a"] = values
    profile.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        DeviceProfile(profile.path)


def test_fixed_support_and_damping_never_assist_motion(profile):
    profile.constant_current_a = [0, -.05, 0, .08, 0, 0, 0]
    profile.constant_damping_a = np.array([0, 0, .002, 0, 0, 0, .002])
    profile.limits[:] = 1
    c = CurrentController(profile)
    for velocity in [1.0, -1.0, 0.0, .024, -.024, .2, -.2]:
        for step in range(250):
            out, record = c.compute(np.full(8, step*.01), np.full(8, velocity), .01, 3)
            assert out[2]*velocity <= 0 and out[6]*velocity <= 0
            assert np.all(out[[0,4,5]] == 0)
            if abs(velocity) <= .05:
                assert out[2] == out[6] == 0
        expected = -.002*np.sign(velocity) if abs(velocity)>.05 else 0
        assert out[2] == pytest.approx(expected)
        assert out[6] == pytest.approx(expected)
        assert out[1] == pytest.approx(-.05)
        assert out[3] == pytest.approx(.08)
