import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_robot_ufactory.gravity_compensation.config import DeviceProfile
from lerobot_robot_ufactory.gravity_compensation.control.model import CurrentController, GravityModel
from lerobot_robot_ufactory.gravity_compensation.control.runtime import GravityRuntime, RuntimeRobot
from lerobot_robot_ufactory.gravity_compensation.hardware.transport import XL330Transport, signed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile(tmp_path):
    original = ROOT / "config/gravity/gello_A_working.yaml"
    data = yaml.safe_load(original.read_text())
    data["urdf"] = str((original.parent / "model/xarm7_gello.urdf").resolve())
    # Keep the unit-test device conservative and neutral without shipping a
    # second fixture profile under config/gravity.
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


class ConstantModel:
    def gravity(self, q):
        return np.arange(1, 8) * 0.001


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
    with pytest.raises(ValueError, match="not commissioned"):
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


def test_experimental_permission_does_not_commission(experimental_profile):
    p = experimental_profile
    p.validate_live(experimental=True)
    with pytest.raises(ValueError, match="not commissioned"):
        p.validate_live()
    assert not any(p.data["commissioning"].values())


@pytest.mark.parametrize("field,value", [
    ("gain", float("inf")), ("gain", float("nan")), ("gain", -0.01), ("limits", np.full(7, 1.01)),
    ("slew", np.full(7, 0.051)), ("ramp_s", 1.9),
    ("temperature_limit_c", 51), ("state_timeout_s", 0.051),
    ("watchdog_ms", 120), ("baudrate", 57600),
])
def test_experiment_rejects_unsafe_parameters(experimental_profile, field, value):
    setattr(experimental_profile, field, value)
    with pytest.raises(ValueError):
        experimental_profile.validate_live(experimental=True)


def test_experiment_requires_alignment_and_exact_model(experimental_profile):
    p = experimental_profile
    p.data["alignment_user_accepted"] = False
    with pytest.raises(ValueError, match="pose alignment"):
        p.validate_experiment()
    p.data["alignment_user_accepted"] = True
    p.data["urdf_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        p.validate_experiment()




@pytest.mark.parametrize("gain", [0, 0.05, 0.1, 1.0, 2.0])
def test_experiment_gain_has_no_upper_cap(experimental_profile, gain):
    experimental_profile.gain = gain
    experimental_profile.validate_experiment()




@pytest.mark.parametrize("joint", [5, 6])
def test_joint_zero_gain_preserves_damping_and_other_joints(profile, joint):
    profile.gain = 0.3
    position = np.zeros(8)
    velocity = np.zeros(8)
    axis = joint - 1
    velocity[axis] = 0.2
    _, baseline = CurrentController(profile, ConstantModel()).compute(position, velocity, 0.01, 10)
    setattr(profile, f"j{joint}_gain", 0.0)
    _, result = CurrentController(profile, ConstantModel()).compute(position, velocity, 0.01, 10)
    others = [i for i in range(7) if i != axis]
    np.testing.assert_allclose(np.array(result["requested_a"])[others], np.array(baseline["requested_a"])[others])
    assert result["requested_a"][axis] == pytest.approx(-profile.damping[axis] * velocity[axis] / profile.nm_per_amp[axis])
    assert result["requested_a"][axis] < 0
    expected = [0.3] * 7
    expected[axis] = 0.0
    assert result["gravity_gains"] == expected
    velocity[axis] = 0
    _, stationary = CurrentController(profile, ConstantModel()).compute(position, velocity, 0.01, 10)
    assert stationary["requested_a"][axis] == 0


def test_j2_can_be_reduced_without_changing_other_axes(profile):
    profile.gain = 0.3
    profile.slew[:] = 100
    position, velocity = np.zeros(8), np.zeros(8)
    _, baseline = CurrentController(profile, ConstantModel()).compute(position, velocity, 0.01, 10)
    profile.joint_gains = [0.3, 0.05, 0.3, 0.3, 0.3, 0.3, 0.3]
    _, light = CurrentController(profile, ConstantModel()).compute(position, velocity, 0.01, 10)
    assert light["requested_a"][1] == pytest.approx(baseline["requested_a"][1] / 6)
    others = [0, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(np.array(light["requested_a"])[others],
                               np.array(baseline["requested_a"])[others])
    assert light["damping_current_a"] == [0.0] * 7
    profile.joint_gains[1] = 0.0
    velocity[1] = 0.2
    _, off = CurrentController(profile, ConstantModel()).compute(position, velocity, 0.01, 10)
    assert off["gravity_current_a"][1] == 0
    assert off["damping_current_a"][1] < 0
    np.testing.assert_allclose(off["requested_a"],
                               np.array(off["gravity_current_a"]) + off["damping_current_a"])


def test_joint_gains_keep_legacy_overrides(profile):
    profile.joint_gains = [0.2] * 7
    profile.j5_gain, profile.j6_gain = 0.0, 0.1
    _, record = CurrentController(profile, ConstantModel()).compute(np.zeros(8), np.zeros(8), 0.01, 10)
    assert record["gravity_gains"] == [0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.2]




@pytest.mark.parametrize("gains", [[0.1] * 6, [0.1] * 8, [-0.1] * 7,
                                    [float("nan")] * 7, [True] * 7])
def test_joint_gains_reject_invalid_config(gains):
    from lerobot_robot_ufactory.gravity_compensation.config import GravityCompensationConfig

    with pytest.raises(ValueError):
        GravityCompensationConfig(enabled=True, profile_path="unused", joint_gains=gains)


def test_joint_gain_session_override_preserves_calibration(experimental_profile, monkeypatch):
    from lerobot_robot_ufactory.gravity_compensation import config as module

    gains = [0.3, 0.05, 0.3, 0.3, 0, 0.1, 0.3]
    monkeypatch.setattr(module, "DeviceProfile", lambda path: experimental_profile)
    config = module.GravityCompensationConfig(enabled=True, profile_path="unused",
                                              experimental=True, gain=0.3, joint_gains=gains)
    loaded = config.load_profile()
    assert loaded.joint_gains == gains
    assert not any(loaded.data["commissioning"].values())
    assert "joint_gains" not in loaded.data


def test_experiment_runtime_has_independent_time_limit(experimental_profile, monkeypatch):
    from lerobot_robot_ufactory.gravity_compensation.control import runtime

    assert runtime.EXPERIMENT_MAX_DURATION_S == 10.0
    monkeypatch.setattr(runtime, "EXPERIMENT_MAX_DURATION_S", 0.05)
    fake = FakeTransport()
    r = GravityRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake, model=ConstantModel())
    r.start()
    r._thread.join(timeout=1)
    assert not r._thread.is_alive()
    r.raise_if_failed()
    assert "write" in fake.calls
    assert fake.calls[-2:] == ["disable", "close"]
    assert all(record["experimental"] for record in r.drain_records())


@pytest.mark.parametrize("displacement_deg", [-45, -44, 44, 45])
def test_experiment_allows_motion_up_to_45_degrees(experimental_profile, monkeypatch, displacement_deg):
    from lerobot_robot_ufactory.gravity_compensation.control import runtime

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
    r = GravityRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake, model=ConstantModel())
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
    r = GravityRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake, model=ConstantModel())
    with pytest.raises(RuntimeError):
        r.start()
    r._thread.join(timeout=1)
    assert "write" not in fake.calls
    assert fake.calls[-2:] == ["disable", "close"]


@pytest.mark.parametrize("mode", ["teleop", "tuning"])
@pytest.mark.parametrize("displacement_deg", [-100, 100])
def test_continuous_support_allows_wide_motion(experimental_profile, monkeypatch, mode, displacement_deg):
    from lerobot_robot_ufactory.gravity_compensation.control import runtime as runtime_module

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
    r = GravityRuntime(experimental_profile, live=True, experimental=True,
                       transport=fake, model=ConstantModel(), **{mode: True})
    try:
        r.start()
        time.sleep(0.06)
        assert r.status()["state"] == "active"
        assert r.diagnostics()["record"]["q"][1] == pytest.approx(np.deg2rad(displacement_deg))
        assert "write" in fake.calls
    finally:
        r.stop()
    assert fake.calls[-2:] == ["disable", "close"]






def test_urdf_checksum_prevents_unreviewed_changes(profile, tmp_path):
    p = commissioned(profile, tmp_path)
    p.validate_live()
    p.urdf.write_text(p.urdf.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        p.validate_live()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_signs", [0] * 7),
        ("nm_per_amp", [0] * 7),
        ("current_limit_a", [2] * 7),
        ("encoder_zero_rad", [float("nan")] * 7),
        ("damping_nm_s_rad", [-1] * 7),
        ("gain", True),
    ],
)
def test_invalid_profiles_fail_before_hardware(profile, field, value):
    data = profile.data.copy()
    data[field] = value
    profile.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        DeviceProfile(profile.path)


def test_mapping_uses_power_consistent_torque_signs(profile):
    profile.signs = np.array([-1, 1, -1, 1, 1, -1, 1])
    profile.slew[:] = 100
    profile.damping[:] = 0
    model = ConstantModel()
    controller = CurrentController(profile, model)
    qdot = np.arange(8) * 0.1
    currents, record = controller.compute(np.zeros(8), qdot, 0.05, 3)
    motor_torque = currents * profile.nm_per_amp
    _, model_velocity = profile.model_state(np.zeros(8), qdot)
    assert np.dot(motor_torque, qdot[:7]) == pytest.approx(
        np.dot(profile.gain * model.gravity(None), model_velocity)
    )


def test_ramp_slew_and_hard_limit(profile):
    controller = CurrentController(profile, ConstantModel())
    first, _ = controller.compute(np.zeros(8), np.zeros(8), 0.05, 0)
    assert np.array_equal(first, np.zeros(7))
    profile.gain = 1
    profile.limits[:] = 0.002
    for _ in range(5):
        current, record = controller.compute(np.zeros(8), np.zeros(8), 0.05, 3)
        assert np.max(np.abs(current)) <= 0.002
    assert record["saturated"]


@pytest.mark.parametrize("dt", [0, -1, 1, float("nan")])
def test_bad_control_interval(profile, dt):
    with pytest.raises(RuntimeError):
        CurrentController(profile, ConstantModel()).compute(np.zeros(8), np.zeros(8), dt, 1)


def test_nan_state(profile):
    position = np.zeros(8)
    position[2] = float("nan")
    with pytest.raises(ValueError):
        CurrentController(profile, ConstantModel()).compute(position, np.zeros(8), 0.05, 1)


def test_observe_does_not_enable_or_write(profile):
    transport = FakeTransport()
    runtime = GravityRuntime(profile, transport=transport, model=ConstantModel())
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
    runtime = GravityRuntime(p, live=True, transport=transport, model=ConstantModel())
    with pytest.raises(RuntimeError, match="timeout"):
        runtime.start()
    assert transport.calls[-2:] == ["disable", "close"]
    assert "write" not in transport.calls


def test_write_failure_latches_fault(profile, tmp_path):
    p = commissioned(profile, tmp_path)
    transport = FakeTransport()
    transport.fail_write = True
    runtime = GravityRuntime(p, live=True, transport=transport, model=ConstantModel())
    with pytest.raises(RuntimeError, match="injected"):
        runtime.start()
    assert runtime.status()["state"] == "fault"
    assert transport.calls[-2:] == ["disable", "close"]


def test_slow_model_never_writes_stale_current(profile, tmp_path):
    class SlowModel:
        def gravity(self, q):
            time.sleep(0.07)
            return np.zeros(7)

    p = commissioned(profile, tmp_path)
    transport = FakeTransport()
    runtime = GravityRuntime(p, live=True, transport=transport, model=SlowModel())
    with pytest.raises(RuntimeError, match="stale"):
        runtime.start()
    assert "write" not in transport.calls


def test_device_fault_does_not_stop_other_worker(profile):
    ta, tb = FakeTransport(), FakeTransport()
    a = GravityRuntime(profile, transport=ta, model=ConstantModel())
    b = GravityRuntime(profile, transport=tb, model=ConstantModel())
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
    runtime = GravityRuntime(profile, transport=transport, model=ConstantModel())
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

    from lerobot_robot_ufactory.gravity_compensation.control import runtime as runtime_module
    from lerobot_robot_ufactory.gravity_compensation.config import GravityCompensationConfig
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
        GelloTeleopConfig,
    )

    p = commissioned(profile, tmp_path)
    transport = FakeTransport()
    actual = GravityRuntime(p, live=True, transport=transport, model=ConstantModel())
    monkeypatch.setattr(runtime_module, "GravityRuntime", lambda *a, **kw: actual)

    def forbidden(*args, **kwargs):
        raise AssertionError("A second upstream serial driver must never be constructed")

    monkeypatch.setattr(upstream, "GelloAgent", forbidden)
    config = GelloTeleopConfig(
        port=p.port, gravity_compensation=GravityCompensationConfig(True, str(p.path))
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


def test_experimental_teleop_keeps_session_gains_support_and_logs(experimental_profile, tmp_path, monkeypatch):
    import gello.agents.gello_agent as upstream
    from lerobot_robot_ufactory.gravity_compensation import config as config_module
    from lerobot_robot_ufactory.gravity_compensation.control import runtime as runtime_module
    from lerobot_robot_ufactory.gravity_compensation.config import GravityCompensationConfig
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    p = experimental_profile
    transport = FakeTransport()
    runtimes = []

    def create_runtime(profile, **kwargs):
        runtime = GravityRuntime(profile, transport=transport, model=ConstantModel(), **kwargs)
        runtimes.append(runtime)
        return runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Second serial connection")

    monkeypatch.setattr(config_module, "DeviceProfile", lambda path: p)
    monkeypatch.setattr(runtime_module, "GravityRuntime", create_runtime)
    monkeypatch.setattr(runtime_module, "EXPERIMENT_MAX_DURATION_S", 0.02)
    monkeypatch.setattr(upstream, "GelloAgent", forbidden)
    gravity = GravityCompensationConfig(
        enabled=True, profile_path=str(p.path), experimental=True,
        gain=0.8, j5_gain=0, j6_gain=0.1, log_dir=str(tmp_path / "logs"),
    )
    teleop = GelloTeleop(GelloTeleopConfig(port=p.port, gravity_compensation=gravity))
    try:
        teleop.connect()
        actual = runtimes[0]
        time.sleep(0.06)
        teleop.check_gravity_compensation()
        assert actual.teleop and actual.experimental
        assert actual.status()["state"] == "active"
        obs = {f"J{i}.pos": 0.25 for i in range(1, 8)} | {"gripper.pos": 0.5}
        teleop.set_teleop_enabled(True, obs)
        assert teleop.get_action()["J1.pos"] == pytest.approx(0.25)
        teleop.set_teleop_enabled(False)
        assert actual.status()["state"] == "active"
        assert transport.calls.count("open") == 1
        assert not any(p.data["commissioning"].values())
    finally:
        teleop.disconnect()
    assert transport.calls[-2:] == ["disable", "close"]
    rows = [json.loads(row) for row in teleop._gravity_log.path.read_text().splitlines()]
    assert rows and all(row["gravity_gains"] == [0.8, 0.8, 0.8, 0.8, 0.0, 0.1, 0.8] for row in rows)
    assert all(row["teleop"] and row["experimental"] for row in rows)
    assert actual.dropped_records == 0


def test_teleop_log_failure_requests_unload(experimental_profile, tmp_path, monkeypatch):
    from lerobot_robot_ufactory.gravity_compensation.monitoring.logging import GravityLog

    fake = FakeTransport()
    runtime = GravityRuntime(experimental_profile, live=True, experimental=True, teleop=True,
                             transport=fake, model=ConstantModel())
    logger = GravityLog(runtime, tmp_path)
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
    runtime = GravityRuntime(experimental_profile, live=True, experimental=True, teleop=True,
                             transport=fake, model=ConstantModel())
    teleop = GelloTeleop(GelloTeleopConfig())
    teleop._gravity_runtime = runtime
    teleop._teleop_enabled = True
    runtime.start()
    fake.stale = True
    runtime._thread.join(timeout=1)
    assert not runtime._thread.is_alive()
    with pytest.raises(RuntimeError, match="timeout"):
        teleop.get_action()
    assert fake.calls[-2:] == ["disable", "close"]


def test_teleop_gravity_configuration_parses_and_overrides(monkeypatch):
    from lerobot_robot_ufactory.scripts.uf_robot_teleop import get_cfg

    monkeypatch.setattr("sys.argv", [
        "teleop", "--config_path", str(ROOT / "config/gello/xarm7_gello_teleop_gravity.yaml"),
        "--teleop.gravity_compensation.gain=0.6", "--teleop.gravity_compensation.j5_gain=0.02",
    ])
    cfg = get_cfg()
    p = cfg.teleop.gravity_compensation.load_profile()
    assert p.gain == 0.6 and p.j5_gain == 0.02 and p.j6_gain is None
    assert p.joint_gains == [0.065, 0.15, 0.115, 0.15, 0.06, 0.1, 0.12]
    assert p.running_current_slew_a_s == [0.05, 0.12, 0.05, 0.12, 0.05, 0.05, 0.05]
    assert cfg.robot.cameras == {}
    assert cfg.teleop.port == p.port
    assert p.limits.tolist() == [1.0] * 7




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

        def stop_gravity_compensation(self):
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


def test_pinocchio_energy_gradient(profile):
    pytest.importorskip("pinocchio")
    model = GravityModel(profile)
    rng = np.random.default_rng(7)
    for q in rng.uniform(-1.5, 1.5, (20, 7)):
        numerical = []
        for axis in range(7):
            delta = np.zeros(7)
            delta[axis] = 1e-6
            numerical.append((model.potential(q + delta) - model.potential(q - delta)) / 2e-6)
        np.testing.assert_allclose(model.gravity(q), numerical, atol=1e-5, rtol=0)
