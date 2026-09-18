import threading

import numpy as np
import pytest

from lerobot_robot_ufactory.teleoperators.gello_teleop import arm_adapter as module
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_adapter import SafeDynamixelDriver
from lerobot_robot_ufactory.utils.arm_feedback import ArmFeedbackConfig


class Packet:
    def __init__(self):
        self.models = {j: 1190 for j in range(1, 9)}
        self.registers = {
            (j, address): value
            for j in range(1, 9)
            for address, value in [(11, 3), (38, 1750), (64, 0), (102, 0), (70, 0), (146, 25)]
        }
        self.writes = []
        self.pings = []
        self.fail_mode_id = None

    def ping(self, port, motor):
        self.pings.append(motor)
        return self.models[motor], 0, 0

    def read1ByteTxRx(self, port, motor, address):
        return self.registers[motor, address], 0, 0

    read2ByteTxRx = read1ByteTxRx

    def write1ByteTxRx(self, port, motor, address, value):
        self.writes.append((motor, address, value))
        if motor == self.fail_mode_id and address == 11 and value == 0:
            return -1, 0
        self.registers[motor, address] = value
        return 0, 0

    write2ByteTxRx = write1ByteTxRx

    def getTxRxResult(self, code):
        return "mock failure"

    def getRxPacketError(self, code):
        return "mock failure"


class Sync:
    fail = False

    def __init__(self, port, packet, address, size):
        assert (address, size) == (102, 2)
        self.params = {}
        self.sent = []

    def addParam(self, motor, data):
        self.params[motor] = data
        return True

    def txPacket(self):
        self.sent.append(dict(self.params))
        return -1 if self.fail else 0

    def clearParam(self):
        self.params.clear()


def setup(monkeypatch, mask=(True,) * 7):
    monkeypatch.setattr(module, "GroupSyncWrite", Sync)
    d = SafeDynamixelDriver.__new__(SafeDynamixelDriver)
    d._ids, d._is_fake, d._lock = tuple(range(1, 9)), False, threading.Lock()
    d._portHandler, d._packetHandler = object(), Packet()
    c = ArmFeedbackConfig(
        enabled=True,
        observe_only=False,
        baseline_verified=True,
        sign_verified=True,
        enabled_joints=mask,
        current_limit_ma=(5,) * 7,
    )
    return module.GelloArmFeedbackAdapter(d, c), d._packetHandler


def test_discovery_sequence_startup_and_shutdown_preserve_id8(monkeypatch):
    a, packet = setup(monkeypatch)
    a.enable()
    assert packet.pings == list(range(1, 8))
    assert packet.writes[:4] == [(1, 64, 0), (1, 11, 0), (1, 102, 0), (1, 64, 1)]
    assert a.active
    a.disable()
    for j in range(1, 8):
        assert packet.registers[j, 102] == 0
        assert packet.registers[j, 64] == 0
        assert packet.registers[j, 11] == 3
    assert not any(j == 8 for j, _, _ in packet.writes)


def test_unknown_model_rejected_before_any_write(monkeypatch):
    a, packet = setup(monkeypatch)
    packet.models[7] = 999
    with pytest.raises(RuntimeError, match="unsupported"):
        a.enable()
    assert not packet.writes and not a.active


def test_syncwrite_signed_packing_and_clamp(monkeypatch):
    a, packet = setup(monkeypatch)
    a.enable()
    result = a.write([-100, 100, 2, -2, 0, 3, 4])
    np.testing.assert_allclose(result, [-5, 5, 2, -2, 0, 3, 4])
    assert a.writer.sent[-1][1] == [251, 255]
    assert a.writer.sent[-1][2] == [5, 0]
    assert len(a.writer.sent) == 1 and not a.writer.params


def test_single_joint_does_not_enable_other_joints(monkeypatch):
    a, packet = setup(monkeypatch, (False,) * 6 + (True,))
    a.enable()
    a.write(np.ones(7))
    assert set(a.writer.sent[-1]) == {7}
    assert {j for j, _, _ in packet.writes} == {7}


def test_model_specific_quantization_never_exceeds_ma_limit(monkeypatch):
    a, packet = setup(monkeypatch)
    packet.models[1], packet.registers[1, 38] = 1030, 1193
    a.enable()
    result = a.write(np.ones(7) * 5)
    assert result[0] == 2.69 and np.all(result <= 5)


@pytest.mark.parametrize("failure", ["sync", "nan", "expired", "health"])
def test_write_failure_disables_and_zeroes_all_selected(monkeypatch, failure):
    a, packet = setup(monkeypatch)
    a.enable()
    values, deadline = np.ones(7), None
    if failure == "sync":
        a.writer.fail = True
    elif failure == "nan":
        values[0] = np.nan
    elif failure == "expired":
        deadline = 1
    else:
        packet.registers[1, 70] = 32
        a.last_health_ns = 0
    with pytest.raises((ValueError, RuntimeError)):
        a.write(values, deadline_ns=deadline)
    assert not a.active and not a.writer.params
    for j in range(1, 8):
        assert packet.registers[j, 64] == 0 and packet.registers[j, 102] == 0


def test_partial_initialization_failure_rolls_back(monkeypatch):
    a, packet = setup(monkeypatch)
    packet.fail_mode_id = 4
    with pytest.raises(RuntimeError):
        a.enable()
    assert not a.active
    for j in range(1, 5):
        assert packet.registers[j, 64] == 0
        assert packet.registers[j, 11] == 3
        assert packet.registers[j, 102] == 0


def test_observe_discovery_never_writes(monkeypatch):
    a, packet = setup(monkeypatch)
    a.config.observe_only = True
    a.discover()
    with pytest.raises(RuntimeError):
        a.enable()
    a.disable()
    assert not packet.writes


def test_shared_lock_serializes_writer(monkeypatch):
    a, packet = setup(monkeypatch)
    a.enable()
    done = threading.Event()
    a.driver._lock.acquire()
    worker = threading.Thread(target=lambda: (a.write(np.ones(7)), done.set()))
    worker.start()
    assert not done.wait(0.02)
    a.driver._lock.release()
    worker.join(1)
    assert done.is_set()


def test_timed_reader_signed_velocity_and_position(monkeypatch):
    adapter, _ = setup(monkeypatch)
    d = adapter.driver
    d._arm_timed_reader = True

    class StopAfterOne:
        calls = 0

        def wait(self, delay):
            self.calls += 1
            return self.calls > 1

    class Read:
        def txRxPacket(self):
            return 0

        def isAvailable(self, *args):
            return True

        def getData(self, motor, address, size):
            return 0xFFFFFFFF if address == 128 else 2048

    d._stop_thread = StopAfterOne()
    d._groupSyncRead = Read()
    d._read_joint_states()
    timestamp, q, v, error, latency, sequence, period = d.arm_state_snapshot()
    np.testing.assert_allclose(q, np.pi)
    np.testing.assert_allclose(v, -0.229 * 2 * np.pi / 60)
    assert timestamp > 0 and sequence == 1 and error is None and latency >= 0


def test_timed_reader_fault_latched_after_recovery(monkeypatch):
    adapter, _ = setup(monkeypatch)
    d = adapter.driver
    d._arm_state_snapshot = (1, np.zeros(7), np.zeros(7), None, 0)
    d._arm_read_fault = "earlier read failed"
    assert d.arm_state_snapshot()[3] == "earlier read failed"
