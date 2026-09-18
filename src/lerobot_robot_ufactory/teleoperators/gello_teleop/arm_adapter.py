"""Explicit ID1--7 current session on the EXISTING locked GELLO driver.

Model data: ROBOTIS eManual XL330-M077/M288, XC330-T288, XM430-W210.
These are register conversions, not calibrated motor torque constants.
"""

import time
from dataclasses import dataclass

import numpy as np
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite

from ...utils.arm_feedback import vector7


@dataclass(frozen=True)
class ArmMotorSpec:
    name: str
    current_unit_ma: float
    max_raw: int


ARM_MODELS = {
    1190: ArmMotorSpec("XL330-M077-T", 1.0, 1750),
    1200: ArmMotorSpec("XL330-M288-T", 1.0, 1750),
    1030: ArmMotorSpec("XM430-W210", 2.69, 1193),
    1220: ArmMotorSpec("XC330-T288-T", 1.0, 910),
}


class GelloArmFeedbackAdapter:
    def __init__(self, driver, config):
        self.driver = driver
        self.config = config
        self.active = False
        self.infos = []
        self.restore = {}
        self.writer = None
        self.last_health_ns = 0
        self.cleanup_errors = []
        self.temperature_c = np.zeros(7)
        self.last_transaction_ms = 0.0

    def discover(self):
        d = self.driver
        if d._is_fake or not set(range(1, 8)).issubset(d._ids):
            raise RuntimeError("arm current requires real IDs 1--7")
        infos = []
        with d._lock:
            for motor in range(1, 8):
                number, code, error = d._packetHandler.ping(d._portHandler, motor)
                d._check_sdk_result(code, error, f"ping ID{motor}")
                if number not in ARM_MODELS:
                    raise RuntimeError(f"unsupported arm motor ID{motor} model {number}")
                spec = ARM_MODELS[number]
                mode = d._read1_locked(motor, 11, "Operating Mode")
                limit = d._read2_locked(motor, 38, "Current Limit")
                if not 0 < limit <= spec.max_raw:
                    raise RuntimeError(f"invalid hardware current limit ID{motor}")
                infos.append(
                    dict(
                        id=motor,
                        model=number,
                        name=spec.name,
                        mode=mode,
                        hardware_limit_raw=limit,
                        current_unit_ma=spec.current_unit_ma,
                    )
                )
        self.infos = infos
        return infos

    def enable(self):
        if self.config.observe_only or not self.config.enabled:
            raise RuntimeError("arm current output is not authorized by configuration")
        if self.active or self.restore:
            raise RuntimeError("arm session already initialized")
        self.discover()  # all seven verified before any writes
        limits = vector7(self.config.current_limit_ma, "current_limit_ma")
        if np.any(limits < 0) or np.any(limits > 100):
            raise ValueError("arm current limits must be within 0--100 mA")
        self.units = np.array([info["current_unit_ma"] for info in self.infos])
        self.raw_limits = np.floor(limits / self.units).astype(int)
        for j, info in enumerate(self.infos):
            if self.config.enabled_joints[j] and not (
                0 < self.raw_limits[j] <= info["hardware_limit_raw"]
            ):
                raise ValueError(f"ID{j + 1} limit below resolution or above hardware limit")
        d = self.driver
        with d._lock:
            try:
                self.writer = GroupSyncWrite(d._portHandler, d._packetHandler, 102, 2)
                for j, info in enumerate(self.infos):
                    if not self.config.enabled_joints[j]:
                        continue
                    motor = j + 1
                    self.restore[motor] = info["mode"]
                    d._write1_locked(motor, 64, 0, "arm torque disable")
                    d._write1_locked(motor, 11, 0, "arm current mode")
                    if d._read1_locked(motor, 11, "arm mode verify") != 0:
                        raise RuntimeError(f"ID{motor} current mode verification failed")
                    # Mode switches can reset Goal Current. ZERO BEFORE torque enable.
                    d._write2_locked(motor, 102, 0, "arm startup zero")
                    if d._read2_locked(motor, 102, "arm zero verify") != 0:
                        raise RuntimeError(f"ID{motor} startup zero verification failed")
                    d._write1_locked(motor, 64, 1, "arm torque enable")
                    if d._read1_locked(motor, 64, "arm torque verify") != 1:
                        raise RuntimeError(f"ID{motor} torque enable verification failed")
                self._health_locked()
                self.active = True
            except BaseException:
                self._disable_locked()
                raise

    def _health_locked(self):
        d = self.driver
        for motor in self.restore:
            error = d._read1_locked(motor, 70, "arm hardware error")
            temperature = d._read1_locked(motor, 146, "arm temperature")
            self.temperature_c[motor - 1] = temperature
            if error or temperature >= self.config.max_temperature_c:
                raise RuntimeError(f"ID{motor} health fault: error={error}, temp={temperature}")
        self.last_health_ns = time.monotonic_ns()

    def write(self, current_ma, deadline_ns=None):
        d = self.driver
        with d._lock:
            try:
                if not self.active:
                    raise RuntimeError("arm current session is not active")
                if deadline_ns is not None and time.monotonic_ns() > deadline_ns:
                    raise RuntimeError("arm command expired while waiting for serial lock")
                currents = vector7(current_ma, "current_ma")
                raw = np.clip(
                    np.rint(currents / self.units), -self.raw_limits, self.raw_limits
                ).astype(int)
                raw *= self.config.enabled_joints
                for motor in self.restore:
                    value = int(raw[motor - 1]) & 0xFFFF
                    if not self.writer.addParam(motor, [value & 255, value >> 8]):
                        raise RuntimeError(f"SyncWrite addParam failed ID{motor}")
                transaction_start = time.monotonic_ns()
                code = self.writer.txPacket()
                self.last_transaction_ms = (time.monotonic_ns() - transaction_start) / 1e6
                if code != COMM_SUCCESS:
                    raise RuntimeError("arm SyncWrite failed")
                if time.monotonic_ns() - self.last_health_ns > 500_000_000:
                    self._health_locked()
                return raw * self.units
            except BaseException:
                self._disable_locked()
                raise
            finally:
                if self.writer is not None:
                    self.writer.clearParam()

    def _disable_locked(self):
        # Unicast on cleanup intentionally obtains per-motor acknowledgements.
        # Continue even when another motor/operation fails. Never touch ID8.
        d = self.driver
        self.active = False
        failures = []
        for motor, mode in list(self.restore.items()):
            for write, address, value, label in (
                (d._write2_locked, 102, 0, "arm shutdown zero"),
                (d._write1_locked, 64, 0, "arm shutdown disable"),
                (d._write1_locked, 11, mode, "arm restore mode"),
                (d._write2_locked, 102, 0, "arm zero after restore"),
            ):
                try:
                    write(motor, address, value, label)
                except Exception as exc:
                    failures.append(f"ID{motor}: {exc}")
        self.cleanup_errors = failures
        if not failures:
            self.restore.clear()

    def disable(self):
        with self.driver._lock:
            self._disable_locked()
        if self.cleanup_errors:
            raise RuntimeError("; ".join(self.cleanup_errors))


class TimedArmReaderMixin:
    """Opt-in replacement of upstream reader; unchanged when arm is disabled.

    A single reader still owns the existing SyncRead; all port operations share
    driver._lock. Immutable snapshots avoid acquiring the serial lock in haptics.
    """

    def _read_joint_states(self):
        if not getattr(self, "_arm_timed_reader", False):
            return super()._read_joint_states()
        sequence = 0
        previous_ns = None
        while not self._stop_thread.wait(0.001):
            start = time.monotonic_ns()
            try:
                with self._lock:
                    measurement_ns = time.monotonic_ns()
                    code = self._groupSyncRead.txRxPacket()
                    if code != COMM_SUCCESS:
                        raise RuntimeError(f"GELLO SyncRead failed: {code}")
                    positions, velocities = [], []
                    for motor in self._ids:
                        values = []
                        for address in (132, 128):
                            if not self._groupSyncRead.isAvailable(motor, address, 4):
                                raise RuntimeError(f"GELLO ID{motor} state unavailable")
                            value = self._groupSyncRead.getData(motor, address, 4)
                            values.append(value - 2**32 if value >= 2**31 else value)
                        positions.append(values[0])
                        velocities.append(values[1])
                    self._joint_angles = np.asarray(positions)
                    self._velocities = np.asarray(velocities)
                    end = time.monotonic_ns()
                    sequence += 1
                    self._arm_state_snapshot = (
                        measurement_ns,
                        self._joint_angles[:7] * np.pi / 2048,
                        self._velocities[:7] * 0.229 * 2 * np.pi / 60,
                        None,
                        (end - start) / 1e6,
                        sequence,
                        0 if previous_ns is None else (measurement_ns - previous_ns) / 1e6,
                    )
                    previous_ns = measurement_ns
            except Exception as exc:
                self._arm_read_fault = str(exc)
                self._arm_state_snapshot = (
                    time.monotonic_ns(),
                    np.zeros(7),
                    np.zeros(7),
                    str(exc),
                    0.0,
                )

    def arm_state_snapshot(self):
        fault = getattr(self, "_arm_read_fault", None)
        value = getattr(self, "_arm_state_snapshot", None)
        if value is None:
            return (0, np.zeros(7), np.zeros(7), "leader_unavailable", 0.0)
        if fault:
            return (*value[:3], fault, *value[4:])
        return value
