"""An explicit SDK connection. No upstream initialization, fake fallback or EEPROM writes.

All methods are called by one owner thread. Diagnostics use only PING and READ.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


class XL330Transport:
    def __init__(self, profile):
        self.profile = profile
        self.port = None
        self.packet = None
        self.touched = []
        self.original = {}
        self.info = []
        self.gripper_limit = None
        device = Path(profile.port).resolve().name
        self.latency_timer_path = Path("/sys/bus/usb-serial/devices") / device / "latency_timer"

    def check_usb_latency(self):
        """Require the host timing used for commissioning, before any motor write."""
        path = self.latency_timer_path
        if not path.is_file():
            raise RuntimeError(f"Cannot verify GELLO USB latency timer: {path}")
        latency_ms = int(path.read_text().strip())
        if latency_ms != 1:
            raise RuntimeError(
                f"GELLO USB latency is {latency_ms} ms; current control requires 1 ms. "
                f"Run: echo 1 | sudo tee {path}"
            )

    def check(self, result, error, label):
        if result != 0 or error:
            raise RuntimeError(f"{label}: communication={result}, device_error={error}")

    def read(self, dxl_id, address, size):
        value, result, error = getattr(self.packet, f"read{size}ByteTxRx")(
            self.port, dxl_id, address
        )
        self.check(result, error, f"read ID{dxl_id} address {address}")
        return value

    def write(self, dxl_id, address, size, value):
        result, error = getattr(self.packet, f"write{size}ByteTxRx")(
            self.port, dxl_id, address, int(value) & ((1 << (8 * size)) - 1)
        )
        self.check(result, error, f"write ID{dxl_id} address {address}")

    def open(self):
        import fcntl
        import termios

        from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler

        if not os.path.exists(self.profile.port):
            raise FileNotFoundError(f"GELLO is not connected: {self.profile.port}")
        busy = subprocess.run(["fuser", self.profile.port], capture_output=True, timeout=3)
        if busy.returncode == 0:
            raise RuntimeError("Serial port is already in use; no process was terminated")
        if busy.returncode not in (0, 1):
            raise RuntimeError("Cannot check serial port ownership")
        self.port = PortHandler(self.profile.port)
        self.packet = PacketHandler(2.0)
        try:
            if not self.port.openPort() or not self.port.setBaudRate(self.profile.baudrate):
                raise RuntimeError("Cannot open serial port at configured baudrate")
            self.port.ser.exclusive = True
            self.port.ser.write_timeout = 0.1
            fcntl.ioctl(self.port.ser.fileno(), termios.TIOCEXCL)
            # Current, velocity, position, voltage and temperature in one transaction.
            self.reader = GroupSyncRead(self.port, self.packet, 126, 21)
            self.writer = GroupSyncWrite(self.port, self.packet, 102, 2)
            for dxl_id in self.profile.all_ids:
                if not self.reader.addParam(dxl_id):
                    raise RuntimeError(f"Cannot add ID{dxl_id} to state reader")
            self.info = self.probe()
        except BaseException:
            self.close()
            raise

    def probe(self):
        records = []
        for dxl_id, expected in zip(self.profile.all_ids, self.profile.model_numbers, strict=True):
            model, result, error = self.packet.ping(self.port, dxl_id)
            self.check(result, error, f"ping ID{dxl_id}")
            if model != expected:
                raise RuntimeError(f"ID{dxl_id}: expected model {expected}, found {model}")
            records.append(
                {
                    "id": dxl_id,
                    "model": model,
                    "firmware": self.read(dxl_id, 6, 1),
                    "baud_raw": self.read(dxl_id, 8, 1),
                    "mode": self.read(dxl_id, 11, 1),
                    "current_limit_raw": self.read(dxl_id, 38, 2),
                    "torque_enabled": self.read(dxl_id, 64, 1),
                    "hardware_error": self.read(dxl_id, 70, 1),
                    "watchdog": self.read(dxl_id, 98, 1),
                }
            )
        return records

    def enable(self, *, experimental=False):
        self.profile.validate_live(experimental=experimental)
        self.check_usb_latency()
        for i, item in enumerate(self.info):
            if item["torque_enabled"] or item["hardware_error"] or item["watchdog"] == 255:
                raise RuntimeError(
                    f"ID{item['id']} is active or faulted; refusing automatic recovery"
                )
            if i < 7 and not 0 < self.profile.limits[i] * 1000 <= item["current_limit_raw"] <= 1750:
                raise ValueError(f"ID{item['id']} configured current exceeds hardware limit")
        # Configure every joint at zero before enabling any joint.
        for item in self.info[:7]:
            dxl_id = item["id"]
            self.original[dxl_id] = (item["mode"], item["watchdog"])
            self.touched.append(dxl_id)
            self.write(dxl_id, 64, 1, 0)
            self.write(dxl_id, 11, 1, 0)
            self.write(dxl_id, 102, 2, 0)
            self.write(dxl_id, 98, 1, self.profile.watchdog_ms // 20)
            if self.read(dxl_id, 11, 1) != 0 or self.read(dxl_id, 102, 2) != 0:
                raise RuntimeError(f"ID{dxl_id} zero-current mode verification failed")
        for dxl_id in self.touched:
            self.write(dxl_id, 64, 1, 1)
            if self.read(dxl_id, 64, 1) != 1:
                raise RuntimeError(f"ID{dxl_id} torque enable verification failed")

    def state(self):
        start = time.monotonic()
        sample_start_ns = time.perf_counter_ns()
        # SDK GroupSyncRead.rxPacket discards each status packet's device error.
        # Keep its broadcast request, but validate every reply ourselves.
        self.check(self.reader.txPacket(), 0, "sync read request")
        payloads = {}
        for dxl_id in self.profile.all_ids:
            data, result, error = self.packet.readRx(self.port, dxl_id, 21)
            self.check(result, error, f"sync read reply ID{dxl_id}")
            if len(data) != 21:
                raise RuntimeError(f"Incomplete state for ID{dxl_id}")
            payloads[dxl_id] = bytes(data)
        sample_end_ns = time.perf_counter_ns()
        q, dq, current, temperature, voltage = [], [], [], [], []
        for dxl_id in self.profile.all_ids:

            def value(address, size, payload=payloads[dxl_id]):
                return int.from_bytes(payload[address - 126 : address - 126 + size], "little")

            current.append(signed(value(126, 2), 16) * 0.001)
            dq.append(signed(value(128, 4), 32) * 0.229 * 2 * np.pi / 60)
            q.append(signed(value(132, 4), 32) * np.pi / 2048)
            voltage.append(value(144, 2) * 0.1)
            temperature.append(value(146, 1))
        return {
            "stamp": start,
            "sample_start_ns": sample_start_ns,
            "sample_end_ns": sample_end_ns,
            "position": np.array(q),
            "velocity": np.array(dq),
            "current_a": current,
            "temperature_c": temperature,
            "voltage_v": voltage,
        }

    def health(self):
        for dxl_id in self.touched:
            hardware_error = self.read(dxl_id, 70, 1)
            watchdog = self.read(dxl_id, 98, 1)
            if hardware_error or watchdog == 255:
                raise RuntimeError(
                    f"ID{dxl_id} hardware/watchdog fault: "
                    f"hardware_error={hardware_error}, watchdog={signed(watchdog, 8)}"
                )
            if self.read(dxl_id, 64, 1) != 1:
                raise RuntimeError(f"ID{dxl_id} unexpectedly lost torque")

    def currents(self, amperes, gripper_a=None):
        values = np.asarray(amperes)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Seven finite arm currents required")
        values = np.clip(values, -self.profile.limits, self.profile.limits)
        try:
            for dxl_id, value in zip(self.profile.ids, values, strict=True):
                # Truncate toward zero so quantization cannot exceed configured limits.
                raw = int(value * 1000) & 0xFFFF
                if not self.writer.addParam(dxl_id, [raw & 255, raw >> 8]):
                    raise RuntimeError(f"Cannot encode current for ID{dxl_id}")
            if self.gripper_limit is not None:
                raw = (
                    int(np.clip(gripper_a or 0.0, -self.gripper_limit, self.gripper_limit) * 1000)
                    & 0xFFFF
                )
                if not self.writer.addParam(8, [raw & 255, raw >> 8]):
                    raise RuntimeError("Cannot encode ID8 current")
            self.check(self.writer.txPacket(), 0, "sync current write")
        finally:
            self.writer.clearParam()

    def gripper_info(self):
        if self.profile.gripper_id != 8 or self.profile.model_numbers[-1] != 1190:
            raise RuntimeError("Existing gripper feedback requires XL330-M077 ID8")
        return {
            "dxl_id": 8,
            "model_number": 1190,
            "model_name": "XL330-M077-T",
            "operating_mode": self.read(8, 11, 1),
            "current_limit_raw": self.read(8, 38, 2),
            "current_limit_ma": float(self.read(8, 38, 2)),
            "current_unit_ma": 1.0,
        }

    def enable_gripper(self, limit_ma):
        if isinstance(limit_ma, bool) or not np.isfinite(limit_ma) or not 0 < limit_ma <= 100:
            raise ValueError("ID8 limit must be finite, positive and at most 100 mA")
        if self.gripper_limit is not None:
            raise RuntimeError("ID8 current mode is already active")
        self.check_usb_latency()
        info = self.gripper_info()
        if limit_ma > info["current_limit_ma"] or self.read(8, 64, 1):
            raise RuntimeError("ID8 is active or configured limit exceeds hardware")
        self.original[8] = (info["operating_mode"], self.read(8, 98, 1))
        self.touched.append(8)
        self.write(8, 64, 1, 0)
        self.write(8, 11, 1, 0)
        self.write(8, 102, 2, 0)
        self.write(8, 98, 1, self.profile.watchdog_ms // 20)
        if self.read(8, 11, 1) != 0 or self.read(8, 102, 2) != 0:
            raise RuntimeError("ID8 mode verification failed")
        self.write(8, 64, 1, 1)
        if self.read(8, 64, 1) != 1:
            raise RuntimeError("ID8 torque verification failed")
        self.gripper_limit = limit_ma / 1000
        return info

    def disable_gripper(self):
        if 8 not in self.touched:
            return
        self.write(8, 64, 1, 0)
        if self.read(8, 64, 1) != 0:
            raise RuntimeError("ID8 torque-off not verified")
        mode, watchdog = self.original[8]
        self.write(8, 98, 1, 0)
        self.write(8, 11, 1, mode)
        self.write(8, 102, 2, 0)
        self.write(8, 98, 1, watchdog)
        self.touched.remove(8)
        self.gripper_limit = None

    def disable(self):
        failures = []
        # Stop ALL motors first. A latched watchdog makes Goal Current read-only,
        # so trying to zero it before torque-off generates Access Error(7).
        for dxl_id in self.touched:
            try:
                self.write(dxl_id, 64, 1, 0)
            except Exception as exc:
                failures.append(str(exc))
        for dxl_id in self.touched:
            try:
                if self.read(dxl_id, 64, 1) != 0:
                    raise RuntimeError(f"ID{dxl_id} torque-off not verified")
                mode, watchdog = self.original[dxl_id]
                self.write(dxl_id, 98, 1, 0)
                self.write(dxl_id, 11, 1, mode)
                self.write(dxl_id, 102, 2, 0)
                if self.read(dxl_id, 102, 2) != 0:
                    raise RuntimeError(f"ID{dxl_id} zero-current cleanup not verified")
                self.write(dxl_id, 98, 1, watchdog)
            except Exception as exc:
                failures.append(str(exc))
        self.touched.clear()
        if failures:
            raise RuntimeError("Cleanup incomplete: " + "; ".join(failures))

    def close(self):
        if self.port is not None:
            self.port.closePort()
            self.port = None
