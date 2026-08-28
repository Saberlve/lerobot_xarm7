import socket
import struct
import time

import pytest

from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module
from lerobot_robot_ufactory.robots.uf_robot.uf_robot import (
    G2_EXTERNAL_REPORT_END,
    G2_EXTERNAL_REPORT_FORMAT,
    G2_EXTERNAL_REPORT_OFFSET,
    UFRobot,
    _decode_g2_external_device_report,
)
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig


def _make_robot(tmp_path, **config_overrides):
    config = UFRobotConfig(
        id="test_g2_current",
        calibration_dir=tmp_path,
        robot_dof=7,
        gripper_type=2,
        **config_overrides,
    )
    return UFRobot(config)


def test_decode_g2_external_device_report_uses_official_big_endian_layout():
    data = bytearray(G2_EXTERNAL_REPORT_END)
    struct.pack_into(
        G2_EXTERNAL_REPORT_FORMAT,
        data,
        G2_EXTERNAL_REPORT_OFFSET,
        2,
        2,
        47,
        -12,
        -1234,
    )

    report = _decode_g2_external_device_report(data)

    assert report is not None
    assert report.gripper_state == 2
    assert report.position_mm == 47
    assert report.speed_mm_s == -12
    assert report.current_ma == -1234


def test_decode_g2_external_device_report_rejects_short_or_wrong_device_packets():
    assert _decode_g2_external_device_report(bytes(G2_EXTERNAL_REPORT_END - 1)) is None

    data = bytearray(G2_EXTERNAL_REPORT_END)
    struct.pack_into(
        G2_EXTERNAL_REPORT_FORMAT,
        data,
        G2_EXTERNAL_REPORT_OFFSET,
        1,
        0,
        0,
        0,
        0,
    )
    assert _decode_g2_external_device_report(data) is None


class _FakeReportSocket:
    def __init__(self, packet, stop_event):
        self.chunks = [bytes(packet[:4]), bytes(packet[4:])]
        self.stop_event = stop_event
        self.closed = False

    def setsockopt(self, *args):
        pass

    def setblocking(self, value):
        pass

    def settimeout(self, value):
        pass

    def connect(self, address):
        pass

    def recv(self, size):
        if self.chunks:
            chunk = self.chunks.pop(0)
            assert len(chunk) <= size
            return chunk
        self.stop_event.set()
        return b""

    def close(self):
        self.closed = True


def test_tcp_report_reader_caches_g2_current_state_and_timestamp(monkeypatch, tmp_path):
    robot = _make_robot(tmp_path, gripper_current_monitor=True)
    robot._gripper_current_monitor_active = True
    packet = bytearray(G2_EXTERNAL_REPORT_END)
    struct.pack_into(">I", packet, 0, len(packet))
    struct.pack_into(
        G2_EXTERNAL_REPORT_FORMAT,
        packet,
        G2_EXTERNAL_REPORT_OFFSET,
        2,
        1,
        30,
        5,
        -321,
    )
    fake_socket = _FakeReportSocket(packet, robot.report_stop_event)
    monkeypatch.setattr(socket, "socket", lambda *args: fake_socket)

    robot.run()

    assert robot._gripper_current_ma == -321
    assert robot._gripper_current_state == 1
    assert robot._gripper_current_sample_monotonic_s is not None
    assert fake_socket.closed is True


def test_current_getter_distinguishes_disabled_no_sample_fresh_and_stale(tmp_path):
    disabled_robot = _make_robot(tmp_path)
    disabled = disabled_robot.get_gripper_current_sample()
    assert disabled.available is False
    assert disabled.stale is False
    assert disabled.current_ma is None
    assert disabled.reason == "monitor_disabled"

    robot = _make_robot(
        tmp_path,
        gripper_current_monitor=True,
        gripper_current_stale_timeout_s=0.1,
    )
    with robot._update_lock:
        robot._gripper_current_monitor_active = True
        cache_busy = robot.get_gripper_current_sample()

    assert cache_busy.available is False
    assert cache_busy.current_ma is None
    assert cache_busy.reason == "cache_busy"

    unavailable = robot.get_gripper_current_sample()
    assert unavailable.available is False
    assert unavailable.current_ma is None
    assert unavailable.reason == "no_sample"

    with robot._update_lock:
        robot._gripper_current_ma = 0
        robot._gripper_current_sample_monotonic_s = time.perf_counter()
        robot._gripper_current_state = 3

    fresh = robot.get_gripper_current_sample()
    assert fresh.available is True
    assert fresh.stale is False
    assert fresh.current_ma == 0
    assert fresh.gripper_state == 3

    with robot._update_lock:
        robot._gripper_current_sample_monotonic_s = time.perf_counter() - 1.0

    stale = robot.get_gripper_current_sample()
    assert stale.available is False
    assert stale.stale is True
    assert stale.current_ma == 0
    assert stale.reason == "stale"


class _MonitorArm:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def set_external_device_monitor_params(self, device_type, frequency):
        self.calls.append((device_type, frequency))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_monitor_setup_success_uses_g2_type_and_configured_frequency(
    monkeypatch,
    tmp_path,
):
    robot = _make_robot(
        tmp_path,
        gripper_current_monitor=True,
        gripper_current_monitor_frequency_hz=125,
    )
    arm = _MonitorArm(0)
    robot.real_arm = arm
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)

    robot._configure_g2_current_monitor()
    robot._configure_g2_current_monitor()

    assert arm.calls == [(2, 125)]
    assert robot._gripper_current_monitor_active is True
    assert robot._gripper_current_monitor_error is None
    assert robot._use_rt_report is True


@pytest.mark.parametrize("result", [42, RuntimeError("SDK unavailable")])
def test_monitor_setup_failure_is_nonfatal_and_getter_reports_unavailable(tmp_path, result):
    robot = _make_robot(tmp_path, gripper_current_monitor=True)
    robot.real_arm = _MonitorArm(result)

    robot._configure_g2_current_monitor()

    sample = robot.get_gripper_current_sample()
    assert sample.available is False
    assert sample.reason == "monitor_error"
    assert sample.error is not None
    assert robot._use_rt_report is False


@pytest.mark.parametrize(
    "config_overrides, message",
    [
        ({"gripper_type": 1, "gripper_current_monitor": True}, "requires gripper_type=2"),
        ({"gripper_current_monitor_frequency_hz": 0}, "positive integer"),
        ({"gripper_current_stale_timeout_s": 0.0}, "finite and positive"),
    ],
)
def test_current_monitor_config_validation(tmp_path, config_overrides, message):
    with pytest.raises(ValueError, match=message):
        UFRobotConfig(
            id="test_g2_current_validation",
            calibration_dir=tmp_path,
            robot_dof=7,
            **config_overrides,
        )
