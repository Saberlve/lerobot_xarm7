import socket
import struct
import time
from threading import Event
from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module
from lerobot_robot_ufactory.robots.uf_robot.uf_robot import (
    G2_EXTERNAL_REPORT_END,
    G2_EXTERNAL_REPORT_FORMAT,
    G2_EXTERNAL_REPORT_OFFSET,
    UFRobot,
    GripperParam,
    TimestampedStateSample,
    _decode_g2_external_device_report,
)
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig


def _make_robot(tmp_path, **config_overrides):
    config = UFRobotConfig(
        id="test_g2_current",
        calibration_dir=tmp_path,
        robot_dof=7,
        min_tcp_z_mm=10.0,
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
    # The TCP reader is still needed for arm state / the TCP height guard.
    assert robot._use_rt_report is True


def _packet(position=None):
    packet = bytearray(520 if position is None else G2_EXTERNAL_REPORT_END)
    struct.pack_into(">I", packet, 0, len(packet))
    if position is not None:
        struct.pack_into(
            G2_EXTERNAL_REPORT_FORMAT, packet, G2_EXTERNAL_REPORT_OFFSET, 2, 1, position, 5, 0
        )
    return packet


def test_position_reports_are_enabled_without_current_diagnostics(monkeypatch, tmp_path):
    robot = _make_robot(tmp_path)
    robot.real_arm = _MonitorArm(0)
    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)
    robot._configure_g2_current_monitor()
    assert robot.real_arm.calls == [(2, 250)]
    assert robot.get_gripper_current_sample().reason == "monitor_disabled"
    # Native feedback needs no polling on the command channel.
    robot._start_gripper_position_reader_if_needed()
    assert robot._gripper_position_thread is None


def test_recorded_state_tracks_measured_opening_instead_of_goal(monkeypatch, tmp_path):
    robot = _make_robot(tmp_path)
    robot._last_gripper_command = 1.0
    robot._gripper_param.gripper_norm = 1.0
    positions = [84, 63, 42, 21, 0]
    packets = [_packet(p) for p in positions]
    fake_socket = _FakeReportSocket(packets[0], robot.report_stop_event)
    fake_socket.chunks.extend(bytes(p) for p in packets[1:])
    monkeypatch.setattr(socket, "socket", lambda *args: fake_socket)
    times = iter(100_000_000_000 + i * 10_000_000 for i in range(5))
    monkeypatch.setattr(uf_robot_module, "_perf_counter_ns", lambda: next(times))
    robot.run()
    assert [s.gripper_position for s in robot._rt_state_history] == pytest.approx(
        [0, 0.25, 0.5, 0.75, 1]
    )
    robot._rt_report_normal = True
    robot._read_synchronized_cameras = lambda _: {}
    obs = robot.get_realtime_observation(target_monotonic_s=100.025)
    assert obs["gripper.pos"] == pytest.approx(0.5)
    assert robot._last_gripper_command == 1.0


def test_missing_feedback_never_falls_back_to_last_goal(monkeypatch, tmp_path):
    robot = _make_robot(tmp_path)
    robot._last_gripper_command = 0.9
    robot._gripper_param.gripper_norm = 0.9
    fake_socket = _FakeReportSocket(_packet(), robot.report_stop_event)
    monkeypatch.setattr(socket, "socket", lambda *args: fake_socket)
    monkeypatch.setattr(uf_robot_module, "_perf_counter_ns", lambda: 100_000_000_000)
    robot.run()
    assert robot._rt_state_history[-1].gripper_position is None
    robot._rt_report_normal = True
    with pytest.raises(TimeoutError, match="Actual gripper position unavailable"):
        robot.get_realtime_observation(target_monotonic_s=100)


def test_polled_position_is_causal_and_keeps_its_receipt_time(monkeypatch, tmp_path):
    robot = _make_robot(tmp_path)
    robot._last_gripper_command = 1.0
    robot._gripper_position_history.extend([(99.99, 0.5), (100.01, 0.75)])
    fake_socket = _FakeReportSocket(_packet(), robot.report_stop_event)
    monkeypatch.setattr(socket, "socket", lambda *args: fake_socket)
    monkeypatch.setattr(uf_robot_module, "_perf_counter_ns", lambda: 100_000_000_000)
    robot.run()
    selected = robot._rt_state_history[-1]
    assert selected.gripper_position == 0.5
    assert selected.gripper_sample_monotonic_s == 99.99
    robot._rt_report_normal = True
    robot._read_synchronized_cameras = lambda _: {}
    assert robot.get_realtime_observation(target_monotonic_s=100)["gripper.pos"] == 0.5


def test_stale_polled_feedback_is_rejected_even_with_fresh_arm_state(tmp_path):
    robot = _make_robot(tmp_path)
    robot._rt_report_normal = True
    robot._rt_state_history.append(
        TimestampedStateSample(
            capture_monotonic_s=100,
            joint_positions=(0.0,) * 7,
            joint_velocities=(0.0,) * 7,
            tcp_pose=(0.0,) * 6,
            tcp_velocity=(0.0,) * 6,
            gripper_position=0.5,
            gripper_sample_monotonic_s=99.8,
        )
    )
    with pytest.raises(TimeoutError, match="Actual gripper position is 200.000 ms old"):
        robot.get_realtime_observation(target_monotonic_s=100)


@pytest.mark.parametrize("position, normalized", [(84, 0), (42, 0.5), (0, 1), (85, 0), (-1, 1)])
def test_actual_position_normalization(position, normalized):
    params = GripperParam("G2", open_pos=84, close_pos=0)
    assert params.get_gripper_norm(position) == normalized


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_invalid_position_cannot_reuse_cached_state(value):
    params = GripperParam("G2", open_pos=84, close_pos=0, gripper_norm=0.8)
    with pytest.raises(ValueError, match="Actual gripper position"):
        params.get_gripper_norm(value)


@pytest.mark.parametrize(
    "feedback, succeeds", [((0, 42), True), ((1, 42), False), ((0, None), False)]
)
def test_background_poll_records_only_valid_measured_positions(tmp_path, feedback, succeeds):
    robot = _make_robot(tmp_path)
    robot._last_gripper_command = 1.0

    def read():
        robot._gripper_position_stop_event.set()
        return feedback

    robot.real_arm = SimpleNamespace(get_gripper_g2_position=read)
    robot._gripper_position_read_worker()
    if succeeds:
        assert robot._gripper_position_history[-1][1] == 0.5
    else:
        assert not robot._gripper_position_history


@pytest.mark.parametrize("native_reports_available", [False, True])
def test_background_reader_stops_on_disconnect_or_native_feedback(
    tmp_path, native_reports_available
):
    robot = _make_robot(tmp_path)
    read_seen = Event()

    def read():
        read_seen.set()
        return 0, 42

    robot.real_arm = SimpleNamespace(get_gripper_g2_position=read, connected=False)
    robot._start_gripper_position_reader_if_needed()
    thread = robot._gripper_position_thread
    try:
        assert read_seen.wait(1)
        if native_reports_available:
            robot._gripper_current_monitor_active = True
            robot._start_gripper_position_reader_if_needed()
            assert robot._gripper_position_history[-1][1] == 0.5
        else:
            robot.disconnect()
        assert not thread.is_alive()
        assert robot._gripper_position_thread is None
    finally:
        robot.disconnect()


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
