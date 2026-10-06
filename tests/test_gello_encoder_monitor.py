"""Read-only lifecycle, calibration mapping, failure reporting and serial release."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_ufactory.gravity_compensation.config import DeviceProfile
from lerobot_robot_ufactory.gravity_compensation.monitoring.encoder_monitor import EncoderMonitor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile():
    p = DeviceProfile(ROOT / "config/gravity/gello_A_working.yaml")
    # Exercise signs independently of the current all-positive real calibration.
    p.signs = np.array([1, -1, 1, -1, 1, -1, 1])
    return p


class ReadOnlyTransport:
    def __init__(self, profile):
        self.profile = profile
        self.info = [{"id": i, "torque_enabled": 0} for i in profile.all_ids]
        self.calls = []
        self.threads = set()
        self.fail = False

    def open(self):
        self.calls.append("open")
        self.threads.add(threading.get_ident())

    def state(self):
        self.calls.append("read")
        self.threads.add(threading.get_ident())
        if self.fail:
            raise RuntimeError("injected USB disconnect")
        return {
            "stamp": time.monotonic(),
            "position": np.arange(8) * 0.2,
            "velocity": np.ones(8),
            "temperature_c": [25] * 8,
            "voltage_v": [5] * 8,
        }

    def close(self):
        self.calls.append("close")
        self.threads.add(threading.get_ident())

    def write(self, *args):
        pytest.fail("Viewer must never write motor registers")

    enable = disable = currents = write


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Monitor did not reach expected state")


def test_calibrated_stream_has_one_read_only_owner_and_stops(profile):
    transport = ReadOnlyTransport(profile)
    monitor = EncoderMonitor(profile, rate_hz=100, transport_factory=lambda _: transport)
    try:
        monitor.start()
        wait_for(lambda: monitor.snapshot()["sequence"] >= 2)
        packet = monitor.snapshot()
        assert packet["status"] == "connected"
        assert packet["age_ms"] < 500
        np.testing.assert_allclose(
            packet["sample"]["model_q_rad"],
            (np.arange(7) * 0.2 - profile.zeros) * profile.signs,
        )
        assert packet["sample"]["encoder_rad"][-1] == pytest.approx(1.4)
        assert packet["sample"]["read_hz"] > 0
        # Slow or multiple browsers never trigger additional serial reads.
        for _ in range(100):
            monitor.snapshot()
        assert transport.threads != {threading.get_ident()}
    finally:
        monitor.stop()
    assert len(transport.threads) == 1
    assert transport.calls[0] == "open"
    assert transport.calls[-1] == "close"
    assert set(transport.calls) == {"open", "read", "close"}


def test_disconnect_retains_labelled_last_sample_and_closes(profile):
    transport = ReadOnlyTransport(profile)
    monitor = EncoderMonitor(profile, transport_factory=lambda _: transport)
    try:
        monitor.start()
        wait_for(lambda: monitor.snapshot()["sequence"] > 0)
        previous = monitor.snapshot()["sample"]
        transport.fail = True
        wait_for(lambda: monitor.snapshot()["status"] == "error")
        packet = monitor.snapshot()
        assert packet["sample"]["model_q_rad"] == previous["model_q_rad"]
        assert "USB disconnect" in packet["error"]
        wait_for(lambda: "close" in transport.calls)
    finally:
        monitor.stop()


def test_missing_device_never_presents_a_pose(profile):
    transport = ReadOnlyTransport(profile)

    def missing():
        raise FileNotFoundError("not connected")

    transport.open = missing
    monitor = EncoderMonitor(profile, transport_factory=lambda _: transport)
    try:
        monitor.start()
        wait_for(lambda: monitor.snapshot()["status"] == "error")
        assert monitor.snapshot()["sample"] is None
        assert "not connected" in monitor.snapshot()["error"]
    finally:
        monitor.stop()
    assert transport.calls == ["close"]


def test_stale_age_and_invalid_read_rate(profile):
    monitor = EncoderMonitor(profile)
    monitor._sample = {"stamp": time.monotonic() - 2}
    monitor._status = "connected"
    assert monitor.snapshot()["status"] == "stale"
    for rate in (0, float("nan"), float("inf"), 201):
        with pytest.raises(ValueError, match="Read rate"):
            EncoderMonitor(profile, rate_hz=rate)

