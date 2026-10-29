"""Serial recovery during capture and fresh-input checks before motion."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from serial import SerialException

from lerobot_robot_ufactory.scripts.uf_lerobot_record import _prepare_recording_episode
from lerobot_robot_ufactory.teleoperators.gello_teleop import gello_adapter as adapter
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig
from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController


def make_driver():
    driver = adapter.SafeDynamixelDriver.__new__(adapter.SafeDynamixelDriver)
    driver._is_fake = False
    driver._ids = [1, 2]
    driver._lock = threading.Lock()
    driver._sample_condition = threading.Condition()
    driver._sample_sequence = 1
    driver._sample_stamp = time.monotonic()
    driver._read_error = None
    driver._reopen_serial_port = lambda: None
    driver._joint_angles = np.array([1024, 2048])
    driver._velocities = np.array([0, -1])
    driver._stop_thread = threading.Event()
    driver._reading_thread = threading.current_thread()
    return driver


@pytest.mark.parametrize("fault", ["exception", "dead_reader", "stopped"])
@pytest.mark.parametrize("method", ["get_joints", "get_positions_and_velocities"])
def test_cached_encoder_sample_is_rejected_after_failure(fault, method):
    driver = make_driver()
    if fault == "exception":
        driver._read_error = SerialException("USB disconnected")
    elif fault == "dead_reader":
        driver._reading_thread = threading.Thread()
    elif fault == "stopped":
        driver._stop_thread.set()
    else:
        driver._sample_stamp -= 2 * adapter.GELLO_STATE_TIMEOUT_S
    with pytest.raises(RuntimeError, match="GELLO unavailable"):
        getattr(driver, method)()


def test_start_requires_new_successful_read_even_if_cached_sample_is_recent(monkeypatch):
    monkeypatch.setattr(adapter, "GELLO_STATE_TIMEOUT_S", 0.03)
    driver = make_driver()
    # Reader looks alive, but has stopped publishing successful transactions.
    with pytest.raises(RuntimeError, match="no fresh encoder sample|data is stale"):
        driver.check_health(require_fresh=True)


def test_initial_sample_wait_is_bounded(monkeypatch):
    monkeypatch.setattr(adapter, "GELLO_STATE_TIMEOUT_S", 0.03)
    driver = make_driver()
    driver._sample_stamp = None
    driver._joint_angles = None
    driver._velocities = None
    with pytest.raises(RuntimeError, match="no fresh encoder sample"):
        driver.get_joints()


def test_fake_fallback_cannot_pass_physical_gello_preflight():
    driver = make_driver()
    driver._is_fake = True
    with pytest.raises(RuntimeError, match="simulated encoder"):
        driver.check_health(require_fresh=True)


class Reader:
    def __init__(self, *, error=None, result=0):
        self.error = error
        self.result = result

    def txRxPacket(self):
        if self.error is not None:
            raise self.error
        return self.result

    def isAvailable(self, *args):
        return True

    def getData(self, dxl_id, address, length):
        if address == adapter.driver_module.ADDR_PRESENT_POSITION:
            return dxl_id * 1024
        return 0xFFFFFFFF if dxl_id == 2 else 0


def test_stationary_gello_still_publishes_fresh_samples():
    driver = make_driver()
    driver._groupSyncRead = Reader()
    driver._start_reading_thread()
    try:
        # Freshness follows successful reads, not whether the operator moves.
        for _ in range(3):
            driver.check_health(require_fresh=True)
            assert driver.get_joints() == pytest.approx([np.pi / 2, np.pi])
        assert driver.get_positions_and_velocities()[1][1] < 0
    finally:
        driver._stop_thread.set()
        driver._reading_thread.join(1)
    assert not driver._reading_thread.is_alive()


def test_serial_exception_keeps_retrying_with_cached_target(monkeypatch):
    monkeypatch.setattr(adapter, "GELLO_STATE_TIMEOUT_S", 0.01)
    driver = make_driver()
    driver._groupSyncRead = Reader(error=SerialException("USB disconnected"))
    stamp = driver._sample_stamp
    driver._start_reading_thread()
    try:
        driver._reading_thread.join(0.05)
        assert driver._reading_thread.is_alive()
        assert driver._read_error is None
        assert driver._sample_stamp == stamp
        assert driver.get_joints() == pytest.approx([np.pi / 2, np.pi])
        with pytest.raises(RuntimeError, match="no fresh encoder sample"):
            driver.check_health(require_fresh=True)
    finally:
        driver._stop_thread.set()
        driver._reading_thread.join(1)


def test_persistent_packet_failure_expires_cached_data(monkeypatch):
    monkeypatch.setattr(adapter, "GELLO_STATE_TIMEOUT_S", 0.03)
    driver = make_driver()
    driver._groupSyncRead = Reader(result=-3001)
    driver._start_reading_thread()
    try:
        with pytest.raises(RuntimeError, match="data is stale|no fresh encoder sample"):
            driver.check_health(require_fresh=True)
        assert driver._sample_sequence == 1
    finally:
        driver._stop_thread.set()
        driver._reading_thread.join(1)


@pytest.mark.parametrize("reset_robot", [False, True])
def test_unplugged_gello_blocks_start_before_any_robot_motion(reset_robot):
    driver = make_driver()
    driver._read_error = SerialException("USB disconnected")
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop._is_connected = True
    teleop.gello_agent = SimpleNamespace(_robot=SimpleNamespace(_driver=driver))
    robot = SimpleNamespace(
        reset_to_initial=lambda **kwargs: pytest.fail("must not reset the arm"),
        get_observation=lambda: pytest.fail("must not prepare capture"),
    )
    with pytest.raises(RuntimeError, match="USB disconnected"):
        _prepare_recording_episode(robot, teleop, True, False, reset_robot=reset_robot)
    assert not teleop._teleop_enabled


def test_enable_checks_health_even_when_alignment_was_previously_complete():
    driver = make_driver()
    driver._read_error = SerialException("USB disconnected during reset")
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop._is_connected = True
    teleop._needs_alignment = False
    teleop.gello_agent = SimpleNamespace(_robot=SimpleNamespace(_driver=driver))
    with pytest.raises(RuntimeError, match="USB disconnected during reset"):
        teleop.set_teleop_enabled(True, {})
    assert not teleop._teleop_enabled


def test_disabling_faulted_gello_clears_output_without_retrying_dead_serial():
    driver = make_driver()
    driver._read_error = SerialException("USB disconnected")
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop._is_connected = True
    teleop._teleop_enabled = True
    teleop.gello_agent = SimpleNamespace(_robot=SimpleNamespace(
        _driver=driver, gripper_open_close=None,
        set_torque_mode=lambda enabled: pytest.fail("must not reuse dead serial"),
    ))
    teleop.set_teleop_enabled(False)
    assert not teleop._teleop_enabled
    assert teleop._needs_alignment


def test_encoder_failure_pauses_arm_and_does_not_send_cached_action():
    driver = make_driver()
    driver._read_error = SerialException("USB disconnected")
    paused = threading.Event()
    robot = SimpleNamespace(
        send_action=lambda action: pytest.fail("must not send stale target"),
        pause_motion=paused.set,
    )
    teleop = SimpleNamespace(get_action=lambda: driver.get_joints())
    controller = RealtimeTeleopController(
        robot, teleop, lambda value: value[0], lambda value: value[0],
        fps=30, initial_observation={},
    )
    try:
        with pytest.raises(RuntimeError, match="Realtime joint control thread failed"):
            controller.start()
    finally:
        controller.stop(raise_on_fault=False)
    assert paused.is_set()
    assert controller._latest_action is None


def test_disconnect_after_control_starts_pauses_and_stops_further_commands():
    driver = make_driver()
    disconnect = threading.Event()
    paused = threading.Event()
    actions = []
    reads = 0

    def get_action():
        nonlocal reads
        reads += 1
        if reads > 1:
            assert disconnect.wait(1)
        return {"J1.pos": float(driver.get_joints()[0])}

    robot = SimpleNamespace(send_action=lambda action: actions.append(action),
                            pause_motion=paused.set)
    controller = RealtimeTeleopController(
        robot, SimpleNamespace(get_action=get_action),
        lambda value: value[0], lambda value: value[0],
        fps=30, initial_observation={},
    )
    try:
        controller.start()
        driver._read_error = SerialException("USB disconnected during capture")
        disconnect.set()
        assert paused.wait(1)
        with pytest.raises(RuntimeError, match="USB disconnected during capture"):
            controller.raise_if_failed()
    finally:
        disconnect.set()
        controller.stop(raise_on_fault=False)
    assert len(actions) == 1


def test_single_failed_packet_can_recover_before_freshness_deadline():
    driver = make_driver()

    class RecoveringReader(Reader):
        calls = 0

        def txRxPacket(self):
            self.calls += 1
            return -3001 if self.calls == 1 else 0

    driver._groupSyncRead = RecoveringReader()
    driver._start_reading_thread()
    try:
        driver.check_health(require_fresh=True)
        assert driver.get_joints() == pytest.approx([np.pi / 2, np.pi])
        assert driver._read_error is None
    finally:
        driver._stop_thread.set()
        driver._reading_thread.join(1)


@pytest.mark.parametrize("initial_sample", [True, False])
def test_transient_serial_exception_recovers(initial_sample, monkeypatch):
    monkeypatch.setattr(adapter, "GELLO_RECONNECT_INTERVAL_S", 0.01)
    driver = make_driver()
    if not initial_sample:
        driver._sample_stamp = None

    class RecoveringReader(Reader):
        calls = 0

        def txRxPacket(self):
            self.calls += 1
            if self.calls <= 2:
                raise SerialException("device reports readiness but returned no data")
            return 0

    driver._groupSyncRead = RecoveringReader()
    driver._start_reading_thread()
    try:
        driver.check_health(require_fresh=True)
        assert driver._groupSyncRead.calls >= 3
        assert driver._read_error is None
        assert driver.get_joints() == pytest.approx([np.pi / 2, np.pi])
    finally:
        driver._stop_thread.set()
        driver._reading_thread.join(1)


def test_capture_accepts_old_complete_sample_but_start_requires_fresh():
    driver = make_driver()
    driver._sample_stamp -= 10
    assert driver.get_joints() == pytest.approx([np.pi / 2, np.pi])
    assert driver.get_positions_and_velocities()[0] == pytest.approx([np.pi / 2, np.pi])
    with pytest.raises(RuntimeError, match="no fresh encoder sample"):
        driver.check_health(require_fresh=True)


def test_next_episode_waits_for_new_sample_when_previous_cache_is_old():
    driver = make_driver()
    driver._sample_stamp -= 10
    waiting = threading.Event()
    original_wait = driver._sample_condition.wait

    def signal_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    driver._sample_condition.wait = signal_wait

    def publish():
        if not waiting.wait(1):
            return
        with driver._sample_condition:
            driver._sample_stamp = time.monotonic()
            driver._sample_sequence += 1
            driver._sample_condition.notify_all()

    publisher = threading.Thread(target=publish)
    publisher.start()
    try:
        driver.check_health(require_fresh=True)
        assert waiting.is_set()
        assert driver._sample_sequence == 2
    finally:
        waiting.set()
        publisher.join(1)


def test_reopen_uses_same_handler_and_restores_baudrate(monkeypatch):
    driver = make_driver()
    calls = []
    driver._port = "/dev/serial/by-id/test"
    driver._baudrate = 1000000
    driver._portHandler = SimpleNamespace(
        ser=object(), is_using=True,
        closePort=lambda: calls.append("close"),
        setBaudRate=lambda baud: calls.append(baud) or True,
    )
    monkeypatch.setattr(adapter.os.path, "exists", lambda path: True)
    monkeypatch.setattr(adapter.os, "access", lambda path, mode: True)
    adapter.SafeDynamixelDriver._reopen_serial_port(driver)
    assert calls == ["close", 1000000]
    assert not driver._portHandler.is_using


def test_missing_port_does_not_attempt_to_kill_processes(monkeypatch):
    driver = make_driver()
    driver._port = "/dev/serial/by-id/missing"
    driver._max_retries = 1
    driver._initialize_hardware = driver._prepare_port
    driver._kill_processes_using_port = lambda: pytest.fail("must not kill processes")
    driver._fix_port_permissions = lambda: pytest.fail("must not change permissions")
    monkeypatch.setattr(adapter.os.path, "exists", lambda path: False)
    assert driver._initialize_with_retries() is False


def test_reader_recovers_after_device_disappears_and_returns(monkeypatch):
    monkeypatch.setattr(adapter, "GELLO_RECONNECT_INTERVAL_S", 0.01)
    driver = make_driver()
    recovered = threading.Event()
    attempts = []
    reader = Reader(error=SerialException("disconnected"))
    driver._groupSyncRead = reader

    def reopen():
        attempts.append(1)
        if len(attempts) < 3:
            raise FileNotFoundError("USB absent")
        reader.error = None
        recovered.set()

    driver._reopen_serial_port = reopen
    driver._start_reading_thread()
    try:
        assert recovered.wait(1)
        driver.check_health(require_fresh=True)
        assert len(attempts) == 3
        assert driver._read_error is None
        assert driver._reading_thread.is_alive()
    finally:
        driver._stop_thread.set()
        driver._reading_thread.join(1)
