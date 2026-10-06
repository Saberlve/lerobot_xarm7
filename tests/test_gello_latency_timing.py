import runpy
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dynamixel_sdk import COMM_SUCCESS
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_adapter import SafeDynamixelDriver, driver_module
from lerobot_robot_ufactory.gravity_compensation.control.runtime import GravityRuntime, RuntimeRobot
from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController

ROOT = Path(__file__).resolve().parents[1]


def test_passive_sample_publishes_matching_angles_sequence_and_timestamps():
    driver = SafeDynamixelDriver.__new__(SafeDynamixelDriver)
    driver._is_fake = False
    driver._ids = [1]
    driver._lock = threading.Lock()
    driver._stop_thread = threading.Event()
    driver._timed_joint_sample = None
    driver._sample_sequence = 0

    class Reader:
        def txRxPacket(self):
            driver._stop_thread.set()
            return COMM_SUCCESS

        def isAvailable(self, *args):
            return True

        def getData(self, motor, address, size):
            return 1024 if address == driver_module.ADDR_PRESENT_POSITION else 0

    driver._groupSyncRead = Reader()
    driver._read_joint_states()
    before = time.perf_counter_ns()
    np.testing.assert_allclose(driver.get_joints(), [np.pi / 2])
    consumed = driver.last_joint_sample_timing.copy()
    assert consumed["sample_sequence"] == 1
    assert consumed["sample_start_ns"] <= consumed["sample_end_ns"] <= before
    driver._timed_joint_sample = (np.array([2048]), {"sample_sequence": 2})
    assert driver.last_joint_sample_timing == consumed


def test_failed_passive_read_does_not_refresh_cached_timestamp():
    driver = SafeDynamixelDriver.__new__(SafeDynamixelDriver)
    driver._ids = [1]
    driver._lock = threading.Lock()
    driver._stop_thread = threading.Event()
    original = (np.array([512]), {"sample_start_ns": 1, "sample_end_ns": 2, "sample_sequence": 3})
    driver._timed_joint_sample = original
    driver._sample_sequence = 3

    def failed_read():
        driver._stop_thread.set()
        return -1

    driver._groupSyncRead = SimpleNamespace(txRxPacket=failed_read)
    driver._read_joint_states()
    assert driver._timed_joint_sample is original
    assert driver._sample_sequence == 3


def test_compensated_adapter_retains_timing_of_consumed_sample():
    runtime = GravityRuntime.__new__(GravityRuntime)
    state = {"position": np.zeros(8), "sample_start_ns": 100, "sample_end_ns": 200, "sample_sequence": 5}
    runtime.state = lambda: state.copy()
    robot = RuntimeRobot(runtime, [1] * 7, [8, 0, 42])
    robot.get_joint_state()
    state.update(sample_start_ns=300, sample_end_ns=400, sample_sequence=6)
    assert robot.last_joint_sample_timing == {"sample_start_ns": 100, "sample_end_ns": 200, "sample_sequence": 5}


def test_realtime_timing_preserves_source_sample_through_send():
    class Teleop:
        def get_action(self):
            end = time.perf_counter_ns() - 10_000_000
            self.timing = {"sample_start_ns": end-2_000_000, "sample_end_ns": end, "sample_sequence": 7}
            return {"J1.pos": 0.0}

        def get_action_sample_timing(self):
            return self.timing.copy()

    teleop = Teleop()
    robot = SimpleNamespace(send_action=lambda action: action)
    controller = RealtimeTeleopController(robot, teleop, lambda pair: pair[0], lambda pair: pair[0], 30, {}, record_timing=True)
    try:
        controller.start()
    finally:
        controller.stop()
    rows = controller.action_timings()
    assert rows
    for row in rows:
        assert row["sample_sequence"] == 7
        assert row["sample_end_ns"] - row["sample_start_ns"] == 2_000_000
        assert row["action_send_start_ns"] - row["sample_end_ns"] >= 10_000_000


def test_analysis_measures_sample_age_and_excludes_warmup():
    measure = runpy.run_path(str(ROOT / "scripts/compare_gello_latency.py"))["measure"]
    rows = []
    for i in range(5):
        send = 3_000_000_000 + i * 1_000_000_000
        rows.append({"sample_start_ns": send-12_000_000, "sample_end_ns": send-10_000_000,
                     "sample_sequence": i // 2, "gello_read_start_ns": send-500_000,
                     "gello_read_end_ns": send-100_000, "action_send_start_ns": send,
                     "action_send_end_ns": send+1_000_000})
    result = measure([rows], fps=1)
    assert result["warmup_actions_excluded"] == 2
    assert result["used_actions"] == 3
    assert result["metrics"]["read_begin_to_send_ms"]["median"] == 12
    assert result["metrics"]["read_end_to_send_ms"]["median"] == 10
    assert result["repeated_sample_pct"] == 50
    assert result["send_gap_over_1_5_period_count"] == 0
    rows[3].pop("sample_start_ns")
    with pytest.raises(ValueError, match="Missing valid"):
        measure([rows], fps=1)
