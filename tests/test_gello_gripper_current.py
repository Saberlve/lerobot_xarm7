import threading

import numpy as np
import pytest
from dynamixel_sdk import COMM_SUCCESS

from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_adapter import (
    ADDR_CURRENT_LIMIT,
    ADDR_GOAL_CURRENT,
    ADDR_OPERATING_MODE,
    CURRENT_CONTROL_MODE,
    GRIPPER_DYNAMIXEL_ID,
    POSITION_CONTROL_MODE,
    ContinuousDynamixelRobot,
    SafeDynamixelDriver,
)


class FakePacketHandler:
    def __init__(self, *, model_number=1200):
        self.model_number = model_number
        self.registers = {
            (GRIPPER_DYNAMIXEL_ID, ADDR_OPERATING_MODE): POSITION_CONTROL_MODE,
            (GRIPPER_DYNAMIXEL_ID, ADDR_CURRENT_LIMIT): 1750,
            (GRIPPER_DYNAMIXEL_ID, 64): 0,
            (GRIPPER_DYNAMIXEL_ID, ADDR_GOAL_CURRENT): 0,
        }
        self.calls = []
        self.fail_nonzero_goal_current = False

    def ping(self, port, dxl_id):
        self.calls.append(("ping", dxl_id))
        return self.model_number, COMM_SUCCESS, 0

    def read1ByteTxRx(self, port, dxl_id, address):
        self.calls.append(("read1", dxl_id, address))
        return self.registers[(dxl_id, address)], COMM_SUCCESS, 0

    def read2ByteTxRx(self, port, dxl_id, address):
        self.calls.append(("read2", dxl_id, address))
        return self.registers[(dxl_id, address)], COMM_SUCCESS, 0

    def write1ByteTxRx(self, port, dxl_id, address, value):
        self.calls.append(("write1", dxl_id, address, value))
        self.registers[(dxl_id, address)] = value
        return COMM_SUCCESS, 0

    def write2ByteTxRx(self, port, dxl_id, address, value):
        self.calls.append(("write2", dxl_id, address, value))
        if self.fail_nonzero_goal_current and address == ADDR_GOAL_CURRENT and value != 0:
            return -1001, 0
        self.registers[(dxl_id, address)] = value
        return COMM_SUCCESS, 0

    def getTxRxResult(self, result):
        return "simulated communication failure"

    def getRxPacketError(self, error):
        return "simulated packet error"


def make_driver(handler=None):
    driver = SafeDynamixelDriver.__new__(SafeDynamixelDriver)
    driver._ids = tuple(range(1, 9))
    driver._is_fake = False
    driver._lock = threading.Lock()
    driver._portHandler = object()
    driver._packetHandler = handler or FakePacketHandler()
    driver._gripper_current_mode_enabled = False
    driver._gripper_current_transition_active = False
    driver._gripper_current_limit_raw = None
    driver._gripper_current_spec = None
    driver._gripper_restore_operating_mode = None
    driver._joint_angles = np.zeros(8, dtype=int)
    return driver


def test_id8_enable_sequence_and_current_clamp_never_write_ids_1_to_7():
    handler = FakePacketHandler()
    driver = make_driver(handler)

    info = driver.enable_gripper_current_mode(8, 100.0)
    assert info.model_number == 1200
    assert info.model_name == "XL330-M288-T"
    assert info.current_unit_ma == 1.0

    writes = [call for call in handler.calls if call[0].startswith("write")]
    assert writes[:4] == [
        ("write1", 8, 64, 0),
        ("write1", 8, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE),
        ("write2", 8, ADDR_GOAL_CURRENT, 0),
        ("write1", 8, 64, 1),
    ]
    assert {call[1] for call in writes} == {8}

    assert driver.write_gripper_current_ma(8, 500.0) == 100.0
    assert handler.registers[(8, ADDR_GOAL_CURRENT)] == 100
    assert driver.write_gripper_current_ma(8, -20.0) == -20.0
    assert handler.registers[(8, ADDR_GOAL_CURRENT)] == 0xFFEC

    driver.disable_gripper_current_mode(8)
    assert handler.registers[(8, ADDR_GOAL_CURRENT)] == 0
    assert handler.registers[(8, 64)] == 0
    assert handler.registers[(8, ADDR_OPERATING_MODE)] == POSITION_CONTROL_MODE


def test_current_control_rejects_every_id_except_8():
    driver = make_driver()

    for dxl_id in range(1, 8):
        with pytest.raises(PermissionError, match="ID8"):
            driver.enable_gripper_current_mode(dxl_id, 20.0)


def test_unknown_physical_id8_model_is_rejected_before_any_write():
    handler = FakePacketHandler(model_number=9999)
    driver = make_driver(handler)

    with pytest.raises(RuntimeError, match="unsupported model number 9999"):
        driver.enable_gripper_current_mode(8, 20.0)

    assert not [call for call in handler.calls if call[0].startswith("write")]


def test_current_write_failure_zeros_disables_torque_and_restores_mode():
    handler = FakePacketHandler()
    driver = make_driver(handler)
    driver.enable_gripper_current_mode(8, 50.0)
    handler.fail_nonzero_goal_current = True

    with pytest.raises(RuntimeError, match="Failed to write ID8 current"):
        driver.write_gripper_current_ma(8, 20.0)

    assert handler.registers[(8, ADDR_GOAL_CURRENT)] == 0
    assert handler.registers[(8, 64)] == 0
    assert handler.registers[(8, ADDR_OPERATING_MODE)] == POSITION_CONTROL_MODE
    assert driver._gripper_current_mode_enabled is False


def test_present_position_still_produces_gripper_pos_in_current_mode():
    handler = FakePacketHandler()
    driver = make_driver(handler)
    robot = ContinuousDynamixelRobot.__new__(ContinuousDynamixelRobot)
    robot._driver = driver
    robot._joint_ids = tuple(range(1, 9))
    robot._joint_offsets = np.zeros(8)
    robot._joint_signs = np.ones(8)
    robot.gripper_open_close = (0.0, np.pi)
    robot._last_pos = None
    robot._alpha = 1.0

    robot.enable_gripper_current_mode(20.0)
    driver._joint_angles[-1] = 1024
    assert robot.get_joint_state()[-1] == pytest.approx(0.5)
    driver._joint_angles[-1] = 1536
    assert robot.get_joint_state()[-1] == pytest.approx(0.75)

    robot.disable_gripper_current_mode()
