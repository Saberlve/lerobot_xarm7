from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from dynamixel_sdk import COMM_SUCCESS
from dynamixel_sdk.robotis_def import (
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)
from gello.dynamixel import driver as driver_module
from gello.dynamixel.driver import DynamixelDriver
from gello.robots.dynamixel import DynamixelRobot


class SafeDynamixelDriver(DynamixelDriver):
    """GELLO driver with serialized writes and complete torque cleanup."""

    def set_joints(self, joint_angles: Sequence[float]) -> None:
        if len(joint_angles) != len(self._ids):
            raise ValueError("joint_angles must match the configured Dynamixel IDs")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        if self._is_fake:
            self._fake_joint_angles = np.asarray(joint_angles, dtype=float)
            return

        with self._lock:
            try:
                for dxl_id, angle in zip(self._ids, joint_angles, strict=True):
                    position_value = int(angle * 2048 / np.pi)
                    parameter = [
                        DXL_LOBYTE(DXL_LOWORD(position_value)),
                        DXL_HIBYTE(DXL_LOWORD(position_value)),
                        DXL_LOBYTE(DXL_HIWORD(position_value)),
                        DXL_HIBYTE(DXL_HIWORD(position_value)),
                    ]
                    if not self._groupSyncWrite.addParam(dxl_id, parameter):
                        raise RuntimeError(
                            f"Failed to set joint angle for Dynamixel ID {dxl_id}"
                        )

                result = self._groupSyncWrite.txPacket()
                if result != COMM_SUCCESS:
                    detail = self._packetHandler.getTxRxResult(result)
                    raise RuntimeError(
                        f"Failed to syncwrite goal position: {detail} ({result})"
                    )
            finally:
                self._groupSyncWrite.clearParam()

    def set_torque_mode(self, enable: bool) -> None:
        if self._is_fake:
            self._torque_enabled = enable
            return

        torque_value = driver_module.TORQUE_ENABLE if enable else driver_module.TORQUE_DISABLE
        failures = []
        with self._lock:
            for dxl_id in self._ids:
                result, error = self._packetHandler.write1ByteTxRx(
                    self._portHandler,
                    dxl_id,
                    driver_module.ADDR_TORQUE_ENABLE,
                    torque_value,
                )
                if result != COMM_SUCCESS:
                    detail = self._packetHandler.getTxRxResult(result)
                    failures.append(f"ID {dxl_id}: {detail} ({result})")
                    continue
                if error == 0:
                    continue

                if not enable:
                    state, read_result, _ = self._packetHandler.read1ByteTxRx(
                        self._portHandler,
                        dxl_id,
                        driver_module.ADDR_TORQUE_ENABLE,
                    )
                    if read_result == COMM_SUCCESS and state == driver_module.TORQUE_DISABLE:
                        continue

                detail = self._packetHandler.getRxPacketError(error)
                failures.append(f"ID {dxl_id}: {detail} ({error})")

        if failures:
            raise RuntimeError("Failed to set torque mode: " + "; ".join(failures))
        self._torque_enabled = enable


class ContinuousDynamixelRobot(DynamixelRobot):
    """Dynamixel GELLO whose arm joints remain continuous across encoder wrap."""

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        if len(pos) != self.num_dofs():
            raise RuntimeError("Unexpected Dynamixel joint count")

        arm_dofs = len(pos) - 1 if self.gripper_open_close is not None else len(pos)
        if self._last_pos is not None:
            pos[:arm_dofs] += 2 * np.pi * np.round(
                (self._last_pos[:arm_dofs] - pos[:arm_dofs]) / (2 * np.pi)
            )

        if self.gripper_open_close is not None:
            gripper_open, gripper_close = self.gripper_open_close
            gripper_pos = (pos[-1] - gripper_open) / (gripper_close - gripper_open)
            pos[-1] = min(max(0.0, gripper_pos), 1.0)

        if self._last_pos is None:
            self._last_pos = pos
        else:
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos
        return pos


@dataclass
class PatchedDynamixelRobotConfig:
    joint_ids: Sequence[int]
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Optional[Tuple[int, float, float]]

    def __post_init__(self) -> None:
        if len(self.joint_ids) != len(self.joint_offsets):
            raise ValueError("joint_ids and joint_offsets must have the same length")
        if len(self.joint_ids) != len(self.joint_signs):
            raise ValueError("joint_ids and joint_signs must have the same length")

    def make_robot(
        self,
        port: str = "/dev/ttyUSB0",
        start_joints: Optional[np.ndarray] = None,
    ) -> ContinuousDynamixelRobot:
        # Upstream DynamixelRobot imports its driver inside __init__. Replace
        # that symbol only while constructing this instance.
        original_driver = driver_module.DynamixelDriver
        driver_module.DynamixelDriver = SafeDynamixelDriver
        try:
            return ContinuousDynamixelRobot(
                joint_ids=self.joint_ids,
                joint_offsets=self.joint_offsets,
                joint_signs=self.joint_signs,
                real=True,
                port=port,
                gripper_config=self.gripper_config,
                start_joints=start_joints,
            )
        finally:
            driver_module.DynamixelDriver = original_driver
