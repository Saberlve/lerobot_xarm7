from __future__ import annotations

import math
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


GRIPPER_DYNAMIXEL_ID = 8
ADDR_OPERATING_MODE = 11
ADDR_CURRENT_LIMIT = 38
ADDR_GOAL_CURRENT = 102
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3
HARD_MAX_GRIPPER_CURRENT_MA = 100.0


@dataclass(frozen=True)
class DynamixelModelSpec:
    model_number: int
    model_name: str
    current_unit_ma: float
    maximum_current_limit_raw: int


@dataclass(frozen=True)
class GripperDynamixelInfo:
    dxl_id: int
    model_number: int
    model_name: str
    operating_mode: int
    current_limit_raw: int
    current_limit_ma: float
    current_unit_ma: float


# The GELLO mechanical design uses an XL330-M288-T for the shared gripper.
# Model Number is read from the physical ID8 before current mode is enabled;
# an unknown/replaced servo is rejected instead of assuming a compatible table.
SUPPORTED_GRIPPER_MODELS = {
    1200: DynamixelModelSpec(
        model_number=1200,
        model_name="XL330-M288-T",
        current_unit_ma=1.0,
        maximum_current_limit_raw=1750,
    ),
}


class SafeDynamixelDriver(DynamixelDriver):
    """GELLO driver with serialized writes and complete torque cleanup."""

    def __init__(self, *args, **kwargs) -> None:
        self._gripper_current_mode_enabled = False
        self._gripper_current_transition_active = False
        self._gripper_current_limit_raw: int | None = None
        self._gripper_current_spec: DynamixelModelSpec | None = None
        self._gripper_restore_operating_mode: int | None = None
        super().__init__(*args, **kwargs)

    def _require_real_id8(self, dxl_id: int) -> None:
        if dxl_id != GRIPPER_DYNAMIXEL_ID:
            raise PermissionError(
                f"Current control is restricted to GELLO gripper ID{GRIPPER_DYNAMIXEL_ID}"
            )
        if dxl_id not in self._ids:
            raise ValueError(f"Dynamixel ID{dxl_id} is not configured")
        if self._is_fake:
            raise RuntimeError("Current control is unavailable with the fake Dynamixel driver")

    def _check_sdk_result(self, result: int, error: int, operation: str) -> None:
        if result != COMM_SUCCESS:
            detail = self._packetHandler.getTxRxResult(result)
            raise RuntimeError(f"{operation}: {detail} ({result})")
        if error != 0:
            detail = self._packetHandler.getRxPacketError(error)
            raise RuntimeError(f"{operation}: {detail} ({error})")

    def _read1_locked(self, dxl_id: int, address: int, label: str) -> int:
        value, result, error = self._packetHandler.read1ByteTxRx(
            self._portHandler, dxl_id, address
        )
        self._check_sdk_result(result, error, f"Failed to read {label} from ID{dxl_id}")
        return int(value)

    def _read2_locked(self, dxl_id: int, address: int, label: str) -> int:
        value, result, error = self._packetHandler.read2ByteTxRx(
            self._portHandler, dxl_id, address
        )
        self._check_sdk_result(result, error, f"Failed to read {label} from ID{dxl_id}")
        return int(value)

    def _write1_locked(self, dxl_id: int, address: int, value: int, label: str) -> None:
        result, error = self._packetHandler.write1ByteTxRx(
            self._portHandler, dxl_id, address, value
        )
        self._check_sdk_result(result, error, f"Failed to write {label} to ID{dxl_id}")

    def _write2_locked(self, dxl_id: int, address: int, value: int, label: str) -> None:
        result, error = self._packetHandler.write2ByteTxRx(
            self._portHandler, dxl_id, address, value & 0xFFFF
        )
        self._check_sdk_result(result, error, f"Failed to write {label} to ID{dxl_id}")

    def _probe_gripper_locked(self, dxl_id: int) -> GripperDynamixelInfo:
        model_number, result, error = self._packetHandler.ping(self._portHandler, dxl_id)
        self._check_sdk_result(result, error, f"Failed to ping ID{dxl_id}")
        model_number = int(model_number)
        spec = SUPPORTED_GRIPPER_MODELS.get(model_number)
        if spec is None:
            supported = ", ".join(
                f"{item.model_name} ({number})"
                for number, item in SUPPORTED_GRIPPER_MODELS.items()
            )
            raise RuntimeError(
                f"ID{dxl_id} reported unsupported model number {model_number}; "
                f"supported gripper model: {supported}"
            )

        operating_mode = self._read1_locked(
            dxl_id, ADDR_OPERATING_MODE, "Operating Mode(11)"
        )
        current_limit_raw = self._read2_locked(
            dxl_id, ADDR_CURRENT_LIMIT, "Current Limit(38)"
        )
        if not 0 < current_limit_raw <= spec.maximum_current_limit_raw:
            raise RuntimeError(
                f"ID{dxl_id} returned invalid Current Limit raw value {current_limit_raw}"
            )
        return GripperDynamixelInfo(
            dxl_id=dxl_id,
            model_number=model_number,
            model_name=spec.model_name,
            operating_mode=operating_mode,
            current_limit_raw=current_limit_raw,
            current_limit_ma=current_limit_raw * spec.current_unit_ma,
            current_unit_ma=spec.current_unit_ma,
        )

    def probe_gripper_dynamixel(self, dxl_id: int) -> GripperDynamixelInfo:
        """Read and validate the physical ID8 model and current-control limits."""
        self._require_real_id8(dxl_id)
        with self._lock:
            return self._probe_gripper_locked(dxl_id)

    def _best_effort_zero_and_disable_locked(self, dxl_id: int) -> list[str]:
        failures = []
        try:
            self._write2_locked(dxl_id, ADDR_GOAL_CURRENT, 0, "Goal Current(102)=0")
        except Exception as exc:  # safety cleanup must continue
            failures.append(f"zero current failed: {exc}")
        try:
            self._write1_locked(
                dxl_id,
                driver_module.ADDR_TORQUE_ENABLE,
                driver_module.TORQUE_DISABLE,
                "Torque Enable(64)=0",
            )
        except Exception as exc:  # verify a possibly successful write before reporting it
            try:
                torque_state = self._read1_locked(
                    dxl_id, driver_module.ADDR_TORQUE_ENABLE, "Torque Enable(64)"
                )
            except Exception:
                failures.append(f"torque disable failed: {exc}")
            else:
                if torque_state != driver_module.TORQUE_DISABLE:
                    failures.append(f"torque disable failed: {exc}")
        self._gripper_current_mode_enabled = False
        return failures

    def _best_effort_restore_mode_locked(self, dxl_id: int) -> list[str]:
        restore_mode = self._gripper_restore_operating_mode
        if restore_mode is None or restore_mode == CURRENT_CONTROL_MODE:
            return []
        try:
            self._write1_locked(
                dxl_id, ADDR_OPERATING_MODE, restore_mode, "restored Operating Mode(11)"
            )
        except Exception as exc:
            return [f"operating mode restore failed: {exc}"]
        return []

    def _clear_gripper_current_state_locked(self) -> None:
        self._gripper_current_mode_enabled = False
        self._gripper_current_limit_raw = None
        self._gripper_current_spec = None
        self._gripper_restore_operating_mode = None

    def enable_gripper_current_mode(
        self, dxl_id: int, current_limit_ma: float
    ) -> GripperDynamixelInfo:
        """Safely enable Current Control Mode on physical Dynamixel ID8 only."""
        self._require_real_id8(dxl_id)
        with self._lock:
            if self._gripper_current_mode_enabled or self._gripper_current_transition_active:
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise RuntimeError(f"ID{dxl_id} current mode was already active{detail}")
            if isinstance(current_limit_ma, bool):
                raise ValueError("gripper current limit must be a finite number")
            try:
                current_limit_ma = float(current_limit_ma)
            except (TypeError, ValueError) as exc:
                raise ValueError("gripper current limit must be a finite number") from exc
            if not math.isfinite(current_limit_ma) or current_limit_ma <= 0:
                raise ValueError("gripper current limit must be finite and positive")
            if current_limit_ma > HARD_MAX_GRIPPER_CURRENT_MA:
                raise ValueError(
                    f"gripper current limit must not exceed {HARD_MAX_GRIPPER_CURRENT_MA:g} mA"
                )

            info = self._probe_gripper_locked(dxl_id)
            spec = SUPPORTED_GRIPPER_MODELS[info.model_number]
            configured_limit_raw = math.floor(current_limit_ma / spec.current_unit_ma)
            if configured_limit_raw <= 0:
                raise ValueError(
                    f"gripper current limit is below one raw unit ({spec.current_unit_ma:g} mA)"
                )
            if configured_limit_raw > info.current_limit_raw:
                raise ValueError(
                    f"configured current limit {current_limit_ma:g} mA exceeds "
                    f"ID{dxl_id} Current Limit {info.current_limit_ma:g} mA"
                )

            self._gripper_current_transition_active = True
            self._gripper_restore_operating_mode = (
                info.operating_mode
                if info.operating_mode != CURRENT_CONTROL_MODE
                else POSITION_CONTROL_MODE
            )
            try:
                self._write1_locked(
                    dxl_id,
                    driver_module.ADDR_TORQUE_ENABLE,
                    driver_module.TORQUE_DISABLE,
                    "Torque Enable(64)=0",
                )
                self._write1_locked(
                    dxl_id,
                    ADDR_OPERATING_MODE,
                    CURRENT_CONTROL_MODE,
                    "Operating Mode(11)=Current Control(0)",
                )
                actual_mode = self._read1_locked(
                    dxl_id, ADDR_OPERATING_MODE, "Operating Mode(11)"
                )
                if actual_mode != CURRENT_CONTROL_MODE:
                    raise RuntimeError(
                        f"ID{dxl_id} Operating Mode verification failed: "
                        f"got {actual_mode}, expected {CURRENT_CONTROL_MODE}"
                    )
                self._write2_locked(
                    dxl_id, ADDR_GOAL_CURRENT, 0, "Goal Current(102)=0"
                )
                self._write1_locked(
                    dxl_id,
                    driver_module.ADDR_TORQUE_ENABLE,
                    driver_module.TORQUE_ENABLE,
                    "Torque Enable(64)=1",
                )
                torque_state = self._read1_locked(
                    dxl_id, driver_module.ADDR_TORQUE_ENABLE, "Torque Enable(64)"
                )
                if torque_state != driver_module.TORQUE_ENABLE:
                    raise RuntimeError(
                        f"ID{dxl_id} torque verification failed: got {torque_state}"
                    )
            except Exception as exc:
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise RuntimeError(
                    f"Failed to enable ID{dxl_id} current mode: {exc}{detail}"
                ) from exc
            finally:
                self._gripper_current_transition_active = False

            self._gripper_current_spec = spec
            self._gripper_current_limit_raw = configured_limit_raw
            self._gripper_current_mode_enabled = True
            return info

    def write_gripper_current_ma(self, dxl_id: int, current_ma: float) -> float:
        """Write a clamped signed current to ID8 and return the applied mA value."""
        self._require_real_id8(dxl_id)
        with self._lock:
            if (
                not self._gripper_current_mode_enabled
                or self._gripper_current_spec is None
                or self._gripper_current_limit_raw is None
            ):
                raise RuntimeError("ID8 Current Control Mode is not enabled")
            if isinstance(current_ma, bool):
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise ValueError(f"gripper current must be a finite number{detail}")
            try:
                current_ma = float(current_ma)
            except (TypeError, ValueError) as exc:
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise ValueError(f"gripper current must be a finite number{detail}") from exc
            if not math.isfinite(current_ma):
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise ValueError(f"gripper current must be finite{detail}")

            spec = self._gripper_current_spec
            limit_raw = self._gripper_current_limit_raw
            raw = round(current_ma / spec.current_unit_ma)
            raw = max(-limit_raw, min(limit_raw, raw))
            try:
                self._write2_locked(
                    dxl_id, ADDR_GOAL_CURRENT, raw, f"Goal Current(102)={raw}"
                )
            except Exception as exc:
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise RuntimeError(f"Failed to write ID{dxl_id} current: {exc}{detail}") from exc
            return raw * spec.current_unit_ma

    def zero_gripper_current(self, dxl_id: int) -> None:
        """Write zero current to ID8; disable torque if the zero write fails."""
        self._require_real_id8(dxl_id)
        with self._lock:
            if not (
                self._gripper_current_mode_enabled or self._gripper_current_transition_active
            ):
                return
            try:
                self._write2_locked(
                    dxl_id, ADDR_GOAL_CURRENT, 0, "Goal Current(102)=0"
                )
            except Exception as exc:
                cleanup = self._best_effort_zero_and_disable_locked(dxl_id)
                cleanup += self._best_effort_restore_mode_locked(dxl_id)
                self._clear_gripper_current_state_locked()
                detail = f"; safety cleanup: {'; '.join(cleanup)}" if cleanup else ""
                raise RuntimeError(f"Failed to zero ID{dxl_id} current: {exc}{detail}") from exc

    def disable_gripper_current_mode(self, dxl_id: int) -> None:
        """Zero ID8, disable its torque, and restore its prior operating mode."""
        self._require_real_id8(dxl_id)
        with self._lock:
            if not (
                self._gripper_current_mode_enabled or self._gripper_current_transition_active
            ):
                return
            failures = self._best_effort_zero_and_disable_locked(dxl_id)
            failures += self._best_effort_restore_mode_locked(dxl_id)
            self._clear_gripper_current_state_locked()
        if failures:
            raise RuntimeError(
                f"Failed to fully disable ID{dxl_id} current mode: {'; '.join(failures)}"
            )

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

    def close(self) -> None:
        if self._gripper_current_mode_enabled or self._gripper_current_transition_active:
            try:
                self.disable_gripper_current_mode(GRIPPER_DYNAMIXEL_ID)
            except Exception:
                # The driver is closing after best-effort zero and torque disable.
                pass
        super().close()


class ContinuousDynamixelRobot(DynamixelRobot):
    """Dynamixel GELLO whose arm joints remain continuous across encoder wrap."""

    def _require_id8_gripper(self) -> int:
        if self.gripper_open_close is None or not self._joint_ids:
            raise RuntimeError("GELLO gripper is not configured")
        gripper_id = int(self._joint_ids[-1])
        if gripper_id != GRIPPER_DYNAMIXEL_ID:
            raise PermissionError(
                f"Current control is restricted to GELLO gripper ID{GRIPPER_DYNAMIXEL_ID}"
            )
        return gripper_id

    def probe_gripper_dynamixel(self) -> GripperDynamixelInfo:
        return self._driver.probe_gripper_dynamixel(self._require_id8_gripper())

    def enable_gripper_current_mode(
        self, current_limit_ma: float
    ) -> GripperDynamixelInfo:
        return self._driver.enable_gripper_current_mode(
            self._require_id8_gripper(), current_limit_ma
        )

    def write_gripper_current_ma(self, current_ma: float) -> float:
        return self._driver.write_gripper_current_ma(
            self._require_id8_gripper(), current_ma
        )

    def zero_gripper_current(self) -> None:
        self._driver.zero_gripper_current(self._require_id8_gripper())

    def disable_gripper_current_mode(self) -> None:
        self._driver.disable_gripper_current_mode(self._require_id8_gripper())

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
