#!/usr/bin/env python
import logging
import time
import math
import numpy as np
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..base_teleop import UFBaseTeleop
from .gello_teleop_config import GelloTeleopConfig


logger = logging.getLogger(__name__)

GELLO_RESET_SPEED_DEG = 30.0
GELLO_RESET_TOLERANCE_DEG = 2.0
GELLO_RESET_CONTROL_HZ = 50.0
GELLO_RESET_TIMEOUT_MARGIN_S = 5.0

class GelloTeleop(UFBaseTeleop):
    """
    GELLO for xArm tele-op, ref: https://wuphilipp.github.io/gello_site/
    """

    config_class = GelloTeleopConfig
    name = "Gello Teleop For xArm"

    def __init__(self, config: GelloTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._teleop_enabled = False
        self._is_calibrated = True # CHECK!!

        from gello.dynamixel.driver import DynamixelDriver
        from gello.agents.gello_agent import DynamixelRobotConfig

        # auto get joint offset from gello
        joint_ids = []
        joint_ids.extend(self.config.joint_ids)
        if self.config.gripper_id >= 0:
            joint_ids.append(self.config.gripper_id)
        driver = DynamixelDriver(joint_ids, port=self.config.port, baudrate=57600)
        for _ in range(10):
            driver.get_joints()  # warmup
        curr_joints = driver.get_joints()
        driver.close()
        joint_offsets = []
        start_joints = list(map(math.radians, self.config.start_joints))
        for i in range(len(start_joints)):
            offset = curr_joints[i] - start_joints[i] / self.config.joint_signs[i]
            joint_offsets.append(offset)
        if self.config.gripper_id >= 0:
            gripper_config = [self.config.gripper_id, np.rad2deg(curr_joints[-1]) - 0.2, np.rad2deg(curr_joints[-1]) - 42]
        else:
            gripper_config = None

        param_dict = {
                "joint_ids": self.config.joint_ids,
                "joint_signs": self.config.joint_signs,
                "joint_offsets": joint_offsets,
                "gripper_config": gripper_config
        }
        self._dynamixel_robo_config = DynamixelRobotConfig(**param_dict)
        print(self._dynamixel_robo_config)
        self.dof = len(start_joints)

    @property
    def action_features(self) -> dict:
        # Add one more dof for gripper
        # act_ft = {
        #     "joint_position": {
        #     "dtype": "float",
        #     "shape": (self.dof+1,)
        #     }
        # }
        act_ft = { f"J{i+1}.pos": float for i in range(self.dof) } | {"gripper.pos": float}
        return act_ft

    @property
    def feedback_features(self) -> dict:
        # fbk_ft = {
        #     "joint_position": {
        #     "dtype": "float",
        #     "shape": (self.dof+1,)
        #     }
        # }
        fbk_ft = { f"J{i+1}.pos": float for i in range(self.dof) } | {"gripper.pos": float}
        return fbk_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        from gello.agents.gello_agent import GelloAgent

        self.gello_agent = GelloAgent(port=self.config.port, dynamixel_config=self._dynamixel_robo_config)
        self.gello_agent._robot.set_torque_mode(False)
        if not self._is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        self._is_connected = True
        super().connect(calibrate)
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        # TODO: Go to sync position slowly? Can not 
        pass

    def reset_to_robot_observation(self, obs):
        """Move the physical Gello to the robot's post-reset joint state."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")

        self._teleop_enabled = False
        gello_robot = self.gello_agent._robot
        driver = gello_robot._driver
        current_raw = np.asarray(driver.get_joints(), dtype=float)
        target_raw = current_raw.copy()
        signs = np.asarray(gello_robot._joint_signs, dtype=float)
        offsets = np.asarray(gello_robot._joint_offsets, dtype=float)

        target_robot_joints = np.asarray(
            [obs[f"J{i + 1}.pos"] for i in range(self.dof)], dtype=float
        )
        target_raw[: self.dof] = target_robot_joints * signs[: self.dof] + offsets[: self.dof]

        if gello_robot.gripper_open_close is not None and len(target_raw) > self.dof:
            gripper_pos = float(obs.get("gripper.pos", 0.0))
            gripper_open, gripper_close = gello_robot.gripper_open_close
            gripper_pos = min(max(gripper_pos, 0.0), 1.0)
            target_raw[self.dof] = gripper_open + gripper_pos * (gripper_close - gripper_open)

        arm_delta = np.max(np.abs(target_raw[: self.dof] - current_raw[: self.dof]))
        reset_speed_rad_s = math.radians(GELLO_RESET_SPEED_DEG)
        duration_s = max(0.5, float(arm_delta / reset_speed_rad_s))
        deadline = time.perf_counter() + duration_s + GELLO_RESET_TIMEOUT_MARGIN_S
        success = False

        try:
            gello_robot.set_torque_mode(True)
            start_t = time.perf_counter()
            while True:
                elapsed_s = time.perf_counter() - start_t
                progress = min(elapsed_s / duration_s, 1.0)
                command = current_raw + (target_raw - current_raw) * progress
                driver.set_joints(command.tolist())
                if progress >= 1.0:
                    break
                time.sleep(1.0 / GELLO_RESET_CONTROL_HZ)

            while time.perf_counter() < deadline:
                measured_raw = np.asarray(driver.get_joints(), dtype=float)
                if np.max(np.abs(measured_raw - target_raw)) <= math.radians(GELLO_RESET_TOLERANCE_DEG):
                    success = True
                    break
                driver.set_joints(target_raw.tolist())
                time.sleep(1.0 / GELLO_RESET_CONTROL_HZ)

            if not success:
                raise RuntimeError("Gello did not reach the robot initial point before timeout")
        finally:
            gello_robot.set_torque_mode(False)
            gello_robot._last_pos = None

    def set_teleop_enabled(self, enabled: bool, obs=None):
        if enabled and not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        if not enabled and self._is_connected and hasattr(self, "gello_agent"):
            self.gello_agent._robot.set_torque_mode(False)
        self._teleop_enabled = enabled
        logger.info("Gello teleoperation %s", "enabled" if enabled else "disabled")

    def get_action(self) -> dict[str, np.ndarray]:
        if not self._teleop_enabled:
            raise RuntimeError("Gello teleop is disabled")
        start = time.perf_counter()
        fake_obs = dict({"joint_state": np.array([0.0]*(self.dof+1))}) # for agent.act() argument, actually no use
        action_array = self.gello_agent.act(fake_obs) # current gello joint pos as np.ndarray
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        action = {}
        for i in range(self.dof):
            action.update({f"J{i+1}.pos": action_array[i]})
        action.update({"gripper.pos": action_array[self.dof]})
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if hasattr(self, "gello_agent"):
            self.gello_agent._robot.set_torque_mode(False)
        self._is_connected = False
        self._teleop_enabled = False
        logger.info(f"{self} disconnected.")
