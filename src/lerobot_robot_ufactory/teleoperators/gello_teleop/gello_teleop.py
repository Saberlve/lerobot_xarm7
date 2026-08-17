#!/usr/bin/env python
import logging
import numpy as np
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..base_teleop import UFBaseTeleop
from .gello_adapter import PatchedDynamixelRobotConfig
from .gello_teleop_config import GelloTeleopConfig


logger = logging.getLogger(__name__)

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
        self._needs_alignment = True
        self._is_calibrated = True # CHECK!!

        joint_offsets = [0.0] * len(self.config.joint_ids)
        self._align_gripper_to_current = self.config.gripper_open_deg is None
        if self.config.gripper_id >= 0:
            if self.config.gripper_open_deg is not None:
                gripper_open_deg = self.config.gripper_open_deg
                gripper_close_deg = self.config.gripper_close_deg
            else:
                # Only the range matters. It is shifted to the current GELLO
                # gripper position whenever teleoperation is enabled.
                gripper_open_deg = 0.0
                gripper_close_deg = -42.0
            gripper_config = [
                self.config.gripper_id,
                gripper_open_deg,
                gripper_close_deg,
            ]
        else:
            gripper_config = None

        param_dict = {
                "joint_ids": self.config.joint_ids,
                "joint_signs": self.config.joint_signs,
                "joint_offsets": joint_offsets,
                "gripper_config": gripper_config
        }
        self._dynamixel_robo_config = PatchedDynamixelRobotConfig(**param_dict)
        print(self._dynamixel_robo_config)
        self.dof = len(self.config.joint_ids)

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

        try:
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
        except BaseException:
            self._is_connected = False
            self._close_gello_driver()
            raise
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
        """Map the current passive GELLO pose to the robot's current pose."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")

        self._teleop_enabled = False
        gello_robot = self.gello_agent._robot
        driver = gello_robot._driver
        gello_robot.set_torque_mode(False)
        current_raw = np.asarray(driver.get_joints(), dtype=float)
        signs = np.asarray(gello_robot._joint_signs, dtype=float)

        robot_joints = np.asarray(
            [obs[f"J{i + 1}.pos"] for i in range(self.dof)], dtype=float
        )
        gello_robot._joint_offsets[: self.dof] = (
            current_raw[: self.dof] - robot_joints * signs[: self.dof]
        )

        if (
            self._align_gripper_to_current
            and gello_robot.gripper_open_close is not None
            and len(current_raw) > self.dof
        ):
            gripper_pos = float(obs.get("gripper.pos", 0.0))
            gripper_open, gripper_close = gello_robot.gripper_open_close
            gripper_pos = min(max(gripper_pos, 0.0), 1.0)
            gripper_span = gripper_close - gripper_open
            gripper_open = current_raw[self.dof] - gripper_pos * gripper_span
            gello_robot.gripper_open_close = (
                gripper_open,
                gripper_open + gripper_span,
            )

        gello_robot._last_pos = None
        self._needs_alignment = False
        logger.info("Current GELLO pose aligned to current robot observation")

    def set_teleop_enabled(self, enabled: bool, obs=None):
        if enabled and not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        if enabled and self._needs_alignment:
            if obs is None:
                raise ValueError("Robot observation is required to enable GELLO teleoperation")
            self.reset_to_robot_observation(obs)
        if not enabled and self._is_connected and hasattr(self, "gello_agent"):
            self.gello_agent._robot.set_torque_mode(False)
            self._needs_alignment = True
        self._teleop_enabled = enabled
        logger.info("Gello teleoperation %s", "enabled" if enabled else "disabled")

    def get_action(self) -> dict[str, np.ndarray]:
        if not self._teleop_enabled:
            raise RuntimeError("Gello teleop is disabled")
        fake_obs = dict({"joint_state": np.array([0.0]*(self.dof+1))}) # for agent.act() argument, actually no use
        action_array = self.gello_agent.act(fake_obs) # current gello joint pos as np.ndarray

        action = {}
        for i in range(self.dof):
            action.update({f"J{i+1}.pos": action_array[i]})
        action.update({"gripper.pos": action_array[self.dof]})
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def _close_gello_driver(self) -> None:
        if not hasattr(self, "gello_agent"):
            return
        gello_robot = self.gello_agent._robot
        try:
            gello_robot.set_torque_mode(False)
        finally:
            gello_robot._driver.close()

    def disconnect(self) -> None:
        try:
            self._close_gello_driver()
        finally:
            self._is_connected = False
            self._teleop_enabled = False
            self._needs_alignment = True
        logger.info(f"{self} disconnected.")
