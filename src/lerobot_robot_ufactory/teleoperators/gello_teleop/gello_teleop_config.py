#!/usr/bin/env python

from dataclasses import dataclass
from typing import Optional, Tuple
from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("uf::gello_teleop")
@dataclass
class GelloTeleopConfig(TeleoperatorConfig):
    # Port to connect to the gello dummy arm
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJZYC7-if00-port0"

    # Others: Calibration angles, joint directions etc
    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    joint_signs: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1) # if follow the original open-sourced gello xarm7 setup
    # Raw Dynamixel zero offsets in degrees. When omitted, the current GELLO
    # pose is treated as start_joints for backwards compatibility.
    joint_offsets: Optional[Tuple[float, ...]] = None
    # GELLO encoder calibration reference; this is not the xArm reset target.
    start_joints: Tuple[float, ...] = (0, 0, 0, 90, 0, 90, 0)  # °
    gripper_id: int = 8  # -1: no gripper
    gripper_open_deg: Optional[float] = None
    gripper_close_deg: Optional[float] = None
    reset_speed_deg_s: float = 10.0
    torque_joint_ids: Tuple[int, ...] = None  # deprecated; reset controls all GELLO joints.

    def __post_init__(self):
        self.id = 'gello_teleop' if self.id is None else self.id
        if len(self.joint_ids) != len(self.joint_signs):
            raise ValueError("joint_ids and joint_signs must have the same length")
        if len(self.joint_ids) != len(self.start_joints):
            raise ValueError("joint_ids and start_joints must have the same length")
        if self.joint_offsets is not None and len(self.joint_ids) != len(self.joint_offsets):
            raise ValueError("joint_ids and joint_offsets must have the same length")
        if (self.gripper_open_deg is None) != (self.gripper_close_deg is None):
            raise ValueError("gripper_open_deg and gripper_close_deg must be set together")
        if self.reset_speed_deg_s <= 0:
            raise ValueError("reset_speed_deg_s must be positive")
