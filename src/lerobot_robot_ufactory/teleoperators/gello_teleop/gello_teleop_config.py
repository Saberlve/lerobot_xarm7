#!/usr/bin/env python

from dataclasses import dataclass
from typing import Optional, Tuple
from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("uf::gello_teleop")
@dataclass
class GelloTeleopConfig(TeleoperatorConfig):
    # Frequency of the independent GELLO -> xArm realtime control loop.
    realtime_control_fps: int = 30
    # Port to connect to the gello dummy arm
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJZYC7-if00-port0"

    # Others: Calibration angles, joint directions etc
    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    joint_signs: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1) # if follow the original open-sourced gello xarm7 setup
    # Accepted for compatibility but ignored: arm zero offsets are captured
    # from the current GELLO and xArm poses whenever teleoperation is enabled.
    joint_offsets: Optional[Tuple[float, ...]] = None
    # Retained for compatibility with existing xArm5/xArm6 YAML files. GELLO
    # alignment now always uses its current pose when teleoperation is enabled.
    start_joints: Tuple[float, ...] = (0, 0, 0, 90, 0, 90, 0)  # °
    gripper_id: int = 8  # -1: no gripper
    gripper_open_deg: Optional[float] = None
    gripper_close_deg: Optional[float] = None
    gripper_control_mode: str = "gello"
    # Keyboard gripper: distance closed/opened per quick tap of C/O (mm).
    # Must be > 0; Recommended >= 2 mm.
    gripper_keyboard_step_mm: float = 5.0
    # Keyboard gripper: how long C/O must be held (seconds) before the
    # gripper switches from fixed steps to continuous motion at gripper_speed.
    gripper_keyboard_hold_delay_s: float = 0.5
    torque_joint_ids: Tuple[int, ...] = None  # deprecated

    def __post_init__(self):
        self.id = 'gello_teleop' if self.id is None else self.id
        if self.realtime_control_fps <= 0:
            raise ValueError("realtime_control_fps must be positive")
        if len(self.joint_ids) != len(self.joint_signs):
            raise ValueError("joint_ids and joint_signs must have the same length")
        if len(self.joint_ids) != len(self.start_joints):
            raise ValueError("joint_ids and start_joints must have the same length")
        if self.joint_offsets is not None and len(self.joint_ids) != len(self.joint_offsets):
            raise ValueError("joint_ids and joint_offsets must have the same length")
        if (self.gripper_open_deg is None) != (self.gripper_close_deg is None):
            raise ValueError("gripper_open_deg and gripper_close_deg must be set together")
        if self.gripper_control_mode not in ("gello", "keyboard"):
            raise ValueError("gripper_control_mode must be 'gello' or 'keyboard'")
        if self.gripper_keyboard_step_mm <= 0:
            raise ValueError("gripper_keyboard_step_mm must be positive")
        if self.gripper_keyboard_hold_delay_s < 0:
            raise ValueError("gripper_keyboard_hold_delay_s must be non-negative")
