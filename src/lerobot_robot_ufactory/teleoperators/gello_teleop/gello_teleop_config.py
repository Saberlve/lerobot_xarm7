#!/usr/bin/env python

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from lerobot.teleoperators import TeleoperatorConfig

@dataclass
class GelloFeedbackConfig:
    """Conditioned G2-current feedback sent to the GELLO gripper."""

    enabled: bool = False
    # Calibrated defaults for the XL330-M077-T gripper on GELLO ID8.
    bias_ma: Optional[float] = 0.0
    deadzone_ma: Optional[float] = 30.0
    input_limit_ma: Optional[float] = 1000.0
    ema_beta: Optional[float] = 0.5
    gain: Optional[float] = 0.12
    output_sign: Optional[int] = 1
    output_limit_ma: Optional[float] = 80.0
    slew_rate_ma_s: Optional[float] = 100.0
    timeout_s: Optional[float] = 0.1


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
    # Phase 2 authorization switch. This does not enable current mode during
    # connect; callers must still explicitly call enable_gripper_current_mode().
    gripper_current_control_enabled: bool = False
    # Required when current control is authorized. The adapter also enforces an
    # independent hard ceiling of 100 mA and checks ID8's hardware Current Limit.
    gripper_current_limit_ma: Optional[float] = None
    feedback: GelloFeedbackConfig = field(default_factory=GelloFeedbackConfig)
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
        if self.gripper_current_limit_ma is not None:
            if (
                isinstance(self.gripper_current_limit_ma, bool)
                or not isinstance(self.gripper_current_limit_ma, (int, float))
                or not math.isfinite(self.gripper_current_limit_ma)
                or self.gripper_current_limit_ma <= 0
                or self.gripper_current_limit_ma > 100.0
            ):
                raise ValueError(
                    "gripper_current_limit_ma must be finite, positive, and no greater than 100 mA"
                )
        if self.gripper_current_control_enabled:
            if self.gripper_id != 8:
                raise ValueError("gripper current control is restricted to gripper_id 8")
            if 8 in self.joint_ids:
                raise ValueError("Dynamixel ID8 must not be included in joint_ids")
            if self.gripper_current_limit_ma is None:
                raise ValueError(
                    "gripper_current_limit_ma must be explicitly configured "
                    "before enabling current control"
                )
        if self.feedback.bias_ma is not None:
            if (
                isinstance(self.feedback.bias_ma, bool)
                or not isinstance(self.feedback.bias_ma, (int, float))
                or not math.isfinite(self.feedback.bias_ma)
            ):
                raise ValueError("gripper_feedback_bias_ma must be finite")
        if self.feedback.deadzone_ma is not None:
            if (
                isinstance(self.feedback.deadzone_ma, bool)
                or not isinstance(self.feedback.deadzone_ma, (int, float))
                or not math.isfinite(self.feedback.deadzone_ma)
                or self.feedback.deadzone_ma < 0
            ):
                raise ValueError(
                    "gripper_feedback_deadzone_ma must be finite and non-negative"
                )
        if self.feedback.input_limit_ma is not None:
            if (
                isinstance(self.feedback.input_limit_ma, bool)
                or not isinstance(self.feedback.input_limit_ma, (int, float))
                or not math.isfinite(self.feedback.input_limit_ma)
                or self.feedback.input_limit_ma <= 0
            ):
                raise ValueError(
                    "gripper_feedback_input_limit_ma must be finite and positive"
                )
        if self.feedback.ema_beta is not None:
            if (
                isinstance(self.feedback.ema_beta, bool)
                or not isinstance(self.feedback.ema_beta, (int, float))
                or not math.isfinite(self.feedback.ema_beta)
                or not 0 <= self.feedback.ema_beta < 1
            ):
                raise ValueError(
                    "gripper_feedback_ema_beta must be finite and in [0, 1)"
                )
        if self.feedback.gain is not None:
            if (
                isinstance(self.feedback.gain, bool)
                or not isinstance(self.feedback.gain, (int, float))
                or not math.isfinite(self.feedback.gain)
                or self.feedback.gain < 0
            ):
                raise ValueError("gripper_feedback_gain must be finite and non-negative")
        if (
            self.feedback.output_sign is not None
            and (
                isinstance(self.feedback.output_sign, bool)
                or self.feedback.output_sign not in (-1, 1)
            )
        ):
            raise ValueError("gripper_feedback_output_sign must be either -1 or 1")
        if self.feedback.output_limit_ma is not None:
            if (
                isinstance(self.feedback.output_limit_ma, bool)
                or not isinstance(self.feedback.output_limit_ma, (int, float))
                or not math.isfinite(self.feedback.output_limit_ma)
                or self.feedback.output_limit_ma <= 0
            ):
                raise ValueError(
                    "gripper_feedback_output_limit_ma must be finite and positive"
                )
            if (
                self.gripper_current_limit_ma is not None
                and self.feedback.output_limit_ma
                > self.gripper_current_limit_ma
            ):
                raise ValueError(
                    "gripper_feedback_output_limit_ma cannot exceed "
                    "gripper_current_limit_ma"
                )
        if self.feedback.slew_rate_ma_s is not None:
            if (
                isinstance(self.feedback.slew_rate_ma_s, bool)
                or not isinstance(self.feedback.slew_rate_ma_s, (int, float))
                or not math.isfinite(self.feedback.slew_rate_ma_s)
                or self.feedback.slew_rate_ma_s <= 0
            ):
                raise ValueError(
                    "gripper_feedback_slew_rate_ma_s must be finite and positive"
                )
        if self.feedback.timeout_s is not None:
            if (
                isinstance(self.feedback.timeout_s, bool)
                or not isinstance(self.feedback.timeout_s, (int, float))
                or not math.isfinite(self.feedback.timeout_s)
                or self.feedback.timeout_s <= 0
            ):
                raise ValueError(
                    "gripper_feedback_timeout_s must be finite and positive"
                )
        if self.feedback.enabled:
            if not self.gripper_current_control_enabled:
                raise ValueError(
                    "gripper_current_control_enabled must be true when force feedback "
                    "is enabled"
                )
            required_feedback_fields = (
                "bias_ma", "deadzone_ma", "input_limit_ma", "ema_beta", "gain",
                "output_sign", "output_limit_ma", "slew_rate_ma_s", "timeout_s",
            )
            missing = [
                name for name in required_feedback_fields if getattr(self.feedback, name) is None
            ]
            if missing:
                raise ValueError(
                    "force feedback requires explicit configuration for: "
                    + ", ".join(missing)
                )
        if self.gripper_keyboard_step_mm <= 0:
            raise ValueError("gripper_keyboard_step_mm must be positive")
        if self.gripper_keyboard_hold_delay_s < 0:
            raise ValueError("gripper_keyboard_hold_delay_s must be non-negative")
