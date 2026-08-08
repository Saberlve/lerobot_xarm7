"""Model-agnostic action safety checks.

Validates the final action dict right before it is sent to the robot; any
failed check triggers an emergency halt. Only relies on the action key
naming convention: `[prefix.]pose.x/y/z/rx/ry/rz`, `*.gripper.pos`, so it
works with any policy (ACT / DP / pi0 / custom backbones) and with both
single-arm and multi-arm setups.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations

POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


@dataclass
class ActionSafetyConfig:
    """Configuration for ActionSafetyGuard.

    All limits are compared against the robot's current actual pose per
    control step. `workspace_min`/`workspace_max` use the same unit as the
    pose commands (e.g. mm) and must be provided together.
    """

    enabled: bool = True
    max_step_mm: float = 25.0
    max_rot_step: float = 0.35
    workspace_min: list | None = None
    workspace_max: list | None = None

    def __post_init__(self):
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must be provided together")


def _rotation_delta_norm(rot_prev, rot_curr) -> float:
    """Relative rotation angle (rad) between two axis-angle rotations.

    Uses rotation matrices for the diff to avoid the ±π discontinuity
    of subtracting raw rotvecs.
    """
    R_prev = Transformations.rxryrz_to_rotation_matrix(*rot_prev)
    R_curr = Transformations.rxryrz_to_rotation_matrix(*rot_curr)
    R_delta = R_prev.T @ R_curr
    delta = Transformations.rotation_matrix_to_rxryrz(R_delta)
    return float(np.linalg.norm(delta))


class ActionSafetyGuard:
    """Safety checks for the action dict about to be sent to the robot.

    check() returns None when the action is safe; otherwise it returns a
    human-readable violation reason (the caller should trigger an e-stop).
    """

    def __init__(self, config: ActionSafetyConfig | None = None):
        config = config or ActionSafetyConfig()
        self.config = config
        self.enabled = config.enabled
        self.max_step_mm = config.max_step_mm
        self.max_rot_step = config.max_rot_step
        self.workspace_min = None if config.workspace_min is None else np.asarray(config.workspace_min, dtype=np.float64)
        self.workspace_max = None if config.workspace_max is None else np.asarray(config.workspace_max, dtype=np.float64)

    def check(self, action: dict, curr_robot_dict: dict, keys) -> str | None:
        """Check the action dict. `action` holds absolute pose commands and
        `curr_robot_dict` holds each arm's current actual pose."""
        if not self.enabled:
            return None

        for key in keys:
            prefix = f".{key}" if key else ""
            pose_keys = [f"{prefix}pose.{axis}" for axis in POSE_AXES]
            if not all(k in action for k in pose_keys):
                continue  # not TCP-pose controlled (e.g. joint-only arm), skip pose checks

            target = np.array([action[k] for k in pose_keys], dtype=np.float64)

            # 1. Numeric validity: NaN / Inf
            if not np.all(np.isfinite(target)):
                return f"[{key or 'arm'}] action contains NaN/Inf: {target.tolist()}"

            # 2. Single-step delta limit, measured from the robot's current actual pose
            curr = curr_robot_dict.get(key)
            if curr is not None and curr.get("type") == 1:
                curr_pose = np.asarray(curr["pose"], dtype=np.float64)
                pos_dist = float(np.linalg.norm(target[:3] - curr_pose[:3]))
                if pos_dist > self.max_step_mm:
                    return (
                        f"[{key or 'arm'}] position step {pos_dist:.1f}mm exceeds limit "
                        f"{self.max_step_mm}mm (current {curr_pose[:3].tolist()} -> target {target[:3].tolist()})"
                    )
                rot_delta = _rotation_delta_norm(curr_pose[3:6], target[3:6])
                if rot_delta > self.max_rot_step:
                    return (
                        f"[{key or 'arm'}] rotation step {rot_delta:.3f}rad exceeds limit "
                        f"{self.max_rot_step}rad"
                    )

            # 3. Workspace bounding box (optional)
            if self.workspace_min is not None:
                pos = target[:3]
                if np.any(pos < self.workspace_min) or np.any(pos > self.workspace_max):
                    return (
                        f"[{key or 'arm'}] target position {pos.tolist()} outside workspace "
                        f"[{self.workspace_min.tolist()}, {self.workspace_max.tolist()}]"
                    )

        # Gripper numeric check
        for k, v in action.items():
            if "gripper" in k and isinstance(v, (int, float)) and not math.isfinite(v):
                return f"[{k}] gripper action contains NaN/Inf: {v}"

        return None

    def log_config(self):
        ws = (
            f"[{self.workspace_min.tolist()}, {self.workspace_max.tolist()}]"
            if self.workspace_min is not None
            else "not set"
        )
        logging.info(
            f"ActionSafetyGuard: max_step={self.max_step_mm}mm, "
            f"max_rot_step={self.max_rot_step}rad, workspace={ws}"
        )
