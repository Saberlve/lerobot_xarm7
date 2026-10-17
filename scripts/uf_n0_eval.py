"""Execute an N0 xArm EEF policy through UFRobot's joint controller.

Run from lerobot_xarm7 with ``python scripts/uf_n0_eval.py``.
The default is a connected dry run: observations and IK, without reset or targets.
"""

from __future__ import annotations

import argparse
import logging
import select
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/eval/xarm7_n0_eval.yaml"
POSE_KEYS = ("pose.x", "pose.y", "pose.z", "pose.r11", "pose.r21", "pose.r31", "pose.r12", "pose.r22", "pose.r32")
TACTILE_SLOTS = ("left_wrist_left_tactile", "left_wrist_right_tactile")


@dataclass
class N0EvalConfig:
    robot: dict
    server_host: str = "127.0.0.1"
    server_port: int = 10093
    unnorm_key: str | None = None
    rgb_keys: list[str] = field(default_factory=lambda: ["third_camera", "wrist_camera"])
    tactile_keys: list[str] = field(default_factory=lambda: ["photon_left", "photon_right"])
    single_task: str = "Pick up the egg and place it in the green egg tray."
    fps: float = 15
    steps_per_inference: int = 50
    temporal_ensemble: bool = False
    steps_per_inference_te: int = 5
    temporal_ensemble_coeff: float = 0.01
    ensemble_gripper: bool = False
    n_episodes: int = 1
    max_episode_steps: int = 900
    dry_run: bool = True
    max_joint_step_deg: float = 15
    ik_position_tolerance_mm: float = 2
    ik_rotation_tolerance_deg: float = 2

    def __post_init__(self):
        for key in ("fps", "max_joint_step_deg", "ik_position_tolerance_mm", "ik_rotation_tolerance_deg"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("steps_per_inference", "steps_per_inference_te", "n_episodes", "max_episode_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.temporal_ensemble, bool) or not isinstance(self.ensemble_gripper, bool):
            raise ValueError("temporal_ensemble and ensemble_gripper must be boolean")
        if not np.isfinite(self.temporal_ensemble_coeff):
            raise ValueError("temporal_ensemble_coeff must be finite")
        if len(self.rgb_keys) != 2 or len(self.tactile_keys) != 2:
            raise ValueError("Supply two RGB keys (external, wrist) and two tactile keys (left, right)")
        if len(set(self.rgb_keys + self.tactile_keys)) != 4:
            raise ValueError("RGB and tactile camera keys must be distinct")
        robot = self.robot
        if (
            robot.get("type") != "uf::robot"
            or robot.get("robot_dof") != 7
            or robot.get("control_space") != "joint"
            or robot.get("record_space") != "both"
        ):
            raise ValueError("N0 EEF execution requires uf::robot, 7 joints, joint control and record_space=both")
        if robot.get("manual_mode", False):
            raise ValueError("manual_mode is incompatible with policy execution")
        if robot.get("min_tcp_z_mm") is None:
            raise ValueError("Set robot.min_tcp_z_mm for the existing TCP height guard")
        if not set(self.rgb_keys + self.tactile_keys) <= robot.get("cameras", {}).keys():
            raise ValueError("All configured RGB/tactile keys must exist under robot.cameras")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be true or false")
        if not self.single_task.strip():
            raise ValueError("single_task must match the training instruction")


def load_config(path: str | Path) -> N0EvalConfig:
    with Path(path).open() as stream:
        return N0EvalConfig(**yaml.safe_load(stream))


def build_observation(robot, obs: dict, cfg: N0EvalConfig) -> dict:
    """Use exactly the recording FK, float32 layout and mm->m conversion."""
    recorded = robot.convert_observation_for_recording(obs)
    prefix = robot.prefix
    raw = np.asarray([recorded[prefix + key] for key in (*POSE_KEYS, "gripper.pos")], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite robot EEF state")
    state = np.zeros(32, dtype=np.float32)
    state[:10] = raw
    state[:3] *= 0.001
    images = [np.asarray(obs[prefix + key]).copy() for key in cfg.rgb_keys]
    return {
        "state": state,
        "image": [*images, np.zeros_like(images[1])],
        "image_mask": [True, True, False],
        "lang": cfg.single_task,
        # N0PolicyClient copies the first frame as the episode baseline.
        "tactile": {slot: np.asarray(obs[prefix + key]).copy() for slot, key in zip(TACTILE_SLOTS, cfg.tactile_keys)},
    }


def eef_target(action: np.ndarray) -> np.ndarray:
    """Absolute canonical EEF -> 4x4 TCP target (mm), with valid SO(3) rotation."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (32,) or not np.isfinite(action).all():
        raise ValueError("Expected one finite absolute canonical 32D action")
    first, second = action[3:6], action[6:9]
    norm = np.linalg.norm(first)
    if norm < 1e-8:
        raise ValueError("Degenerate rot6d first column")
    first = first / norm
    second = second - np.dot(first, second) * first
    norm = np.linalg.norm(second)
    if norm < 1e-8:
        raise ValueError("Degenerate rot6d second column")
    second = second / norm
    target = np.eye(4)
    target[:3, :3] = np.stack([first, second, np.cross(first, second)], axis=1)
    target[:3, 3] = action[:3] * 1000
    return target


def sdk_rpy_pose(target: np.ndarray) -> list[float]:
    """SDK IK consumes Rz(yaw) Ry(pitch) Rx(roll), in mm and radians."""
    rotation = target[:3, :3]
    cos_pitch = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], cos_pitch)
    if cos_pitch > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return [*target[:3, 3].tolist(), float(roll), float(pitch), float(yaw)]


def joint_action(robot, action: np.ndarray, reference: np.ndarray, cfg: N0EvalConfig) -> dict:
    """Solve near the last sent joints; verify IK against the recording FK."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (7,) or not np.isfinite(reference).all():
        raise ValueError("IK requires seven finite reference joints")
    target = eef_target(action)
    if target[2, 3] < cfg.robot["min_tcp_z_mm"]:
        raise ValueError(f"Predicted TCP z {target[2, 3]:.2f} mm is below robot.min_tcp_z_mm")
    code, angles = robot.real_arm.get_inverse_kinematics(
        sdk_rpy_pose(target),
        input_is_radian=True,
        return_is_radian=True,
        limited=False,
        ref_angles=np.asarray(reference).tolist(),
    )
    if code != 0:
        raise RuntimeError(f"xArm IK failed, code={code}")
    joints = np.asarray(angles, dtype=np.float64)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("xArm IK returned invalid joints")
    if np.max(np.abs(joints - reference)) > np.deg2rad(cfg.max_joint_step_deg):
        raise ValueError("IK joint jump exceeds max_joint_step_deg; no target was sent")
    actual = robot._local_kinematics.forward_matrix(joints)
    if np.shape(actual) != (4, 4) or not np.isfinite(actual).all():
        raise ValueError("Recording FK returned an invalid transform")
    position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
    rotation_error = np.arccos(
        np.clip(
            (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2,
            -1,
            1,
        )
    )
    if position_error > cfg.ik_position_tolerance_mm or rotation_error > np.deg2rad(cfg.ik_rotation_tolerance_deg):
        raise ValueError(f"IK/FK mismatch: {position_error:.3f} mm, {np.rad2deg(rotation_error):.3f} deg")
    return {
        **{f"{robot.prefix}J{i + 1}.pos": float(v) for i, v in enumerate(joints)},
        f"{robot.prefix}gripper.pos": float(np.clip(action[9], 0, 1)),
    }


def make_robot(cfg: N0EvalConfig):
    import draccus
    import lerobot_robot_ufactory  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    values = {key: value for key, value in cfg.robot.items() if key != "type"}
    if cfg.dry_run:
        values["no_action"] = True
    elif values.get("no_action", False):
        raise ValueError("Disable robot.no_action to execute")
    return UFRobot(draccus.decode(UFRobotConfig, values))


def make_policy(cfg: N0EvalConfig):
    from starvla_client.n0_policy_client import N0PolicyClient

    return N0PolicyClient(host=cfg.server_host, port=cfg.server_port, unnorm_key=cfg.unnorm_key)


def read_command() -> str | None:
    """Terminal controls work over SSH too; input is line buffered."""
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def run(
    cfg: N0EvalConfig,
    *,
    robot_factory=make_robot,
    policy_factory=make_policy,
    command_reader=read_command,
    sleep=time.sleep,
):
    robot = policy = te = None
    exit_requested = False

    def poll_exit():
        nonlocal exit_requested
        exit_requested = exit_requested or command_reader() == "q"
        return exit_requested

    try:
        # Fail on incompatible policy metadata before touching the robot.
        policy = policy_factory(cfg)
        if (
            policy.profile["action_layout"] != "canonical_eef_32"
            or policy.state_dim != 32
            or policy.action_dim != 32
            or policy.chunk_size < 1
        ):
            raise ValueError("This runner requires an N0 single-arm canonical 32D EEF checkpoint")
        if not cfg.temporal_ensemble and cfg.steps_per_inference > policy.chunk_size:
            raise ValueError("steps_per_inference exceeds the policy's chunk size")
        if cfg.temporal_ensemble:
            from starvla_client.n0_policy_client import N0TemporalEnsembleClientWrapper

            te = N0TemporalEnsembleClientWrapper(
                policy, query_period=cfg.steps_per_inference_te, action_horizon=policy.chunk_size,
                coeff=cfg.temporal_ensemble_coeff, ensemble_gripper=cfg.ensemble_gripper,
            )
            logging.info("N0 TE enabled: period=%d, horizon=%d, coeff=%g, ensemble_gripper=%s",
                         cfg.steps_per_inference_te, policy.chunk_size,
                         cfg.temporal_ensemble_coeff, cfg.ensemble_gripper)
        robot = robot_factory(cfg)
        robot.connect(defer_motion=True)
        if robot._local_kinematics is None:
            raise RuntimeError("Recording FK was not initialized")
        logging.info(
            "Mode=%s, fps=%s, execution horizon=%s. r + Enter: next episode; q + Enter or Ctrl-C: stop.",
            "DRY RUN" if cfg.dry_run else "EXECUTE",
            cfg.fps,
            cfg.steps_per_inference,
        )
        for episode in range(cfg.n_episodes):
            if poll_exit():
                break
            if not cfg.dry_run:
                robot.reset_to_initial(cancel_check=poll_exit)
            (te if te is not None else policy).reset()
            logging.info("Episode %d/%d", episode + 1, cfg.n_episodes)
            actions = None
            offset = 0
            reference = None
            for step in range(cfg.max_episode_steps):
                command = command_reader()
                if command == "q":
                    exit_requested = True
                if exit_requested or command == "r":
                    break
                if te is not None:
                    obs = robot.get_realtime_observation()
                    if reference is None:
                        reference = np.asarray([obs[f"{robot.prefix}J{i}.pos"] for i in range(1, 8)])
                    tick_start = time.monotonic()
                    action = te.get_action(build_observation(robot, obs, cfg))
                    command = command_reader()
                    if command in {"r", "q"}:
                        exit_requested = command == "q"
                        break
                elif actions is None or offset >= cfg.steps_per_inference:
                    obs = robot.get_realtime_observation()
                    reference = np.asarray([obs[f"{robot.prefix}J{i}.pos"] for i in range(1, 8)])
                    start = time.monotonic()
                    actions = np.asarray(policy.predict_action(build_observation(robot, obs, cfg))["actions"])
                    if actions.shape != (policy.chunk_size, 32) or not np.isfinite(actions).all():
                        raise ValueError("Policy returned invalid canonical EEF actions")
                    logging.info(
                        "Episode=%d step=%d inference=%.1f ms", episode + 1, step, (time.monotonic() - start) * 1000
                    )
                    offset = 0
                    # Discard a completed inference when reset/exit arrived while waiting.
                    command = command_reader()
                    if command in {"r", "q"}:
                        exit_requested = command == "q"
                        break
                start = tick_start if te is not None else time.monotonic()
                if te is None:
                    action = actions[offset]
                command = joint_action(robot, action, reference, cfg)
                if cfg.dry_run:
                    sent = command
                else:
                    sent = robot.send_action(command)
                reference = np.asarray([sent[f"{robot.prefix}J{i}.pos"] for i in range(1, 8)])
                offset += 1
                sleep(max(0, 1 / cfg.fps - (time.monotonic() - start)))
            if not cfg.dry_run:
                robot.pause_motion()
            if exit_requested:
                break
    except InterruptedError:
        if not exit_requested:
            raise
    finally:
        # Cleanup also covers camera/connection failure and KeyboardInterrupt.
        if robot is not None:
            try:
                if robot.is_connected and not cfg.dry_run:
                    robot.pause_motion()
            finally:
                try:
                    robot.disconnect()
                finally:
                    if policy is not None:
                        (te if te is not None else policy).close()
        elif policy is not None:
            (te if te is not None else policy).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Reset robot and send joint/gripper targets")
    mode.add_argument("--dry-run", action="store_true", help="Observe and validate IK without sending targets")
    parser.add_argument("--check-config", action="store_true", help="Validate local YAML without connecting to server or robot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config_path)
    if args.execute or args.dry_run:
        cfg.dry_run = not args.execute
    if args.check_config:
        logging.info("Local config valid. Policy metadata is checked on server connection.")
        return
    try:
        run(cfg)
    except KeyboardInterrupt:
        logging.info("Stopped")


if __name__ == "__main__":
    main()
