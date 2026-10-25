"""Model-independent xArm7 canonical EEF decoding and execution.

Consumes absolute 10D actions: metres, rotation columns and gripper.
Policies and observation builders are injected by each model entry.
"""
from __future__ import annotations
import logging
import os
import select
import sys
import time
from dataclasses import dataclass,field
import numpy as np

POSE_KEYS = ("pose.x", "pose.y", "pose.z", "pose.r11", "pose.r21", "pose.r31", "pose.r12", "pose.r22", "pose.r32")


def validate_gripper_config(cfg):
    if cfg.action_space != "eef10" or cfg.temporal_ensemble or cfg.n_episodes != 1:
        raise ValueError("continue_move requires EEF10, one episode and blocking inference")


@dataclass
class EEFExecutionConfig:
    robot: dict
    deployment_profile: str | None = None
    action_space: str = "eef10"
    execution_mode: str = "full"
    gripper_log_dir: str = ""
    server_host: str = "127.0.0.1"
    server_port: int = 10093
    state_stats_key: str | None = None
    camera_keys: list[str] = field(default_factory=lambda: ["third_camera", "wrist_camera"])
    tactile_keys: list[str] = field(default_factory=lambda: ["photon_left", "photon_right"])
    single_task: str = "Pick up the egg and place it in the green egg tray."
    fps: float = 15
    steps_per_inference: int = 50
    action_chunk_size: int | None = None
    temporal_ensemble: bool = False
    steps_per_inference_te: int = 5
    temporal_ensemble_coeff: float = 0.01
    te_ensemble_gripper: bool = False
    n_episodes: int = 1
    max_episode_steps: int = 900
    dry_run: bool = True
    record_video: bool = False
    video_dir: str = ""
    enable_logs: bool = False
    log_dir: str = "logs"
    max_joint_step_deg: float = 90
    ik_position_tolerance_mm: float = 2
    ik_rotation_tolerance_deg: float = 2
    max_tcp_step_mm: float = 75
    max_tcp_rotation_step_deg: float = 45

    def __post_init__(self):
        if self.execution_mode not in ("full", "continue_move"):
            raise ValueError("execution_mode must be full or continue_move")
        if self.execution_mode == "continue_move":
            validate_gripper_config(self)
        if type(self.record_video) is not bool:
            raise ValueError("record_video must be boolean")
        if self.action_space not in ("eef10", "joint"):
            raise ValueError("action_space must be eef10 or joint")
        if self.action_chunk_size is not None and (type(self.action_chunk_size) is not int or self.action_chunk_size < 1):
            raise ValueError("action_chunk_size must be a positive integer")
        limits = (("max_tcp_step_mm", "max_tcp_rotation_step_deg")
                  if self.robot.get("control_space") == "cartesian" else
                  ("max_joint_step_deg", "ik_position_tolerance_mm", "ik_rotation_tolerance_deg"))
        for key in ("fps", *limits):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("steps_per_inference", "steps_per_inference_te", "n_episodes", "max_episode_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.temporal_ensemble, bool) or not isinstance(self.te_ensemble_gripper, bool):
            raise ValueError("temporal_ensemble and te_ensemble_gripper must be boolean")
        if not np.isfinite(self.temporal_ensemble_coeff):
            raise ValueError("temporal_ensemble_coeff must be finite")
        if len(self.camera_keys) != 2 or len(self.tactile_keys) != 2:
            raise ValueError("Supply two RGB keys (external, wrist) and two tactile keys (left, right)")
        if len(set(self.camera_keys + self.tactile_keys)) != 4:
            raise ValueError("RGB and tactile camera keys must be distinct")
        robot = self.robot
        if (
            robot.get("type") != "uf::robot"
            or robot.get("robot_dof") != 7
            or (robot.get("control_space"), robot.get("record_space", "joint"))
            not in (("joint", "both"), ("cartesian", "joint"))
        ):
            raise ValueError("xArm7 EEF requires uf::robot, 7 joints, joint control/record_space=both "
                             "or cartesian control/record_space=joint")
        if robot.get("manual_mode", False):
            raise ValueError("manual_mode is incompatible with policy execution")
        if robot.get("min_tcp_z_mm") is None:
            raise ValueError("Set robot.min_tcp_z_mm for the existing TCP height guard")
        if not set(self.camera_keys + self.tactile_keys) <= robot.get("cameras", {}).keys():
            raise ValueError("All configured RGB/tactile keys must exist under robot.cameras")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be true or false")
        if not self.single_task.strip():
            raise ValueError("single_task must match the training instruction")


def measured_state(robot, obs, cfg):
    if cfg.robot.get("control_space") == "cartesian":
        joints = np.asarray([obs[f"{robot.prefix}J{i}.pos"] for i in range(1, 8)])
        transform = robot._local_kinematics.forward_matrix(joints)
        pose = [*transform[:3, 3], *transform[:3, 0], *transform[:3, 1]]
        recorded = {**obs, **{robot.prefix + key: value for key, value in zip(POSE_KEYS, pose)}}
    else:
        recorded = robot.convert_observation_for_recording(obs)
    raw = np.asarray([recorded[robot.prefix + key] for key in (*POSE_KEYS, "gripper.pos")], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite robot EEF state")
    return raw


def build_observation(robot, obs, cfg):
    raw = measured_state(robot, obs, cfg)
    camera_keys = cfg.camera_keys
    result = dict(state=raw, lang=cfg.single_task,
                  image=[np.asarray(obs[robot.prefix + key]).copy() for key in camera_keys])
    tactile_keys = getattr(cfg, "tactile_keys", [])
    if tactile_keys:
        result["tactile"] = [np.asarray(obs[robot.prefix + key]).copy() for key in tactile_keys]
    return result


def make_policy(cfg):
    from starvla_client.policy_client import PolicyClient
    policy = PolicyClient(host=cfg.server_host, port=cfg.server_port,
                             unnorm_key=cfg.state_stats_key or None)
    try:
        if policy.action_space != cfg.action_space:
            raise ValueError("Evaluation action_space differs from server action_space")
        if cfg.action_chunk_size is not None and policy.chunk_size != cfg.action_chunk_size:
            raise ValueError("Configured action_chunk_size differs from checkpoint chunk size")
        fps = policy.profile.get("fps")
        if fps is not None and fps != cfg.fps:
            raise ValueError("Evaluation fps differs from checkpoint training fps")
        prompt = policy.profile.get("prompt")
        if prompt and prompt.strip() != cfg.single_task.strip():
            raise ValueError("Task instruction differs from checkpoint")
    except Exception:
        policy.close()
        raise
    return policy


def validate_policy(policy):
    if (policy.action_space != "eef10" or policy.action_dim != 10
            or policy.profile.get("position_unit") != "m"
            or policy.profile.get("rotation_representation") != "columns_6d"):
        raise ValueError("Executor requires EEF10 actions in metres and rotation columns")


def eef_target(action: np.ndarray) -> np.ndarray:
    """Absolute canonical EEF -> 4x4 TCP target (mm), with valid SO(3) rotation."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (10,) or not np.isfinite(action).all():
        raise ValueError("Expected one finite absolute EEF10 action")
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
    """SDK IK/Cartesian targets use Rz(yaw) Ry(pitch) Rx(roll), mm and radians."""
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


def joint_action(robot, action: np.ndarray, reference: np.ndarray, cfg: EEFExecutionConfig) -> dict:
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


def cartesian_action(robot, action: np.ndarray, reference: np.ndarray, cfg) -> dict:
    """Canonical EEF -> absolute SDK pose (mm/RPY radians), without client IK."""
    target = eef_target(action)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (4, 4) or not np.isfinite(reference).all():
        raise ValueError("Cartesian execution requires a finite reference transform")
    if target[2, 3] < cfg.robot["min_tcp_z_mm"]:
        raise ValueError("Predicted TCP z is below robot.min_tcp_z_mm; no target was sent")
    distance = np.linalg.norm(target[:3, 3] - reference[:3, 3])
    angle = np.arccos(np.clip((np.trace(reference[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))
    if distance > cfg.max_tcp_step_mm:
        raise ValueError("Cartesian step exceeds max_tcp_step_mm; no target was sent")
    if angle > np.deg2rad(cfg.max_tcp_rotation_step_deg):
        raise ValueError("Cartesian rotation exceeds max_tcp_rotation_step_deg; no target was sent")
    return {
        **{f"{robot.prefix}pose.{key}": value
           for key, value in zip(("x", "y", "z", "rx", "ry", "rz"), sdk_rpy_pose(target))},
        "_pose_is_rpy": True,
        f"{robot.prefix}gripper.pos": float(np.clip(action[9], 0, 1)),
    }


def decode_robot_config(cfg: EEFExecutionConfig):
    import draccus
    import lerobot_robot_ufactory  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    values = {key: value for key, value in cfg.robot.items() if key != "type"}
    if cfg.dry_run:
        values["no_action"] = True
    elif values.get("no_action", False):
        raise ValueError("Disable robot.no_action to execute")
    return draccus.decode(UFRobotConfig, values)


def make_robot(cfg: EEFExecutionConfig):
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot

    return UFRobot(decode_robot_config(cfg))


def read_command() -> str | None:
    """Terminal controls work over SSH too; input is line buffered."""
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def run(
    cfg: EEFExecutionConfig,
    *,
    robot_factory=make_robot,
    policy_factory=make_policy,
    observation_builder=build_observation,
    policy_validator=validate_policy,
    command_reader=read_command,
    sleep=time.sleep,
):
    if getattr(cfg, "execution_mode", "full") not in ("full", "continue_move"):
        raise ValueError("Unsupported execution mode; use full or continue_move")
    robot = policy = te = recording = trace = action_log = None
    diagnostic = getattr(cfg, "execution_mode", "full") == "continue_move"
    if diagnostic:
        validate_gripper_config(cfg)
    exit_requested = False
    cartesian = cfg.robot.get("control_space") == "cartesian"

    def observe():
        # Cartesian commands still use joint-feedback FK for the training state.
        if cartesian:
            return robot.get_realtime_observation(joint_feedback=True)
        return robot.get_realtime_observation()

    def observation_reference(obs):
        joints = np.asarray([obs[f"{robot.prefix}J{i}.pos"] for i in range(1, 8)])
        return robot._local_kinematics.forward_matrix(joints) if cartesian else joints

    def poll_exit():
        nonlocal exit_requested
        exit_requested = exit_requested or command_reader() == "q"
        return exit_requested

    try:
        # Fail on incompatible policy metadata before touching the robot.
        stage_start = time.monotonic()
        logging.info("阶段：连接模型服务并检查策略配置")
        policy = policy_factory(cfg)
        policy_validator(policy)
        expected_chunk = getattr(cfg, "action_chunk_size", None)
        if expected_chunk is not None and policy.chunk_size != expected_chunk:
            raise ValueError(f"Configured action_chunk_size={expected_chunk} differs from checkpoint chunk size {policy.chunk_size}")
        if not cfg.temporal_ensemble and cfg.steps_per_inference > policy.chunk_size:
            raise ValueError("steps_per_inference exceeds the policy's chunk size")
        logging.info("阶段：策略检查完成，耗时 %.2f 秒", time.monotonic() - stage_start)
        if cfg.temporal_ensemble:
            from starvla_client.policy_client import PolicyTemporalEnsembleClientWrapper

            te = PolicyTemporalEnsembleClientWrapper(
                policy, query_period=cfg.steps_per_inference_te, action_horizon=policy.chunk_size,
                coeff=cfg.temporal_ensemble_coeff, ensemble_gripper=cfg.te_ensemble_gripper,
            )
            logging.info("Canonical EEF TE enabled: period=%d, horizon=%d, coeff=%g, ensemble_gripper=%s",
                         cfg.steps_per_inference_te, policy.chunk_size,
                         cfg.temporal_ensemble_coeff, cfg.te_ensemble_gripper)
        stage_start = time.monotonic()
        logging.info("阶段：连接机械臂和相机")
        if diagnostic:
            from lerobot_robot_ufactory.evaluation_logging.motion_diagnostics import MotionDiagnostics
            directory = cfg.gripper_log_dir or os.environ.get('VLARMORY_GRIPPER_LOG_DIR')
            trace = MotionDiagnostics(directory)
        elif getattr(cfg, 'enable_logs', False) and getattr(cfg, 'log_dir', 'logs'):
            from lerobot_robot_ufactory.evaluation_logging.eef_action_log import EEFActionLog
            action_log = EEFActionLog(getattr(cfg, 'log_dir', 'logs'))
        robot = robot_factory(cfg)
        if diagnostic:
            robot.connect(defer_motion=True, from_current_pose=True)
        else:
            robot.connect(defer_motion=True)
        record_video = getattr(cfg, "record_video", False) or os.environ.get("VLARMORY_RECORD_VIDEO") == "1"
        if record_video and not diagnostic:
            from lerobot_robot_ufactory.datasets.evaluation_video import start_evaluation_video
            recording = start_evaluation_video(robot, cfg)
        if robot._local_kinematics is None:
            raise RuntimeError("Recording FK was not initialized")
        logging.info("阶段：机械臂和相机连接完成，耗时 %.2f 秒", time.monotonic() - stage_start)
        logging.info(
            "Mode=%s, control_space=%s, fps=%s, execution horizon=%s. r + Enter: next episode; q + Enter or Ctrl-C: stop.",
            "DRY RUN" if cfg.dry_run else "EXECUTE",
            cfg.robot.get("control_space", "joint"),
            cfg.fps,
            cfg.steps_per_inference,
        )
        for episode in range(cfg.n_episodes):
            if poll_exit():
                break
            if not cfg.dry_run:
                stage_start = time.monotonic()
                if diagnostic:
                    if poll_exit():
                        break
                    robot.prepare_policy_from_current()
                    logging.info("闭环动作诊断：从当前臂位同时执行机械臂与夹爪，不预先闭合或复位")
                else:
                    logging.info("阶段：复位机械臂（第 %d/%d 条）", episode + 1, cfg.n_episodes)
                    robot.reset_to_initial(cancel_check=poll_exit)
                    logging.info("阶段：复位完成，耗时 %.2f 秒", time.monotonic() - stage_start)
            if diagnostic and record_video:
                from lerobot_robot_ufactory.datasets.evaluation_video import start_evaluation_video
                recording = start_evaluation_video(robot, cfg)
            (te if te is not None else policy).reset()
            logging.info("Episode %d/%d", episode + 1, cfg.n_episodes)
            actions = None
            offset = 0
            reference = None
            chunk = 0
            inference_ms = 0.
            observation_elapsed_s = None
            if trace:
                obs = observe()
                trace.begin([obs[f'{robot.prefix}J{i}.pos'] for i in range(1, 8)],
                            measured_state(robot, obs, cfg), cfg)
            for step in range(cfg.max_episode_steps):
                command = command_reader()
                if command == "q":
                    exit_requested = True
                if exit_requested or command == "r":
                    break
                if te is not None:
                    obs = observe()
                    if reference is None:
                        reference = observation_reference(obs)
                    tick_start = time.monotonic()
                    action = te.get_action(observation_builder(robot, obs, cfg))
                    command = command_reader()
                    if command in {"r", "q"}:
                        exit_requested = command == "q"
                        break
                elif actions is None or offset >= cfg.steps_per_inference:
                    obs = observe()
                    if trace:
                        observation_elapsed_s = trace.elapsed()
                        input_state = measured_state(robot, obs, cfg)
                    reference = observation_reference(obs)
                    start = time.monotonic()
                    actions = np.asarray(policy.predict_action(observation_builder(robot, obs, cfg))["actions"])
                    if actions.shape != (policy.chunk_size, 10) or not np.isfinite(actions).all():
                        raise ValueError("Policy returned invalid canonical EEF actions")
                    logging.info(
                        "Episode=%d step=%d inference=%.1f ms", episode + 1, step, (time.monotonic() - start) * 1000
                    )
                    offset = 0
                    chunk += 1
                    inference_ms = (time.monotonic() - start) * 1000
                    if trace:
                        trace.prediction(actions, chunk=chunk, step=step,
                                         observation_elapsed_s=observation_elapsed_s,
                                         inference_ms=inference_ms, state=input_state)
                    # Discard a completed inference when reset/exit arrived while waiting.
                    command = command_reader()
                    if command in {"r", "q"}:
                        exit_requested = command == "q"
                        break
                start = tick_start if te is not None else time.monotonic()
                if trace:
                    # Telemetry uses RT cache only: preserve the original RGB /
                    # tactile acquisition cadence and avoid added controller RPCs.
                    feedback = robot.get_motion_observation()
                    feedback_elapsed_s = feedback.get('_state_sample_monotonic_s', time.monotonic()) - trace.started
                    gripper_feedback_elapsed_s = feedback.get('_gripper_sample_monotonic_s', time.monotonic()) - trace.started
                if te is None:
                    action = actions[offset]
                command = (cartesian_action if cartesian else joint_action)(robot, action, reference, cfg)
                if action_log:
                    action_feedback = action_log.feedback(robot, cfg, measured_state)
                    command_time = time.monotonic()
                if trace:
                    command_elapsed_s = trace.elapsed()
                if cfg.dry_run:
                    sent = command
                else:
                    sent = robot.send_action(command)
                if action_log:
                    action_log.record(episode=episode, step=step, action=action, command=command,
                                      command_time=command_time, feedback=action_feedback, cfg=cfg,
                                      chunk=chunk, offset=offset, inference_ms=inference_ms, robot=robot)
                if trace:
                    trace.record(robot=robot, state=measured_state(robot, feedback, cfg),
                                 joints=[feedback[f'{robot.prefix}J{i}.pos'] for i in range(1, 8)],
                                 action=action, feedback_elapsed_s=feedback_elapsed_s,
                                 gripper_feedback_elapsed_s=gripper_feedback_elapsed_s,
                                 command_elapsed_s=command_elapsed_s,
                                 observation_elapsed_s=observation_elapsed_s, step=step,
                                 chunk=chunk, offset=offset, inference_ms=inference_ms, dry_run=cfg.dry_run)
                reference = (eef_target(action) if cartesian else
                             np.asarray([sent[f"{robot.prefix}J{i}.pos"] for i in range(1, 8)]))
                offset += 1
                sleep(max(0, 1 / cfg.fps - (time.monotonic() - start)))
            if not cfg.dry_run:
                if trace:
                    trace.event('policy_loop_end', reason='operator_stop' if exit_requested else 'step_limit')
                robot.pause_motion()
            if exit_requested:
                break
    except InterruptedError:
        if trace:
            exit_requested = True
            trace.event('cancelled')
        elif not exit_requested:
            raise
    except BaseException as exc:
        if trace:
            trace.failure(exc)
        raise
    finally:
        try:
            # Cleanup also covers camera/connection failure and KeyboardInterrupt.
            if robot is not None:
                try:
                    if robot.is_connected and not cfg.dry_run:
                        robot.pause_motion()
                finally:
                    try:
                        try:
                            if recording is not None:
                                recording.close()
                        finally:
                            robot.disconnect()
                    finally:
                        if policy is not None:
                            (te if te is not None else policy).close()
            elif policy is not None:
                (te if te is not None else policy).close()
        finally:
            try:
                if trace:
                    trace.close(stopped=exit_requested)
            finally:
                if action_log:
                    action_log.close()
