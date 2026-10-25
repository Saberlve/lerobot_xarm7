"""Real-robot evaluation against an external starVLA policy server.

Usage:
    # 1. Start the starVLA policy server (in the starVLA repo / environment):
    python deployment/model_server/server_policy.py \
        --ckpt_path <your_checkpoint_dir> --port 10093 --use_bf16

    # 2. Run this eval script on the robot machine:
    python scripts/uf_starvla_eval.py --config_path config/eval/xarm7_starvla_eval_config.yaml

One entry for ACT, PI05, N0 and FTP1. Set action_space=eef10 for EEF policies;
action_space=joint retains the 8D joint path. --check_config=true validates
local YAML only. EEF execution supports joint IK or direct Cartesian commands
according to robot.control_space.

The policy runs on the server; this script only streams observations
(configured robot state + camera image(s) + task text) over WebSocket and executes the
returned action chunk on the xArm.

Joint keyboard controls (same as uf_lerobot_eval):
    Right/Left arrow : reset current episode (robot returns to initial pose)
    Esc              : exit eval loop and disconnect

EEF controls: r + Enter starts the next episode; q + Enter exits.

RTC (real-time chunking): set `rtc_mode: prefix_pin` (PI0/PI05) or another
framework-supported mode in the config to overlap inference with execution —
the next chunk is generated in the background while the robot executes the
tail of the current one, and the server pins the new chunk's first
`inference_delay` steps to the old chunk's tail, removing the pause at chunk
boundaries. `rtc_mode: null` (or "none") keeps the default blocking loop.

Temporal ensembling (TE): set `temporal_ensemble: true` to re-infer in the
background every `steps_per_inference_te` steps and execute, at every control tick,
the exponentially-weighted average of all action chunks covering the current
step (ACT Algorithm 2, same weighting as lerobot's ACTTemporalEnsembler).
Mutually exclusive with `rtc_mode`; `steps_per_inference` is ignored.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from pprint import pformat

import numpy as np
import yaml

from eef_executor import EEFExecutionConfig, make_policy, run as run_eef_executor, validate_gripper_config
from starvla_client import RTCPolicyClientWrapper, TemporalEnsembleClientWrapper

@dataclass
class StarVLAEvalConfig:
    robot: dict
    deployment_profile: str | None = None
    # Robot state/action representation is independent of the model framework.
    action_space: str = "joint"
    execution_mode: str = "full"
    gripper_log_dir: str = ""
    dry_run: bool = True
    record_video: bool = False
    video_dir: str = ""
    check_config: bool = False
    max_episode_steps: int = 900
    max_joint_step_deg: float = 90
    ik_position_tolerance_mm: float = 2
    ik_rotation_tolerance_deg: float = 2
    # Direct Cartesian execution: maximum change from feedback/last target.
    max_tcp_step_mm: float = 75
    max_tcp_rotation_step_deg: float = 45
    # starVLA policy server address (server binds 0.0.0.0; use the server IP here)
    server_host: str = "127.0.0.1"
    server_port: int = 10093
    # Control frequency for streaming actions to the robot.
    fps: float | None = None
    # Execute the first N steps of each predicted action chunk, then re-infer.
    # N=1 means fully closed-loop (re-infer every step). Chunk length comes
    # from the checkpoint's action_horizon.
    steps_per_inference: int | None = None
    # Expected checkpoint chunk length; None accepts the server's chunk length.
    action_chunk_size: int | None = None
    # --- RTC (real-time chunking) ---
    # RTC mode: None / "none" disables (default blocking chunk loop). Any other
    # value enables RTC and is forwarded to the framework as `mode`:
    # "prefix_pin" for PI0/PI05 (their only mode)
    rtc_mode: str | None = None
    # Steps the robot keeps executing from the old chunk while the next
    # inference runs
    inference_delay: int = 8
    # --- Temporal ensembling (TE) ---
    # Mutually exclusive with rtc_mode. When enabled, a background thread
    # re-infers every `steps_per_inference_te` steps and every control tick executes
    # the exponentially-weighted average of all chunks covering the current
    # step (ACT Algorithm 2; mirrors lerobot's ACTTemporalEnsembler).
    # `steps_per_inference` is ignored in this mode.
    temporal_ensemble: bool = False
    # Issue a new inference every N executed steps; with a 40-step chunk,
    # N=10 keeps ~4 chunks overlapping in the ensemble.
    steps_per_inference_te: int | None = None
    # Exponential weighting coefficient (lerobot/ACT naming + default):
    # positive favors older chunks, 0 = uniform average, negative favors
    # newer chunks.
    temporal_ensemble_coeff: float = 0.01
    # True: average the gripper dim like every other dim (lerobot behavior).
    # False: always take the newest chunk's gripper value (avoids
    # intermediate openings from averaging a 0/1-style command).
    te_ensemble_gripper: bool | None = None
    single_task: str | None = None
    n_episodes: int | None = None
    # Keys of the cameras in the robot observation dict.
    camera_keys: list[str] | None = None
    tactile_keys: list[str] | None = None
    # Master switch for action logs (EEF CSV + SVG; joint steps + seams). When false,
    # no log files are created regardless of log_dir.
    enable_logs: bool = False
    # Directory for state/action logs at the configured control frequency
    # (only used when enable_logs is true).
    log_dir: str = "logs"
    # Top-level key in dataset_statistics.json ("" = auto-pick when single).
    state_stats_key: str = ""

    def __post_init__(self):
        if self.execution_mode not in ("full", "continue_move"):
            raise ValueError("execution_mode must be full or continue_move")
        if self.execution_mode == "continue_move" and self.action_space != "eef10":
            raise ValueError("continue_move requires an EEF10 model")
        if type(self.record_video) is not bool:
            raise ValueError("record_video must be boolean")
        # Preserve each former entry's defaults when a config omits a field.
        defaults = dict(fps=30, steps_per_inference=25, steps_per_inference_te=10,
                        te_ensemble_gripper=True, n_episodes=50, camera_keys=["camera"],
                        tactile_keys=[], single_task="Pick up the white bar and drop it in the bag.")
        if self.action_space == "eef10" and self.deployment_profile in (
            "xarm7_eef_n0vtla", "xarm7_eef_ftp1",
        ):
            defaults.update(fps=15, steps_per_inference=(32 if self.deployment_profile == "xarm7_eef_ftp1" else 50),
                            steps_per_inference_te=5, te_ensemble_gripper=False, n_episodes=1,
                            camera_keys=["third_camera", "wrist_camera"],
                            tactile_keys=["photon_left", "photon_right"],
                            single_task="Pick up the egg and place it in the green egg tray.")
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        if self.action_space not in ("eef10", "joint"):
            raise ValueError("action_space must be eef10 or joint")
        if self.action_space == "eef10" and isinstance(self.robot, dict):
            _validate_eef_config(self)


def load_config(path: str | Path) -> StarVLAEvalConfig:
    with Path(path).open() as stream:
        return StarVLAEvalConfig(**yaml.safe_load(stream))


def _validate_eef_config(cfg):
    if cfg.execution_mode == "continue_move":
        validate_gripper_config(cfg)
    profile = cfg.deployment_profile
    if profile not in (None, "xarm7_eef10", "xarm7_eef_n0vtla", "xarm7_eef_ftp1"):
        raise ValueError("Unsupported EEF deployment_profile")
    if (cfg.rtc_mode or "none").lower() != "none":
        raise ValueError("EEF execution does not support RTC")
    if profile in ("xarm7_eef_n0vtla", "xarm7_eef_ftp1"):
        EEFExecutionConfig(**{item.name: getattr(cfg, item.name) for item in fields(EEFExecutionConfig)})
        if profile == "xarm7_eef_ftp1" and cfg.temporal_ensemble:
            raise ValueError("FTP1 runner currently uses blocking chunks; disable temporal_ensemble")
        return
    if cfg.temporal_ensemble:
        raise ValueError("These imported EEF10 checkpoints use ordinary chunk inference")
    robot = cfg.robot
    control_space = robot.get("control_space", "joint")
    if (robot.get("type"), robot.get("robot_dof")) != ("uf::robot", 7) or (
        (control_space, robot.get("record_space", "joint")) not in (("joint", "both"), ("cartesian", "joint"))
    ):
        raise ValueError("EEF10 requires xArm7 with joint control/record_space=both or cartesian control/record_space=joint")
    if robot.get("manual_mode", False) or robot.get("min_tcp_z_mm") is None:
        raise ValueError("EEF10 execution requires the existing height guard and policy control mode")
    if len(cfg.camera_keys) != 2 or len(set(cfg.camera_keys)) != 2 or not set(cfg.camera_keys) <= robot.get("cameras", {}).keys():
        raise ValueError("Provide external then wrist RGB camera keys")
    limits = (("max_tcp_step_mm", "max_tcp_rotation_step_deg") if control_space == "cartesian" else
              ("max_joint_step_deg", "ik_position_tolerance_mm", "ik_rotation_tolerance_deg"))
    for name in ("fps", *limits):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("steps_per_inference", "n_episodes", "max_episode_steps"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(cfg.dry_run, bool) or not cfg.single_task.strip():
        raise ValueError("Set a boolean dry_run mode and the training task instruction")


def _eef_config(cfg):
    if not isinstance(cfg.robot, dict):
        import draccus
        robot = draccus.encode(cfg.robot)
        robot["type"] = cfg.robot.type
        for key, camera in cfg.robot.cameras.items():
            robot["cameras"][key]["type"] = camera.type
        cfg = replace(cfg, robot=robot)
    _validate_eef_config(cfg)
    return cfg


def _joint_config(cfg):
    # Defer robot SDKs and LeRobot until the joint path actually needs them.
    global init_logging, make_robot_from_config, is_headless, precise_sleep, init_keyboard_listener
    import draccus
    import lerobot_robot_ufactory  # noqa: F401
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.robots import RobotConfig, make_robot_from_config
    from lerobot.utils.control_utils import is_headless
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging
    from lerobot_robot_ufactory.utils.utils import init_keyboard_listener
    register_third_party_plugins()
    robot = draccus.decode(RobotConfig, cfg.robot) if isinstance(cfg.robot, dict) else cfg.robot
    return replace(cfg, robot=robot)



def _build_state(obs: dict, gripper_cmd: float | None) -> np.ndarray:
    """Raw joint radians and commanded gripper; normalization belongs to server."""
    gripper = obs["gripper.pos"] if gripper_cmd is None else gripper_cmd
    return np.asarray([obs[f"J{i}.pos"] for i in range(1, 8)] + [gripper], dtype=np.float32)


def _joint_observation(obs, gripper_cmd, cfg):
    result = dict(state=_build_state(obs, gripper_cmd), lang=cfg.single_task,
                  image=[np.asarray(obs[key]).copy() for key in cfg.camera_keys])
    if cfg.tactile_keys:
        result["tactile"] = [np.asarray(obs[key]).copy() for key in cfg.tactile_keys]
    return result


def _build_action_dict(action: np.ndarray) -> dict:
    """Map one (8,) action row to the robot action dict.

    Actions are already denormalized by the server: absolute joint positions
    (rad) + gripper in [0, 1] (0=open, 1=close), matching robot conventions.
    """
    action_dict = {f"J{i + 1}.pos": float(action[i]) for i in range(7)}
    action_dict["gripper.pos"] = float(action[7])
    return action_dict


def eval_loop(cfg: StarVLAEvalConfig):
    representation = getattr(cfg, "action_space", "joint")
    if representation == "eef10":
        return run_eef_executor(_eef_config(cfg))
    if representation != "joint":
        raise ValueError(f"Unknown action_space: {representation}")
    cfg = _joint_config(cfg)
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    client = make_policy(cfg)
    # Echoed so the operator can manually verify action_chunk_size, image
    # size/count etc. against the training setup before running episodes.
    server_meta = client.transport.get_server_metadata()
    logging.info(f"starVLA server metadata: {pformat(server_meta)}")
    if cfg.action_chunk_size is not None and (
        type(cfg.action_chunk_size) is not int or cfg.action_chunk_size < 1
        or server_meta.get("action_chunk_size") != cfg.action_chunk_size
    ):
        raise ValueError(f"Configured action_chunk_size={cfg.action_chunk_size!r} differs from server metadata {server_meta.get('action_chunk_size')!r}")

    rtc = None
    rtc_mode = (cfg.rtc_mode or "").strip().lower()
    if rtc_mode and rtc_mode != "none":
        if not 0 < cfg.inference_delay < cfg.steps_per_inference:
            raise ValueError(
                f"inference_delay ({cfg.inference_delay}) must be in "
                f"(0, steps_per_inference={cfg.steps_per_inference}) for RTC."
            )
        if not server_meta.get("rtc_supported", False):
            logging.warning(
                "rtc_mode=%r but the server reports rtc_supported=False — "
                "chunks will NOT be prefix-conditioned; the loop still runs "
                "(background prefetch + splice), but seams may jerk.",
                cfg.rtc_mode,
            )
        rtc = RTCPolicyClientWrapper(
            client.transport,
            inference_delay=cfg.inference_delay,
            execution_horizon=cfg.steps_per_inference,
            mode=cfg.rtc_mode,
        )
        logging.info(
            f"RTC enabled: mode={cfg.rtc_mode} inference_delay={cfg.inference_delay} "
            f"execution_horizon={cfg.steps_per_inference} (fps={cfg.fps})"
        )

    te = None
    if cfg.temporal_ensemble:
        if rtc is not None:
            raise ValueError(
                "temporal_ensemble and rtc_mode are mutually exclusive: TE does "
                "its own cross-chunk averaging, so RTC prefix pinning is "
                "redundant. Set rtc_mode to null/none."
            )
        action_horizon = server_meta.get("action_chunk_size")
        if not isinstance(action_horizon, int) or action_horizon < 1:
            raise ValueError(
                f"temporal_ensemble needs a positive integer action_chunk_size "
                f"in the server metadata, got {action_horizon!r}."
            )
        if cfg.steps_per_inference_te >= action_horizon:
            raise ValueError(
                f"steps_per_inference_te ({cfg.steps_per_inference_te}) must be smaller than "
                f"the action chunk size ({action_horizon}) so chunks overlap."
            )
        te = TemporalEnsembleClientWrapper(
            client.transport,
            query_period=cfg.steps_per_inference_te,
            action_horizon=action_horizon,
            coeff=cfg.temporal_ensemble_coeff,
            ensemble_gripper=cfg.te_ensemble_gripper,
        )
        logging.info(
            f"Temporal ensembling enabled: steps_per_inference_te={cfg.steps_per_inference_te} "
            f"coeff={cfg.temporal_ensemble_coeff} horizon={action_horizon} "
            f"ensemble_gripper={cfg.te_ensemble_gripper} (fps={cfg.fps}; "
            f"steps_per_inference is ignored)"
        )

    events = {"reset": False, "exit": False}
    listener = None

    if not is_headless():
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    print("Right arrow key pressed. Resetting...")
                    events["reset"] = True
                elif key == keyboard.Key.left:
                    print("Left arrow key pressed. Resetting....")
                    events["reset"] = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping ...")
                    events["exit"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        listener, events = init_keyboard_listener(events=events, on_press=on_press)

    sleep_time_s = 1 / cfg.fps

    print("\n********** starVLA Policy Eval Episode Loop Start **********")

    action_log = None
    if cfg.enable_logs and cfg.log_dir:
        from lerobot_robot_ufactory.evaluation_logging.joint_action_log import JointActionLog
        action_log = JointActionLog(cfg.log_dir)
    recording = None
    if cfg.record_video or os.environ.get("VLARMORY_RECORD_VIDEO") == "1":
        from lerobot_robot_ufactory.datasets.evaluation_video import start_evaluation_video
        recording = start_evaluation_video(robot, cfg)

    try:
        episode = 0
        while episode < cfg.n_episodes and not events["exit"]:
            print(f"\n********** Episode {episode + 1}/{cfg.n_episodes} **********")
            reset = getattr(robot, "reset_to_initial", None)
            if reset is None:
                reset = robot.configure
            reset()
            events["reset"] = False
            # Gripper state fed to the policy: last command actually sent by
            # send_action (mirrors the training state). None until the first
            # command, in which case the measured position is used once.
            gripper_cmd = None
            current_actions = None
            current_step = 0
            chunk_index = -1
            last_action = None
            if rtc is not None:
                rtc.reset()  # drop server-side prev chunk + local chunk state
            if te is not None:
                te.reset()  # drop ensemble buffer + server-side policy state

            while True:
                if events["reset"] or events["exit"]:
                    events["reset"] = False
                    print("\n********** starVLA Policy Eval Episode (Reset) **********")
                    break

                start_loop_t = time.perf_counter()
                if rtc is not None:
                    # RTC: build a fresh observation every tick; the wrapper
                    # fires the (background) inference at the right offset and
                    # returns the aligned action for this tick.
                    obs = robot.get_observation()
                    policy_observation = _joint_observation(obs, gripper_cmd, cfg)
                    query = {"examples": [policy_observation],
                             "unnorm_key": client.unnorm_key}
                    prev_chunk_index = rtc.chunk_index
                    action = rtc.get_action(query)
                    current_step = rtc.current_step_in_chunk + 1
                    if rtc.chunk_index != prev_chunk_index:
                        chunk_index = rtc.chunk_index
                        if action_log:
                            action_log.record_seam(episode=episode, chunk=chunk_index,
                                                   inference_ms=rtc.last_inference_ms,
                                                   previous=last_action, action=action)
                    last_action = action
                elif te is not None:
                    # TE: fresh observation every tick; the wrapper re-infers
                    # in the background every steps_per_inference_te steps and returns
                    # the temporally-ensembled action for this tick.
                    obs = robot.get_observation()
                    policy_observation = _joint_observation(obs, gripper_cmd, cfg)
                    query = {"examples": [policy_observation],
                             "unnorm_key": client.unnorm_key}
                    prev_chunk_index = te.chunk_index
                    action = te.get_action(query)
                    current_step += 1
                    # Seam diagnostic: one row per newly merged chunk, with the
                    # jump between consecutive ensembled actions.
                    if te.chunk_index != prev_chunk_index:
                        chunk_index = te.chunk_index
                        if action_log:
                            action_log.record_seam(episode=episode, chunk=chunk_index,
                                                   inference_ms=te.last_inference_ms,
                                                   previous=last_action, action=action)
                    last_action = action
                elif current_actions is None or current_step >= min(
                    cfg.steps_per_inference, len(current_actions)
                ):
                    # Get robot observation
                    obs = robot.get_observation()
                    policy_observation = _joint_observation(obs, gripper_cmd, cfg)

                    # NOTE: inference is blocking (one flow-matching pass can take
                    # several hundred ms) and no actions are sent while waiting.
                    # Mode 6 holds the last commanded online-trajectory target, so
                    # the arm simply pauses between action chunks.
                    infer_start_t = time.perf_counter()
                    resp = client.predict_action(
                        policy_observation
                    )
                    inference_ms = (time.perf_counter() - infer_start_t) * 1000
                    new_actions = np.asarray(resp["actions"])  # (T, 8), denormalized

                    # Seam diagnostic: how far the new chunk's first action
                    # jumps from the last action actually executed.
                    if action_log and current_actions is not None and current_step > 0:
                        action_log.record_seam(episode=episode, chunk=chunk_index + 1,
                                               inference_ms=inference_ms,
                                               previous=current_actions[current_step - 1], action=new_actions[0])
                    current_actions = new_actions
                    current_step = 0
                    chunk_index += 1

                if rtc is None and te is None:
                    start_loop_t = time.perf_counter()
                    action = current_actions[current_step]
                    current_step += 1

                robot.send_action(_build_action_dict(action))

                # After send_action, _last_gripper_command holds the
                # command that passed the robot-side threshold/rate
                # coalescing -- exactly what GELLO recordings store as the
                # gripper state. Fall back to the raw model output only
                # before the first command is ever sent.
                sent_gripper = getattr(robot, "_last_gripper_command", None)
                if sent_gripper is not None:
                    gripper_cmd = sent_gripper
                elif gripper_cmd is None:
                    gripper_cmd = float(action[7])

                if action_log:
                    action_log.record_step(robot=robot, episode=episode, chunk=chunk_index,
                                           step=current_step - 1, gripper=gripper_cmd, action=action)

                dt_s = time.perf_counter() - start_loop_t
                precise_sleep(max(sleep_time_s - dt_s, 0.0))

            episode += 1

    finally:
        print("\n********** starVLA Policy Eval Loop Exit **********")
        if recording is not None:
            recording.close()
        if action_log:
            action_log.close()
        client.close()
        if robot.is_connected:
            robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()


def get_cfg() -> StarVLAEvalConfig:
    from lerobot.configs import parser

    @parser.wrap()
    def parse(cfg: StarVLAEvalConfig) -> StarVLAEvalConfig:
        return cfg

    return parse()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = get_cfg()
    if cfg.check_config:
        if cfg.action_space == "eef10":
            from eef_executor import decode_robot_config
            decode_robot_config(_eef_config(cfg))
        else:
            _joint_config(cfg)
        print("StarVLA local config valid; no model, camera or robot constructed")
        return
    try:
        eval_loop(cfg)
    except KeyboardInterrupt:
        if cfg.action_space != "eef10" or cfg.deployment_profile not in (
            "xarm7_eef_n0vtla", "xarm7_eef_ftp1",
        ):
            raise
        logging.info("Stopped")


if __name__ == "__main__":
    main()
