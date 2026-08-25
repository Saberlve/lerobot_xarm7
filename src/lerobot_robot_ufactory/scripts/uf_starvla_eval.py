"""Real-robot evaluation against an external starVLA policy server.

Usage:
    # 1. Start the starVLA policy server (in the starVLA repo / environment):
    python deployment/model_server/server_policy.py \
        --ckpt_path <your_checkpoint_dir> --port 10093 --use_bf16

    # 2. Run this eval script on the robot machine:
    uf-starvla-eval --config_path config/eval/xarm7_starvla_eval_config.yaml

The policy runs on the server; this script only streams observations
(joint state + camera image(s) + task text) over WebSocket and executes the
returned action chunk on the xArm.

Keyboard controls (same as uf_lerobot_eval):
    Right/Left arrow : reset current episode (robot returns to initial pose)
    Esc              : exit eval loop and disconnect

RTC (real-time chunking): set `rtc_mode: prefix_pin` (PI0/PI05) or another
framework-supported mode in the config to overlap inference with execution —
the next chunk is generated in the background while the robot executes the
tail of the current one, and the server pins the new chunk's first
`inference_delay` steps to the old chunk's tail, removing the pause at chunk
boundaries. `rtc_mode: null` (or "none") keeps the default blocking loop.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import numpy as np

import lerobot_robot_ufactory  # patch: registers uf:: robot/camera types
# Register camera config subclasses ("opencv", "intelrealsense") so draccus
# can decode the robot.cameras section; uf_lerobot_eval gets these transitively
# via lerobot.scripts.lerobot_record, which this script does not import.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.control_utils import is_headless
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from lerobot_robot_ufactory.utils.starvla_ws_client import RTCPolicyClientWrapper, WebsocketClientPolicy
from lerobot_robot_ufactory.utils.utils import init_keyboard_listener


@dataclass
class StarVLAEvalConfig:
    robot: RobotConfig
    # starVLA policy server address (server binds 0.0.0.0; use the server IP here)
    server_host: str = "127.0.0.1"
    server_port: int = 10093
    # Control frequency for streaming actions to the robot.
    fps: int = 30
    # Execute the first N steps of each predicted action chunk, then re-infer.
    # N=1 means fully closed-loop (re-infer every step). Chunk length comes
    # from the checkpoint (action_horizon=40 for the xarm7 pi05 runs).
    steps_per_inference: int = 25
    # --- RTC (real-time chunking) ---
    # RTC mode: None / "none" disables (default blocking chunk loop). Any other
    # value enables RTC and is forwarded to the framework as `mode`:
    # "prefix_pin" for PI0/PI05 (their only mode)
    rtc_mode: str | None = None
    # Steps the robot keeps executing from the old chunk while the next
    # inference runs
    inference_delay: int = 8
    single_task: str = "Pick up the white bar and drop it in the bag."
    n_episodes: int = 50
    # Keys of the cameras in the robot observation dict.
    camera_keys: list[str] = field(default_factory=lambda: ["camera"])
    # Master switch for the diagnostic CSV logs (steps + seams). When false,
    # no log files are created regardless of log_dir.
    enable_logs: bool = True
    # Directory for 30 Hz state/action diagnostic logs and per-seam diffs
    # (only used when enable_logs is true).
    log_dir: str = "logs"
    # Path to the training-time dataset_statistics.json. REQUIRED: the policy
    # was trained on q99-normalized state (see the checkpoint's DataConfig),
    # and the server does NOT normalize state -- the client must.
    dataset_statistics_path: str = ""
    # Top-level key in dataset_statistics.json ("" = auto-pick when single).
    state_stats_key: str = ""


def _build_state(
    obs: dict,
    gripper_cmd: float | None,
    state_q01: np.ndarray | None = None,
    state_q99: np.ndarray | None = None,
) -> np.ndarray:
    """8-dim proprio state: 7 joint positions (rad) + gripper (0=open, 1=close).

    GELLO recordings store the last *commanded* gripper value in the state
    (see get_realtime_observation), not the measured position. Feeding the
    measured position here would expose the policy to slow gripper ramps it
    never saw in training, so the commanded value is preferred.

    The policy was trained on q99-normalized state (training DataConfig:
    ``state.joint_positions: "q99"``) and the server does NOT normalize state,
    so the raw values are normalized here exactly like the training transform:
    ``2 * (x - q01) / (q99 - q01) - 1``, clamped to [-2.2, 2.2].
    """
    gripper = obs["gripper.pos"] if gripper_cmd is None else gripper_cmd
    raw = np.array([obs[f"J{i}.pos"] for i in range(1, 8)] + [gripper], dtype=np.float32)
    if state_q01 is None:
        return raw
    mask = state_q01 != state_q99
    normalized = raw.copy()
    normalized[mask] = 2 * (raw[mask] - state_q01[mask]) / (state_q99[mask] - state_q01[mask]) - 1
    return np.clip(normalized, -2.2, 2.2)


def _load_state_norm_stats(path: str, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load state q01/q99 arrays from the training-time dataset_statistics.json."""
    if not path:
        logging.warning(
            "dataset_statistics_path not set -- sending RAW state to the policy. "
            "The model was trained on q99-normalized state; this is almost "
            "certainly wrong."
        )
        return None
    with open(path) as f:
        stats = json.load(f)
    if not key:
        if len(stats) != 1:
            raise ValueError(
                f"Multiple keys in {path}: {list(stats.keys())}. Set state_stats_key."
            )
        key = next(iter(stats))
    state_stats = stats[key]["state"]
    q01 = np.asarray(state_stats["q01"], dtype=np.float32)
    q99 = np.asarray(state_stats["q99"], dtype=np.float32)
    if q01.shape != (8,) or q99.shape != (8,):
        raise ValueError(f"Expected 8-dim state stats, got {q01.shape} / {q99.shape}")
    logging.info(f"State q99 normalization loaded from {path} (key={key})")
    return q01, q99


def _build_action_dict(action: np.ndarray) -> dict:
    """Map one (8,) action row to the robot action dict.

    Actions are already denormalized by the server: absolute joint positions
    (rad) + gripper in [0, 1] (0=open, 1=close), matching robot conventions.
    """
    action_dict = {f"J{i + 1}.pos": float(action[i]) for i in range(7)}
    action_dict["gripper.pos"] = float(action[7])
    return action_dict


def _rt_joint_state(robot) -> list[float] | None:
    """Latest joint positions from the 250 Hz RT report cache (non-blocking).

    This is the same feedback source GELLO recordings use for the state, so
    logging it at 30 Hz does not perturb the control loop. Returns None while
    the report thread has not delivered the first packet.
    """
    if not getattr(robot, "_rt_report_normal", False):
        return None
    with robot._update_lock:
        return list(robot.rt_actual_joint_pos)


def _open_diagnostic_logs(log_dir: str):
    """Open per-tick and per-seam diagnostic CSVs.

    steps CSV: one row per 30 Hz control tick (RT state + sent action).
    seams CSV: one row per re-inference (new_chunk[0] - last_executed_action).
    """
    if not log_dir:
        return None, None, lambda: None
    import csv
    from pathlib import Path

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    step_file = (log_path / f"starvla_eval_{stamp}_steps.csv").open("w", newline="", buffering=1)
    seam_file = (log_path / f"starvla_eval_{stamp}_seams.csv").open("w", newline="", buffering=1)
    step_writer = csv.writer(step_file)
    seam_writer = csv.writer(seam_file)
    step_writer.writerow(
        ["t_s", "episode", "chunk", "step_in_chunk"]
        + [f"s{i}" for i in range(1, 8)]
        + ["sg"]
        + [f"a{i}" for i in range(1, 8)]
        + ["ag"]
    )
    seam_writer.writerow(
        ["t_s", "episode", "chunk", "inference_ms"]
        + [f"d{i}" for i in range(1, 8)]
        + ["dg"]
        + [f"last{i}" for i in range(1, 8)]
        + ["last_g"]
        + [f"new{i}" for i in range(1, 8)]
        + ["new_g"]
    )
    print(f"Diagnostic logs: {step_file.name}, {seam_file.name}")

    def _close():
        step_file.close()
        seam_file.close()

    return step_writer, seam_writer, _close


def eval_loop(cfg: StarVLAEvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    norm_stats = _load_state_norm_stats(cfg.dataset_statistics_path, cfg.state_stats_key)
    state_q01, state_q99 = norm_stats if norm_stats is not None else (None, None)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    client = WebsocketClientPolicy(cfg.server_host, cfg.server_port)
    # Echoed so the operator can manually verify action_chunk_size, image
    # size/count etc. against the training setup before running episodes.
    server_meta = client.get_server_metadata()
    logging.info(f"starVLA server metadata: {pformat(server_meta)}")

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
            client,
            inference_delay=cfg.inference_delay,
            execution_horizon=cfg.steps_per_inference,
            mode=cfg.rtc_mode,
        )
        logging.info(
            f"RTC enabled: mode={cfg.rtc_mode} inference_delay={cfg.inference_delay} "
            f"execution_horizon={cfg.steps_per_inference} (fps={cfg.fps})"
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

    step_writer, seam_writer, close_logs = _open_diagnostic_logs(
        cfg.log_dir if cfg.enable_logs else ""
    )
    run_start_t = time.perf_counter()

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

            while True:
                if events["reset"] or events["exit"]:
                    events["reset"] = False
                    print("\n********** starVLA Policy Eval Episode (Reset) **********")
                    break

                if rtc is not None:
                    # RTC: build a fresh observation every tick; the wrapper
                    # fires the (background) inference at the right offset and
                    # returns the aligned action for this tick.
                    obs = robot.get_observation()
                    state = _build_state(obs, gripper_cmd, state_q01, state_q99)
                    images = [obs[key] for key in cfg.camera_keys]
                    query = {"examples": [{"image": images, "lang": cfg.single_task, "state": state}]}
                    prev_chunk_index = rtc.chunk_index
                    action = rtc.get_action(query)
                    current_step = rtc.current_step_in_chunk + 1
                    if rtc.chunk_index != prev_chunk_index:
                        chunk_index = rtc.chunk_index
                        if seam_writer is not None and last_action is not None:
                            seam_writer.writerow(
                                [f"{time.perf_counter() - run_start_t:.6f}", episode, chunk_index, f"{rtc.last_inference_ms:.1f}"]
                                + [f"{v:.6f}" for v in (action - last_action)]
                                + [f"{v:.6f}" for v in last_action]
                                + [f"{v:.6f}" for v in action]
                            )
                    last_action = action
                    start_loop_t = time.perf_counter()
                elif current_actions is None or current_step >= min(
                    cfg.steps_per_inference, len(current_actions)
                ):
                    # Get robot observation
                    obs = robot.get_observation()
                    state = _build_state(obs, gripper_cmd, state_q01, state_q99)
                    images = [obs[key] for key in cfg.camera_keys]  # uint8 HWC RGB, in training order

                    # NOTE: inference is blocking (one flow-matching pass can take
                    # several hundred ms) and no actions are sent while waiting.
                    # Mode 6 holds the last commanded online-trajectory target, so
                    # the arm simply pauses between action chunks.
                    infer_start_t = time.perf_counter()
                    resp = client.predict_action(
                        {"examples": [{"image": images, "lang": cfg.single_task, "state": state}]}
                    )
                    inference_ms = (time.perf_counter() - infer_start_t) * 1000
                    new_actions = np.asarray(resp["data"]["actions"][0])  # (T, 8), denormalized

                    # Seam diagnostic: how far the new chunk's first action
                    # jumps from the last action actually executed.
                    if seam_writer is not None and current_actions is not None and current_step > 0:
                        last_executed = current_actions[current_step - 1]
                        seam_writer.writerow(
                            [f"{time.perf_counter() - run_start_t:.6f}", episode, chunk_index + 1, f"{inference_ms:.1f}"]
                            + [f"{v:.6f}" for v in (new_actions[0] - last_executed)]
                            + [f"{v:.6f}" for v in last_executed]
                            + [f"{v:.6f}" for v in new_actions[0]]
                        )
                    current_actions = new_actions
                    current_step = 0
                    chunk_index += 1

                if rtc is None:
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

                # 30 Hz state/action row. State comes from the RT report
                # cache (same source as GELLO recordings), so this does not
                # add any blocking controller reads to the control loop.
                if step_writer is not None:
                    joints = _rt_joint_state(robot) or [float("nan")] * 7
                    step_writer.writerow(
                        [f"{time.perf_counter() - run_start_t:.6f}", episode, chunk_index, current_step - 1]
                        + [f"{v:.6f}" for v in joints]
                        + ["" if gripper_cmd is None else f"{gripper_cmd:.4f}"]
                        + [f"{v:.6f}" for v in action]
                    )

                dt_s = time.perf_counter() - start_loop_t
                precise_sleep(sleep_time_s - dt_s)

            episode += 1

    finally:
        print("\n********** starVLA Policy Eval Loop Exit **********")
        close_logs()
        client.close()
        if robot.is_connected:
            robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()


@parser.wrap()
def get_cfg(cfg: StarVLAEvalConfig) -> StarVLAEvalConfig:
    return cfg


def main():
    register_third_party_plugins()
    cfg = get_cfg()
    eval_loop(cfg)


if __name__ == "__main__":
    main()
