"""Real-robot evaluation against an external starVLA policy server.

Usage:
    # 1. Start the starVLA policy server (in the starVLA repo / environment):
    python deployment/model_server/server_policy.py \
        --ckpt_path <your_checkpoint_dir> --port 10093 --use_bf16

    # 2. Run this eval script on the robot machine:
    uf-starvla-eval --config_path config/eval/xarm7_starvla_eval_config.yaml

The policy runs on the server; this script only streams observations
(joint state + one camera image + task text) over WebSocket and executes the
returned action chunk on the xArm.

Keyboard controls (same as uf_lerobot_eval):
    Right/Left arrow : reset current episode (robot returns to initial pose)
    Esc              : exit eval loop and disconnect
"""

import logging
import time
from dataclasses import asdict, dataclass
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

from lerobot_robot_ufactory.utils.starvla_ws_client import WebsocketClientPolicy
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
    # N=1 means fully closed-loop (re-infer every step). Server chunks are T=50.
    steps_per_inference: int = 25
    single_task: str = "Pick up the black bottle and place it on the blue bag"
    n_episodes: int = 50
    # Key of the camera in the robot observation dict (camera name in robot config).
    camera_key: str = "camera"


def _build_state(obs: dict) -> np.ndarray:
    """8-dim proprio state: 7 joint positions (rad) + gripper (0=open, 1=close)."""
    return np.array([obs[f"J{i}.pos"] for i in range(1, 8)] + [obs["gripper.pos"]], dtype=np.float32)


def _build_action_dict(action: np.ndarray) -> dict:
    """Map one (8,) action row to the robot action dict.

    Actions are already denormalized by the server: absolute joint positions
    (rad) + gripper in [0, 1] (0=open, 1=close), matching robot conventions.
    """
    action_dict = {f"J{i + 1}.pos": float(action[i]) for i in range(7)}
    action_dict["gripper.pos"] = float(action[7])
    return action_dict


def eval_loop(cfg: StarVLAEvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    client = WebsocketClientPolicy(cfg.server_host, cfg.server_port)
    # Echoed so the operator can manually verify action_chunk_size, image
    # size/count etc. against the training setup before running episodes.
    logging.info(f"starVLA server metadata: {pformat(client.get_server_metadata())}")

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

    try:
        episode = 0
        while episode < cfg.n_episodes and not events["exit"]:
            print(f"\n********** Episode {episode + 1}/{cfg.n_episodes} **********")
            reset = getattr(robot, "reset_to_initial", None)
            if reset is None:
                reset = robot.configure
            reset()
            events["reset"] = False

            while True:
                if events["reset"] or events["exit"]:
                    events["reset"] = False
                    print("\n********** starVLA Policy Eval Episode (Reset) **********")
                    break

                # Get robot observation
                obs = robot.get_observation()
                state = _build_state(obs)
                image = obs[cfg.camera_key]  # uint8 HWC RGB

                # NOTE: inference is blocking (one flow-matching pass can take
                # several hundred ms) and no actions are sent while waiting.
                # This is acceptable here: xArm ServoJ holds the last commanded
                # position, so the arm simply pauses between action chunks.
                resp = client.predict_action(
                    {"examples": [{"image": [image], "lang": cfg.single_task, "state": state}]}
                )
                actions = np.asarray(resp["data"]["actions"][0])  # (T, 8), denormalized

                # Execute the first N steps of the chunk, then re-infer.
                for action in actions[: cfg.steps_per_inference]:
                    if events["reset"] or events["exit"]:
                        break

                    start_loop_t = time.perf_counter()
                    action_dict = _build_action_dict(action)

                    robot.send_action(action_dict)

                    dt_s = time.perf_counter() - start_loop_t
                    precise_sleep(sleep_time_s - dt_s)

            episode += 1

    finally:
        print("\n********** starVLA Policy Eval Loop Exit **********")
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
