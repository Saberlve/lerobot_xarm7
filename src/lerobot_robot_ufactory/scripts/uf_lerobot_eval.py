import sys
import argparse
import logging
import time
import math
from pathlib import Path
from dataclasses import asdict, dataclass
from pprint import pformat
import numpy as np
import lerobot_robot_ufactory # patch
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.utils import make_robot_action
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_STR
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.control_utils import (
    is_headless,
    predict_action,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_ufactory.utils.utils import init_keyboard_listener
from lerobot_robot_ufactory.utils.action_safety import ActionSafetyConfig, ActionSafetyGuard
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations


def continuous_rotvec(new_rv, prev_rv):
    """Keep rotvec in the same sign-hemisphere as prev to avoid ±π flips.
    When accumulated rotation crosses π, as_rotvec() can flip the axis sign
    (e.g. rx jumps 3.14 → -3.13), causing the robot to make a large motion.
    This re-maps the equivalent rotation to stay consistent with prev."""
    new_rv = np.asarray(new_rv, dtype=np.float64)
    prev_rv = np.asarray(prev_rv, dtype=np.float64)
    if np.dot(new_rv, prev_rv) < 0:
        angle = np.linalg.norm(new_rv)
        if angle > 1e-6:
            axis = new_rv / angle
            new_rv = -(2 * np.pi - angle) * axis
    return new_rv

def compute_relative_axis_angle(rot_prev, rot_curr):
    """
    计算两个轴角之间的相对旋转。
    逻辑: R_diff = R_prev.T @ R_curr
    返回: 相对轴角向量
    """
    # 1. 转为矩阵
    R_prev = Transformations.rxryrz_to_rotation_matrix(*rot_prev)
    R_curr = Transformations.rxryrz_to_rotation_matrix(*rot_curr)
    
    # 2. 计算相对旋转矩阵
    # R_delta 表示从 prev 坐标系到 curr 坐标系的旋转
    R_delta = R_prev.T @ R_curr
    
    # 3. 转回轴角
    return Transformations.rotation_matrix_to_rxryrz(R_delta)

def compute_target_axis_angle(rot_prev, rot_delta):
    """
    根据起始轴角和相对轴角计算目标轴角
    """
    R_prev = Transformations.rxryrz_to_rotation_matrix(*rot_prev)
    R_delta = Transformations.rxryrz_to_rotation_matrix(*rot_delta)
    R_curr = R_prev @ R_delta
    # R_curr = R_prev.apply(R_delta)
    return Transformations.rotation_matrix_to_rxryrz(R_curr)


@dataclass
class EvalConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: dict = None # no use
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    n_episodes: int = 50
    single_task: str | None = "pick_place"

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            policy_path = Path(policy_path).expanduser()
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Choose a policy to control the robot")
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def eval_loop(cfg: EvalConfig, relative=False, rx_continuous=False, safety_guard: ActionSafetyGuard | None = None):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if safety_guard is not None:
        safety_guard.log_config()

    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        dataset_metadata = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)
        dataset_features = dataset_metadata.features
        print("Loaded dataset metadata successfully.")
    except Exception:
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(
                    action=robot.action_features
                ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
                use_videos=cfg.dataset.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=cfg.dataset.video,
            ),
        )
        # Create empty dataset or load existing saved episodes
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        dataset_metadata = dataset.meta
        print("Created new dataset metadata successfully.")
    
    if cfg.dataset.fps != dataset_metadata.fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset_metadata.fps} != {cfg.dataset.fps}).")
        

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset_metadata)
    # policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        dataset_stats=dataset_metadata.stats
    )

    robot.connect()

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

    device = get_safe_torch_device(policy.config.device, log=True)
    sleep_time_s = 1 / dataset_metadata.fps

    print("\n********** Policy Eval Episode Loop Start **********")
    print(f'relative: {relative}')

    # with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
    while True:
        reset = getattr(robot, "reset_to_initial", None)
        if reset is None:
            reset = robot.configure
        reset()
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        obs = robot.get_observation()

        prev_robot_dict = {}
        prev_action_dict = {}

        if hasattr(cfg.robot, 'robots'):
            keys = cfg.robot.robots.keys()
        else:
            keys = ['']
        for key in keys:
            prefix = f'.{key}' if key else ''
            is_tcp = f'{prefix}pose.x' in obs and f'{prefix}pose.y' in obs and f'{prefix}pose.z' in obs and f'{prefix}pose.rx' in obs and f'{prefix}pose.ry' in obs and f'{prefix}pose.rz' in obs
            if is_tcp:
                pose = [obs[f'{prefix}pose.x'], obs[f'{prefix}pose.y'], obs[f'{prefix}pose.z'], obs[f'{prefix}pose.rx'], obs[f'{prefix}pose.ry'], obs[f'{prefix}pose.rz']]
            else:
                pose = []
            prev_robot_dict[key] = {'type': 1 if is_tcp else 0, 'pose': np.array(pose)}
            prev_action_dict[key] = {'type': 1 if is_tcp else 0, 'pose': np.array(pose)}

        safety_halted = False

        while True:
            start_loop_t = time.perf_counter()

            if events["reset"] or events["exit"]:
                events["reset"] = False
                print("\n********** Policy Eval Episode (Reset) **********")
                break

            # Safety halt: stop inference and stop sending actions,
            # wait for the operator to press the right arrow key to resume
            if safety_halted:
                precise_sleep(sleep_time_s)
                continue

            # Get robot observation
            obs = robot.get_observation()

            curr_robot_dict = {}
            for key in keys:
                prefix = f'.{key}' if key else ''
                is_tcp = f'{prefix}pose.x' in obs and f'{prefix}pose.y' in obs and f'{prefix}pose.z' in obs and f'{prefix}pose.rx' in obs and f'{prefix}pose.ry' in obs and f'{prefix}pose.rz' in obs
                if not is_tcp or prev_robot_dict[key]['type'] != 1:
                    curr_robot_dict[key] = {'type': 0, 'pose': []}
                    continue
                if rx_continuous and not relative and f'{prefix}pose.rx' in obs and obs[f'{prefix}pose.rx'] < 0:
                    obs[f'{prefix}pose.rx'] += 2 * np.pi
                prev_robot_pose = prev_robot_dict[key]['pose']
                curr_robot_pose = np.array([obs[f'{prefix}pose.x'], obs[f'{prefix}pose.y'], obs[f'{prefix}pose.z'], obs[f'{prefix}pose.rx'], obs[f'{prefix}pose.ry'], obs[f'{prefix}pose.rz']])
                curr_rot_normalized = continuous_rotvec(curr_robot_pose[3:6], prev_robot_pose[3:6])
                curr_robot_pose[3] = float(curr_rot_normalized[0])
                curr_robot_pose[4] = float(curr_rot_normalized[1])
                curr_robot_pose[5] = float(curr_rot_normalized[2])
                curr_robot_dict[key] = {'type': 1, 'pose': curr_robot_pose}

                if relative:
                    delta = compute_relative_axis_angle(prev_robot_pose[3:6], curr_robot_pose[3:6])
                    obs[f'{prefix}pose.x'] = curr_robot_pose[0] - prev_robot_pose[0]
                    obs[f'{prefix}pose.y'] = curr_robot_pose[1] - prev_robot_pose[1]
                    obs[f'{prefix}pose.z'] = curr_robot_pose[2] - prev_robot_pose[2]
                    obs[f'{prefix}pose.rx'] = delta[0]
                    obs[f'{prefix}pose.ry'] = delta[1]
                    obs[f'{prefix}pose.rz'] = delta[2]
                    prev_robot_dict[key]['pose'] = curr_robot_pose                 

            # Applies a pipeline to the raw robot observation, default is IdentityProcessor
            obs_processed = robot_observation_processor(obs)

            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=cfg.single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))

            for key in keys:
                prefix = f'.{key}' if key else ''
                if not curr_robot_dict[key]['type'] != 1 or prev_action_dict[key]['type'] != 1:
                    continue
                if relative:
                    rot_delta = np.array([robot_action_to_send[f'{prefix}pose.rx'], robot_action_to_send[f'{prefix}pose.ry'], robot_action_to_send[f'{prefix}pose.rz']])
                    prev_action_pose = prev_action_dict[key]['pose']
                    rot_curr = compute_target_axis_angle(prev_action_pose[3:6], rot_delta)
                    robot_action_to_send[f'{prefix}pose.x'] = prev_action_pose[0] + robot_action_to_send[f'{prefix}pose.x']
                    robot_action_to_send[f'{prefix}pose.y'] = prev_action_pose[1] + robot_action_to_send[f'{prefix}pose.y']
                    robot_action_to_send[f'{prefix}pose.z'] = prev_action_pose[2] + robot_action_to_send[f'{prefix}pose.z']
                    robot_action_to_send[f'{prefix}pose.rx'] = rot_curr[0]
                    robot_action_to_send[f'{prefix}pose.ry'] = rot_curr[1]
                    robot_action_to_send[f'{prefix}pose.rz'] = rot_curr[2]
                elif rx_continuous and f'{prefix}pose.rx' in robot_action_to_send and robot_action_to_send[f'{prefix}pose.rx'] > math.pi:
                    robot_action_to_send[f'{prefix}pose.rx'] -= 2 * np.pi

                # robot_action_to_send[f'{prefix}pose.z'] = max(robot_action_to_send[f'{prefix}pose.z'], 199)

                if relative:
                    curr_action_pose = np.array([
                        robot_action_to_send[f'{prefix}pose.x'], robot_action_to_send[f'{prefix}pose.y'], robot_action_to_send[f'{prefix}pose.z'],
                        robot_action_to_send[f'{prefix}pose.rx'], robot_action_to_send[f'{prefix}pose.ry'], robot_action_to_send[f'{prefix}pose.rz']
                    ])
                    prev_action_dict[key]['pose'] = curr_action_pose

            # Safety check: any violation triggers an e-stop; the action is
            # NOT sent and the loop waits for the operator to press right arrow
            if safety_guard is not None:
                violation = safety_guard.check(robot_action_to_send, curr_robot_dict, keys)
                if violation is not None:
                    safety_halted = True
                    logging.error(f"*** SAFETY HALT *** {violation}")
                    print(f"\n*** SAFETY HALT *** {violation}\nAction was NOT sent. Press right arrow (->) to reset and resume, ESC to exit.")
                    continue

            robot.send_action(robot_action_to_send)
        
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(sleep_time_s - dt_s)

        if events["exit"]:
            break

    print("\n********** Policy Eval Loop Exit **********")
    if not is_headless() and listener is not None:
        listener.stop()

@parser.wrap()
def get_cfg(cfg: EvalConfig) -> EvalConfig:
    return cfg

def main():
    parser = argparse.ArgumentParser(description='configuration args')
    parser.add_argument('--relative', action='store_true', help='is relative motion or not')
    parser.add_argument('--rx_continuous', action='store_true', help='rx continuous or not')
    parser.add_argument('--enable_safety', action='store_true', help='enable action safety guard (thresholds in ActionSafetyConfig)')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    # Action safety guard: tune thresholds in ActionSafetyConfig directly.
    # Any violation triggers an e-stop; press right arrow to resume.
    safety_guard = ActionSafetyGuard(ActionSafetyConfig(enabled=args.enable_safety))
    eval_loop(cfg, args.relative, args.rx_continuous, safety_guard)


if __name__ == "__main__":
    main()
