import sys
import csv
import copy
import time
import queue
import argparse
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
import lerobot_robot_ufactory # patch
from lerobot.scripts.lerobot_record import *
from lerobot.scripts.lerobot_record import RecordConfig as LeRobotRecordConfig
from lerobot_robot_ufactory.teleoperators.uf_mock_teleop import UFMockTeleop
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop
from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController
from lerobot_robot_ufactory.utils.utils import init_keyboard_listener


@dataclass
class UFRecordConfig(LeRobotRecordConfig):
    """RecordConfig variant that permits UFACTORY manual-mode recording."""

    def __post_init__(self):
        manual_mode = getattr(self.robot, "manual_mode", False)
        if manual_mode:
            if self.teleop is not None or self.policy is not None:
                raise ValueError("manual_mode recording cannot be combined with a teleop or policy")
            return
        super().__post_init__()


def _get_dataset_writer(dataset):
    return getattr(dataset, "writer", None)


def _get_episode_buffer(dataset):
    try:
        return dataset.episode_buffer
    except AttributeError:
        pass

    writer = _get_dataset_writer(dataset)
    if writer is not None and hasattr(writer, "episode_buffer"):
        return writer.episode_buffer
    raise RuntimeError("Unable to access dataset episode buffer for async save.")


def _set_episode_buffer(dataset, episode_buffer):
    updated = False
    writer = _get_dataset_writer(dataset)
    if writer is not None and hasattr(writer, "episode_buffer"):
        writer.episode_buffer = episode_buffer
        updated = True

    try:
        getattr(dataset, "episode_buffer")
    except AttributeError:
        pass
    else:
        try:
            dataset.episode_buffer = episode_buffer
            updated = True
        except AttributeError:
            pass

    if not updated:
        raise RuntimeError("Unable to replace dataset episode buffer for async save.")


def _to_int(value):
    if isinstance(value, (list, tuple)):
        return int(value[0])
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(value[0])


def _episode_buffer_size(episode_buffer):
    return _to_int(episode_buffer.get("size", 0))


def _episode_buffer_index(episode_buffer):
    return _to_int(episode_buffer["episode_index"])


def _current_episode_index(dataset):
    try:
        return _episode_buffer_index(_get_episode_buffer(dataset))
    except Exception:
        return dataset.num_episodes


def _manual_gripper_action_key(action_features):
    return next((key for key in action_features if key.endswith("gripper.pos")), None)


def _diagnostic_logs_enabled(robot) -> bool:
    """Return whether optional per-cycle diagnostics are enabled for a robot."""
    config = getattr(robot, "config", None)
    if config is not None and hasattr(config, "enable_logs"):
        return bool(config.enable_logs)

    child_robots = getattr(robot, "robots", None)
    if child_robots:
        return any(_diagnostic_logs_enabled(child) for child in child_robots.values())
    return False


def _manual_action_from_observation(observation, action_features, gripper_target=None):
    """Keep only robot action fields when mirroring manual-mode state."""
    action = {key: value for key, value in observation.items() if key in action_features}
    if gripper_target is not None:
        gripper_key = _manual_gripper_action_key(action_features)
        if gripper_key is not None and gripper_key in action:
            action[gripper_key] = float(gripper_target)
    return action


def _update_manual_gripper_key_state(key, pressed, key_state):
    char = getattr(key, "char", None)
    if not isinstance(char, str):
        return

    char = char.lower()
    if char == "c":
        key_state["close"] = pressed
    elif char == "o":
        key_state["open"] = pressed


def _update_manual_gripper_target(target, key_state, speed, fps):
    if target is None or fps <= 0:
        return target

    close_pressed = bool(key_state.get("close", False))
    open_pressed = bool(key_state.get("open", False))
    if close_pressed == open_pressed:
        return target

    direction = 1.0 if close_pressed else -1.0
    return min(max(target + direction * speed / fps, 0.0), 1.0)


def _create_empty_episode_buffer(dataset, episode_index, template_episode_buffer):
    writer = _get_dataset_writer(dataset)

    if writer is not None and hasattr(writer, "_create_episode_buffer"):
        episode_buffer = writer._create_episode_buffer()
    elif hasattr(dataset, "create_episode_buffer"):
        episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)
    elif hasattr(dataset, "_create_episode_buffer"):
        episode_buffer = dataset._create_episode_buffer()
    else:
        episode_buffer = copy.deepcopy(template_episode_buffer)
        for key, value in list(episode_buffer.items()):
            if key == "size":
                episode_buffer[key] = 0
            elif key == "episode_index":
                continue
            elif isinstance(value, list):
                episode_buffer[key] = []
            else:
                episode_buffer[key] = []

    if _episode_buffer_index(episode_buffer) != episode_index:
        episode_buffer["episode_index"] = episode_index
    return episode_buffer


def _create_next_episode_buffer(dataset, current_episode_buffer):
    current_episode_index = _episode_buffer_index(current_episode_buffer)
    return _create_empty_episode_buffer(dataset, current_episode_index + 1, current_episode_buffer)


class AsyncEpisodeSaver:
    _STOP = object()

    def __init__(self, dataset):
        self.dataset = dataset
        self._queue = queue.Queue()
        self._total_cnts = 0
        self._finish_cnts = 0
        self._exception = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="uf-async-episode-saver", daemon=True)
        self._thread.start()

    def submit_current_episode(self):
        self._raise_if_failed()
        episode_buffer = _get_episode_buffer(self.dataset)
        if _episode_buffer_size(episode_buffer) == 0:
            raise RuntimeError("Cannot async save an empty episode buffer.")

        episode_index = _episode_buffer_index(episode_buffer)
        next_episode_buffer = _create_next_episode_buffer(self.dataset, episode_buffer)
        _set_episode_buffer(self.dataset, next_episode_buffer)
        self._queue.put((episode_index, episode_buffer))
        return episode_index

    def wait_idle(self):
        self._queue.join()
        self._raise_if_failed()

    def close(self):
        if self._closed:
            return
        self._queue.join()
        self._queue.put(self._STOP)
        self._queue.join()
        self._thread.join()
        self._closed = True
        self._raise_if_failed()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                episode_index, episode_buffer = item
                print(f'[Async] saving episode {episode_index}')
                try:
                    self.dataset.save_episode(episode_data=episode_buffer)
                except TypeError as exc:
                    if "episode_data" in str(exc):
                        raise RuntimeError(
                            "--async-save requires LeRobotDataset.save_episode(episode_data=...)."
                        ) from exc
                    raise
                self._delete_saved_image_dirs(episode_index)
                print(f'[Async] save episode {episode_index} finish')
            except BaseException as exc:
                self._exception = exc
                print(f'[Async] episode {episode_index} save failed, {exc}')
            finally:
                self._queue.task_done()

    def _delete_saved_image_dirs(self, episode_index):
        writer = _get_dataset_writer(self.dataset)
        meta = getattr(writer, "_meta", getattr(self.dataset, "meta", None))
        image_keys = getattr(meta, "image_keys", [])
        image_dir_owner = writer if writer is not None and hasattr(writer, "_get_image_file_dir") else self.dataset
        if not image_keys or not hasattr(image_dir_owner, "_get_image_file_dir"):
            return

        for cam_key in image_keys:
            img_dir = image_dir_owner._get_image_file_dir(episode_index, cam_key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async episode save failed.") from self._exception


def _disconnect_recording_resources(robot, teleop, listener):
    """Release recording devices while preserving cleanup after partial failures."""
    try:
        if getattr(robot, "_is_connected", False) or getattr(robot, "real_arm", None) is not None:
            robot.disconnect()
    finally:
        try:
            if teleop is not None and getattr(teleop, "is_connected", False):
                teleop.disconnect()
        finally:
            if listener is not None:
                listener.stop()


class _RecordingCleanup:
    def __init__(self, robot, teleop, listener, async_episode_saver):
        self.robot = robot
        self.teleop = teleop
        self.listener = listener
        self.async_episode_saver = async_episode_saver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.async_episode_saver is not None:
                self.async_episode_saver.close()
        finally:
            _disconnect_recording_resources(self.robot, self.teleop, self.listener)
        return False
    

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    frame_callback: callable = None,
    manual_mode: bool = False,
    manual_gripper_keys: dict[str, bool] | None = None,
    manual_gripper_speed: float = 0.5,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    last_robot_cmd = robot.get_observation()
    # only positional cmd for now: Remove velo from observation for cmd if needed!
    last_robot_cmd = { k: v for k,v in last_robot_cmd.items() if not "vel" in k }

    manual_gripper_keys = manual_gripper_keys or {}
    manual_gripper_target = None
    manual_gripper_action_key = _manual_gripper_action_key(robot.action_features)

    realtime_controller = None
    diagnostic_logs_enabled = _diagnostic_logs_enabled(robot)
    sync_log_file = None
    sync_log_writer = None
    sync_frame_index = 0
    if (
        policy is None
        and isinstance(teleop, UFBaseTeleop)
        and getattr(robot, "_control_space", None) == "joint"
        and hasattr(robot, "get_realtime_observation")
    ):
        realtime_controller = RealtimeTeleopController(
            robot=robot,
            teleop=teleop,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            fps=int(teleop.config.realtime_control_fps),
            initial_observation=last_robot_cmd,
        )
        realtime_controller.start()
        if diagnostic_logs_enabled:
            sync_log_dir = Path("logs")
            sync_log_dir.mkdir(parents=True, exist_ok=True)
            sync_log_path = sync_log_dir / (
                f"gello_record_sync_{time.strftime('%Y%m%d_%H%M%S')}_"
                f"{time.time_ns() % 1_000_000:06d}.csv"
            )
            sync_log_file = sync_log_path.open("w", newline="", buffering=1)
            sync_log_writer = csv.DictWriter(
                sync_log_file,
                fieldnames=[
                    "frame",
                    "state_sample_s",
                    "action_sent_s",
                    "action_age_ms",
                    "observation_end_s",
                    "state_to_observation_end_ms",
                    "camera_timings",
                    "frame_loop_ms",
                ],
            )
            sync_log_writer.writeheader()
            logging.info("Realtime dataset synchronization log: %s", sync_log_path)

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        if realtime_controller is not None:
            obs = robot.get_realtime_observation()
            observation_monotonic_s = getattr(robot, "_last_realtime_observation_monotonic_s", None)
            if observation_monotonic_s is None:
                observation_monotonic_s = time.perf_counter()
            realtime_controller.update_observation(obs)
            if diagnostic_logs_enabled:
                matched_action, matched_action_sent_s = realtime_controller.action_sample_at(
                    observation_monotonic_s
                )
            else:
                matched_action = realtime_controller.action_at(observation_monotonic_s)
        else:
            obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        elif policy is None and manual_mode:
            # In manual mode the physical arm is the source of both the
            # observation and the demonstrated target state.
            if manual_gripper_action_key is not None and manual_gripper_target is None:
                gripper_value = obs_processed.get(manual_gripper_action_key)
                if gripper_value is None:
                    gripper_value = obs.get(manual_gripper_action_key)
                if gripper_value is not None:
                    manual_gripper_target = min(max(float(gripper_value), 0.0), 1.0)

            manual_gripper_target = _update_manual_gripper_target(
                manual_gripper_target,
                manual_gripper_keys,
                manual_gripper_speed,
                fps,
            )
            act = _manual_action_from_observation(
                obs_processed,
                robot.action_features,
                gripper_target=manual_gripper_target,
            )
            act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, Teleoperator):
            if realtime_controller is not None:
                act_processed_teleop = matched_action
                act = None
            else:
                act = teleop.get_action()

            # (space mouse) from delta Cartesian cmd to absolute command
            if act is not None and "pose.dx" in act:
                last_robot_cmd.update({"pose.x": last_robot_cmd["pose.x"] + act["pose.dx"], "pose.y": last_robot_cmd["pose.y"] + act["pose.dy"], "pose.z": last_robot_cmd["pose.z"] + act["pose.dz"]})
                act = last_robot_cmd.copy() # watch out this is shallow copy, not for nested dict

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            if realtime_controller is None:
                act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        if realtime_controller is None:
            _sent_action = robot.send_action(robot_action_to_send)
        else:
            _sent_action = matched_action
        # Robots may clamp or otherwise sanitize a command before sending it.
        # Store that effective command so demonstrations match the motion.
        if isinstance(_sent_action, dict):
            action_values = _sent_action

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            if frame_callback is not None:
                frame = frame_callback(frame)
            dataset.add_frame(frame)

        if sync_log_writer is not None:
            observation_end_s = getattr(
                robot, "_last_realtime_observation_end_monotonic_s", observation_monotonic_s
            )
            camera_timings = getattr(robot, "_last_realtime_camera_timings", {})
            sync_log_writer.writerow(
                {
                    "frame": sync_frame_index,
                    "state_sample_s": f"{observation_monotonic_s:.9f}",
                    "action_sent_s": f"{matched_action_sent_s:.9f}",
                    "action_age_ms": f"{(observation_monotonic_s - matched_action_sent_s) * 1000:.3f}",
                    "observation_end_s": f"{observation_end_s:.9f}",
                    "state_to_observation_end_ms": f"{(observation_end_s - observation_monotonic_s) * 1000:.3f}",
                    "camera_timings": repr(camera_timings),
                    "frame_loop_ms": f"{(time.perf_counter() - start_loop_t) * 1000:.3f}",
                }
            )
            sync_frame_index += 1

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t

    if realtime_controller is not None:
        realtime_controller.stop()
    if sync_log_file is not None:
        sync_log_file.close()


def _prepare_recording_episode(robot, teleop, is_uf_teleop, manual_mode):
    if is_uf_teleop:
        # Stop teleop output before handing control to the xArm reset motion.
        teleop.set_teleop_enabled(False)

    if is_uf_teleop or manual_mode:
        reset = getattr(robot, "reset_to_initial", None)
        if reset is None:
            reset = robot.configure
        reset()

    if is_uf_teleop:
        obs = robot.get_observation()
        teleop.set_teleop_enabled(True, obs)


def _print_record_controls(is_recorded, manual_mode):
    if is_recorded:
        controls = '[ESC] Exit  [←] Reset  [→] Save'
    else:
        start_label = 'Reset / Start' if manual_mode else 'Start'
        controls = f'[ESC] Exit  [Space] {start_label}  [←] Reset  [→] Save'
    if manual_mode:
        controls += '  [C] Close  [O] Open'
    print(f'⌨   {controls}')


def _ask_choice(prompt: str, options: dict[str, str]) -> str:
    """Prompt the user to pick one of the given options (lowercase keys)."""
    keys = "/".join(options)
    while True:
        print(f"\n{prompt}")
        for key, description in options.items():
            print(f"  [{key}] {description}")
        try:
            choice = input(f"Choose [{keys}]: ").strip().lower()
        except EOFError:
            print("No input available, cancelling.")
            raise SystemExit(1)
        if choice in options:
            return choice
        print(f"Invalid choice, please enter {keys}.")


def _missing_dataset_files(root: Path) -> list[str]:
    """Return the local files required before a dataset can be resumed."""
    missing = []
    for relative_path in ("meta/info.json", "meta/tasks.parquet"):
        if not (root / relative_path).is_file():
            missing.append(relative_path)
    if not any((root / "meta" / "episodes").glob("*/*.parquet")):
        missing.append("meta/episodes/*/*.parquet")
    if not any((root / "data").glob("*/*.parquet")):
        missing.append("data/*/*.parquet")
    return missing


def _prepare_dataset_root(cfg: UFRecordConfig) -> None:
    """Prepare an existing dataset root without pre-creating a new one."""
    root = Path(cfg.dataset.root)
    existed = root.exists()

    if not existed:
        if cfg.resume:
            raise RuntimeError(f"Cannot resume because the dataset directory does not exist: {root}")
        return

    missing = _missing_dataset_files(root)
    if missing:
        missing_text = ", ".join(missing)
        message = (
            f"Dataset directory is incomplete and cannot be resumed: {root}\n"
            f"Missing: {missing_text}"
        )
        if cfg.resume or not sys.stdin.isatty():
            raise RuntimeError(message)
        choice = _ask_choice(
            message,
            options={
                "o": "Overwrite: remove this directory and record a new dataset",
                "c": "Cancel",
            },
        )
        if choice == "o":
            shutil.rmtree(root)
        else:
            raise SystemExit("Recording cancelled.")
        return

    if cfg.resume:
        return

    # A valid LeRobot dataset already exists.
    if not sys.stdin.isatty():
        # Non-interactive run: keep the previous auto-resume behaviour.
        cfg.resume = True
        print(f"Existing dataset found, resuming recording (non-interactive): {root}")
        return

    choice = _ask_choice(
        f"Dataset directory already exists: {root}",
        options={
            "o": "Overwrite: delete the existing dataset and record a new one",
            "r": "Resume: keep existing episodes and continue recording",
            "c": "Cancel",
        },
    )
    if choice == "o":
        shutil.rmtree(root)
    elif choice == "r":
        cfg.resume = True
    else:
        raise SystemExit("Recording cancelled.")


def record(cfg: UFRecordConfig, async_save: bool = False) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    _prepare_dataset_root(cfg)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    manual_mode = bool(getattr(cfg.robot, "manual_mode", False))

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

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

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
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

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    try:
        robot.connect()
        if teleop is not None:
            teleop.connect()
    except BaseException:
        try:
            _disconnect_recording_resources(robot, teleop, None)
        except BaseException:
            logging.exception("Failed to clean up after recording device connection failure")
        raise

    is_evt = not is_headless()
    is_uf_teleop = isinstance(teleop, UFBaseTeleop)
    is_recorded = False
    key_dict = {}
    manual_gripper_keys = {"close": False, "open": False}
    listener = None
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    if is_evt:
        from pynput import keyboard

        key_dict = {
            keyboard.Key.space: 0,  # start
            keyboard.Key.enter: 0,  # help
        }

        def on_press(key):
            _update_manual_gripper_key_state(key, True, manual_gripper_keys)
            try:
                if key == keyboard.Key.right:
                    print("Right arrow key pressed. Exiting loop...")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")
            if key in key_dict:
                key_dict[key] = True

        def on_release(key):
            _update_manual_gripper_key_state(key, False, manual_gripper_keys)
            try:
                if key == keyboard.Key.enter:
                    _print_record_controls(is_recorded, manual_mode)
                    # is_recorded = True
            except Exception as e:
                print(f"Error handling key release: {e}")
            if key in key_dict:
                key_dict[key] = False

        listener, events = init_keyboard_listener(events=events, on_press=on_press, on_release=on_release)
        print("\n********** Episode Record Loop Start **********")
        _print_record_controls(is_recorded, manual_mode)
    else:
        input('⌨   Press Enter to start record >>> ')
        is_recorded = True
        print('\n********** Episode Record Loop Start **********')

    frame_callback = None
    async_episode_saver = AsyncEpisodeSaver(dataset) if async_save else None
    if async_episode_saver is not None:
        print('Async episode saving is enabled.')

    with _RecordingCleanup(robot, teleop, listener, async_episode_saver), VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            time.sleep(0.01)
            if is_evt:
                if not is_recorded and key_dict[keyboard.Key.space]:
                    is_recorded = True

            if teleop is not None and isinstance(teleop, UFMockTeleop):
                if events["stop_recording"]:
                    continue
                teleop.configure(events=events)
                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    input('\n⌨   Press Enter to regenerate random target location >>>>> ')
                    continue
                if events["stop_recording"]:
                    continue
                is_recorded = True

            if is_recorded:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if is_uf_teleop or manual_mode:
                    _prepare_recording_episode(robot, teleop, is_uf_teleop, manual_mode)
                log_say(f"Recording episode {_current_episode_index(dataset)}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    frame_callback=frame_callback,
                    manual_mode=manual_mode,
                    manual_gripper_keys=manual_gripper_keys,
                    manual_gripper_speed=getattr(cfg.robot, "manual_gripper_speed", 0.5),
                )
            else:
                continue
            if events['stop_recording']:
                break
            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if is_uf_teleop:
                    teleop.set_teleop_enabled(False)
                episode_buffer = _get_episode_buffer(dataset)
                if _episode_buffer_size(episode_buffer) > 0:
                    if async_episode_saver is None:
                        dataset.clear_episode_buffer()
                    else:
                        episode_index = _episode_buffer_index(episode_buffer)
                        empty_episode_buffer = _create_empty_episode_buffer(dataset, episode_index, episode_buffer)
                        _set_episode_buffer(dataset, empty_episode_buffer)
                is_recorded = False
                if is_evt:
                    _print_record_controls(is_recorded, manual_mode)
                else:
                    input('\n⌨   Press Enter to rerecord this episode >>>>> ')
                    is_recorded = True
                continue

            if is_recorded and not events['stop_recording']:
                episode_index = _current_episode_index(dataset)
                log_say(f"Save episode {episode_index}", cfg.play_sounds)
                if is_uf_teleop:
                    teleop.set_teleop_enabled(False)
                if async_episode_saver is None:
                    dataset.save_episode()
                    log_say(f"[Finish] Save episode {episode_index}", cfg.play_sounds)
                else:
                    queued_episode_index = async_episode_saver.submit_current_episode()
                    if queued_episode_index is not None:
                        log_say(f"[Queued] Save episode {queued_episode_index}", cfg.play_sounds)

                recorded_episodes += 1
                is_recorded = False
                if is_evt:
                    _print_record_controls(is_recorded, manual_mode)
                else:
                    input('⌨   Press Enter to record at the next episode >>>>> ')
                    is_recorded = True

        if async_episode_saver is not None:
            print('Waiting for pending async episode saves.')
            async_episode_saver.close()

    print("\n********** Episode Record Loop Exit **********")

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

@parser.wrap()
def get_cfg(cfg: UFRecordConfig) -> UFRecordConfig:
    return cfg

def main():
    parser = argparse.ArgumentParser(description='configuration args')
    parser.add_argument('-r',
                       action='store_true', # specify --resume if resume needs to be True
                       default=False,
                       help='Whether contitue recording on existing dataset (default: False)')
    parser.add_argument('-a', '--async_save',
                       action='store_true',
                       default=False,
                       help='Enable async background saving (default: False)')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    if args.r:
        cfg.resume = True
    cfg.play_sounds = False
    record(cfg, async_save=args.async_save)


if __name__ == "__main__":
    main()
