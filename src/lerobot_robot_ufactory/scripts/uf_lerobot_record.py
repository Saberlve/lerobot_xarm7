import sys
import csv
import copy
import time
import queue
import argparse
import logging
import shutil
import threading
import os
import json
from uuid import uuid4
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
from inspect import signature
import lerobot_robot_ufactory # patch
from lerobot.scripts.lerobot_record import *
from lerobot.scripts.lerobot_record import RecordConfig as LeRobotRecordConfig
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot_robot_ufactory.teleoperators.uf_mock_teleop import UFMockTeleop
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop
from lerobot_robot_ufactory.utils.realtime_teleop import (
    RealtimeTeleopController,
    apply_keyboard_gripper_stop,
    apply_pending_gello_joint_mode,
    update_gello_joint_mode_key,
)
from lerobot_robot_ufactory.utils.recording_lock import exclusive_recording
from lerobot_robot_ufactory.utils.utils import init_keyboard_listener
from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview, WebPreviewConfig
from lerobot_robot_ufactory.utils.episode_images import discard_episode_images, validate_episode_images
from lerobot_robot_ufactory.utils.raw_episodes import (
    RawEpisodeStore, open_recording_dataset, postprocess_raw_episodes, recover_postprocessing,
)


@dataclass
class UFRecordConfig(LeRobotRecordConfig):
    """RecordConfig variant that permits UFACTORY manual-mode recording."""

    web_preview: WebPreviewConfig = field(default_factory=WebPreviewConfig)
    # Keep timing sidecars out of the training schema while making diagnostics
    # available by default for GELLO recording.
    synchronize: bool = True
    offline_mesh3dflow: bool = False

    def __post_init__(self):
        self.web_preview.validate()
        manual_mode = getattr(self.robot, "manual_mode", False)
        if manual_mode:
            if self.teleop is not None or self.policy is not None:
                raise ValueError("manual_mode recording cannot be combined with a teleop or policy")
            return
        super().__post_init__()


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, object], prefix: str
) -> dict[str, np.ndarray]:
    """Build a dataset frame, including fixed-shape tactile tensors.

    LeRobot's stock helper handles vector states and images. Photon
    ``Marker3DFlow`` is a float32 ``(rows, cols, 3)`` field, so it needs to
    pass through as an explicit tensor instead of being classified as an
    image solely because it has three dimensions.
    """
    frame = {}
    for key, feature in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        if feature["dtype"] in ("image", "video"):
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
            continue
        if feature["dtype"] not in ("float32", "float64"):
            continue

        raw_key = key.removeprefix(f"{prefix}.")
        if feature.get("names") is not None and len(feature["shape"]) == 1:
            value = np.array([values[name] for name in feature["names"]], dtype=feature["dtype"])
        else:
            value = np.asarray(values[raw_key], dtype=feature["dtype"])
            if value.ndim == 0 and tuple(feature["shape"]) == (1,):
                value = value.reshape(1)
        if tuple(value.shape) != tuple(feature["shape"]):
            raise ValueError(
                f"Feature '{key}' has shape {value.shape}; expected {feature['shape']}"
            )
        frame[key] = value
    return frame


def _dataset_robot_type(robot) -> str:
    """Use the concrete xArm model name in the dataset metadata."""
    robot_dof = getattr(getattr(robot, "config", None), "robot_dof", None)
    if isinstance(robot_dof, int) and robot_dof in (5, 6, 7):
        return f"xarm{robot_dof}"
    return robot.name


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


class EpisodeSynchronization:
    """Per-episode timing sidecar, intentionally outside LeRobot features."""

    def __init__(
        self,
        controller: RealtimeTeleopController | None,
        fps: int,
        *,
        dataset_root: Path | None = None,
        episode_index: int | None = None,
        tactile_stream_names: tuple[str, ...] = (),
    ):
        self.controller = controller
        self.fps = fps
        self.frames: list[dict] = []
        self.action_rows: list[dict] = []
        self.tactile_recorder = None
        if tactile_stream_names:
            if dataset_root is None or episode_index is None:
                raise ValueError(
                    "dataset_root and episode_index are required for tactile streams"
                )
            from lerobot_robot_ufactory.tactile.persistence import TactileStreamRecorder

            self.tactile_recorder = TactileStreamRecorder(
                dataset_root,
                episode_index,
                tactile_stream_names,
            )

    def add_frame(
        self,
        frame_index: int,
        state_sample_s: float,
        state_rt_receive_s: float | None,
        action_sent_s: float | None = None,
        camera_timing: dict | None = None,
        state_age_ms: float | None = None,
        *,
        action_send_start_s: float | None = None,
        action_send_end_s: float | None = None,
        action_index: int | None = None,
        state_sample_ns: int | None = None,
        state_rt_receive_ns: int | None = None,
        action_send_start_ns: int | None = None,
        action_send_end_ns: int | None = None,
        tactile_window_start_s: float | None = None,
        tactile_window_start_ns: int | None = None,
        tactile_samples: dict[str, tuple] | None = None,
    ) -> None:
        # ``action_sent_s`` was the historical send-end field. Accept it for
        # callers outside this repository while making send-start primary.
        if action_send_start_s is None:
            if action_sent_s is None:
                raise ValueError("action_send_start_s is required")
            action_send_start_s = action_sent_s
        if action_send_end_s is None:
            action_send_end_s = (
                action_send_start_s if action_sent_s is None else action_sent_s
            )
        if not all(
            np.isfinite(value)
            for value in (state_sample_s, action_send_start_s, action_send_end_s)
        ):
            raise ValueError("state/action timestamps must be finite")
        if action_send_end_s < action_send_start_s:
            raise ValueError("action send-end cannot precede send-start")
        action_send_start_ns = int(
            action_send_start_ns
            if action_send_start_ns is not None
            else round(action_send_start_s * 1_000_000_000)
        )
        action_send_end_ns = int(
            action_send_end_ns
            if action_send_end_ns is not None
            else round(action_send_end_s * 1_000_000_000)
        )
        state_sample_ns = int(
            state_sample_ns
            if state_sample_ns is not None
            else round(state_sample_s * 1_000_000_000)
        )
        state_timestamp_s = (
            state_rt_receive_s if state_rt_receive_s is not None else state_sample_s
        )
        state_timestamp_ns = int(
            state_rt_receive_ns
            if state_rt_receive_ns is not None
            else (
                state_sample_ns
                if state_rt_receive_s is None
                else round(state_rt_receive_s * 1_000_000_000)
            )
        )
        if state_timestamp_s > action_send_start_s:
            raise AssertionError(
                f"Future state selected: {state_timestamp_s:.9f} > "
                f"action {action_send_start_s:.9f}"
            )

        camera_timing = camera_timing or {}
        camera_timestamps = {}
        for key, timing in camera_timing.items():
            camera_timestamp = {
                "frame_index": timing.get("frame_index"),
                "read_start_ns": (
                    None
                    if timing.get("read_start_s") is None
                    else round(timing["read_start_s"] * 1_000_000_000)
                ),
                "read_end_ns": (
                    None
                    if timing.get("read_end_s") is None
                    else round(timing["read_end_s"] * 1_000_000_000)
                ),
            }
            if "capture_monotonic_s" in timing:
                capture_s = timing["capture_monotonic_s"]
                if capture_s > action_send_start_s:
                    raise AssertionError(
                        f"Future camera sample selected for {key}: "
                        f"{capture_s:.9f} > action {action_send_start_s:.9f}"
                    )
                camera_timestamp["capture_monotonic_ns"] = int(
                    timing.get("capture_monotonic_ns")
                    if timing.get("capture_monotonic_ns") is not None
                    else round(capture_s * 1_000_000_000)
                )
                camera_timestamp["age_to_action_ms"] = (
                    action_send_start_s - capture_s
                ) * 1_000
            if "sensor_timestamp_s" in timing:
                camera_timestamp["sensor_timestamp_s"] = timing["sensor_timestamp_s"]
            if "device_to_host_offset_s" in timing:
                camera_timestamp["device_to_host_offset_s"] = timing[
                    "device_to_host_offset_s"
                ]
            if "sync_offset_ms" in timing:
                camera_timestamp["sync_target_monotonic_ns"] = int(
                    timing.get("sync_target_monotonic_ns")
                    if timing.get("sync_target_monotonic_ns") is not None
                    else round(timing["sync_target_monotonic_s"] * 1_000_000_000)
                )
                camera_timestamp["sync_offset_ms"] = timing["sync_offset_ms"]
                camera_timestamp["sync_signed_offset_ms"] = timing.get("sync_signed_offset_ms")
                camera_timestamp["pair_skew_ms"] = timing.get("pair_skew_ms")
            camera_timestamps[key] = camera_timestamp

        tactile_timestamps = {}
        tactile_samples = tactile_samples or {}
        if self.tactile_recorder is not None:
            if tactile_window_start_s is None:
                raise ValueError("tactile_window_start_s is required for tactile streams")
            for name in self.tactile_recorder.stream_names:
                tactile_timestamps[name] = self.tactile_recorder.add_window(
                    name,
                    tactile_samples.get(name, ()),
                    tactile_window_start_s,
                    action_send_start_s,
                    start_monotonic_ns=tactile_window_start_ns,
                    end_monotonic_ns=action_send_start_ns,
                )
            for camera_name, timing in camera_timing.items():
                stream_name = timing.get("tactile_stream_name")
                if stream_name not in tactile_timestamps:
                    continue
                capture_s = timing.get("capture_monotonic_s")
                if capture_s is None:
                    continue
                capture_ns = int(
                    timing.get("capture_monotonic_ns")
                    if timing.get("capture_monotonic_ns") is not None
                    else round(capture_s * 1_000_000_000)
                )
                tactile_timestamps[stream_name][
                    "representative_capture_monotonic_ns"
                ] = capture_ns
                tactile_timestamps[stream_name][
                    "representative_tactile_index"
                ] = self.tactile_recorder.representative_index(stream_name, capture_ns)

        state_age_to_action_ms = (
            action_send_start_s - state_timestamp_s
        ) * 1_000
        self.frames.append(
            {
                "frame_index": frame_index,
                "action_index": action_index,
                "state_sample_ns": state_sample_ns,
                "state_rt_receive_ns": (
                    None
                    if state_rt_receive_s is None
                    else state_timestamp_ns
                ),
                "state_timestamp_ns": state_timestamp_ns,
                "action_send_start_ns": action_send_start_ns,
                "action_send_end_ns": action_send_end_ns,
                "action_send_latency_ms": (
                    action_send_end_s - action_send_start_s
                ) * 1_000,
                # Kept for readers of the older sidecar; now positive age of
                # the selected state relative to action send-start.
                "action_state_age_ms": state_age_to_action_ms,
                "state_age_ms": state_age_to_action_ms,
                "state_read_age_ms": state_age_ms,
                "camera_timing_json": json.dumps(camera_timestamps, sort_keys=True),
                "tactile_timing_json": json.dumps(tactile_timestamps, sort_keys=True),
            }
        )
        if self.controller is None:
            self.action_rows.append(
                {
                    "action_index": action_index,
                    "action_send_start_ns": action_send_start_ns,
                    "action_send_end_ns": action_send_end_ns,
                    "command_send_start_ns": action_send_start_ns,
                    "command_send_end_ns": action_send_end_ns,
                    "sent_at_ns": action_send_end_ns,
                    "send_latency_ns": action_send_end_ns - action_send_start_ns,
                }
            )

    def write(
        self,
        dataset_root: Path,
        episode_index: int,
        *,
        defer_commit: bool = False,
        extra_items: list[tuple[Path, Path]] | None = None,
    ) -> None:
        """Prepare then transactionally publish tactile streams and their sidecars."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_dir = dataset_root / "timestamps"
        base = f"episode_{episode_index:06d}"
        action_rows = (
            self.controller.action_timings()
            if self.controller is not None
            else self.action_rows
        )
        if self.tactile_recorder is not None:
            self.tactile_recorder.prepare(episode_index)
            staging_dir = self.tactile_recorder.staging_root / "timestamps"
        else:
            staging_dir = output_dir / f".staging_{base}_{uuid4().hex}"
        staging_dir.mkdir(parents=True, exist_ok=False)

        frame_path = staging_dir / f"{base}.parquet"
        action_path = staging_dir / f"{base}_actions.parquet"
        pq.write_table(pa.Table.from_pylist(self.frames), frame_path)
        pq.write_table(pa.Table.from_pylist(action_rows), action_path)
        summary_path = staging_dir / f"{base}_summary.json"
        summary_path.write_text(json.dumps(self.statistics(), indent=2) + "\n")
        csv_path = staging_dir / f"{base}.csv"
        with csv_path.open("w", newline="") as handle:
            if self.frames:
                writer = csv.DictWriter(handle, fieldnames=list(self.frames[0]))
                writer.writeheader()
                writer.writerows(self.frames)
        sidecar_paths = (frame_path, action_path, summary_path, csv_path)

        if self.tactile_recorder is None:
            if defer_commit:
                raise ValueError("Deferred synchronization commit requires tactile streams")
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                for source, destination in [
                    *((path, output_dir / path.name) for path in sidecar_paths),
                    *(extra_items or []),
                ]:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
            return

        commit_path = staging_dir / f"{base}_commit.json"
        commit_path.write_text(
            json.dumps(
                {
                    "transaction_id": self.tactile_recorder.transaction_id,
                    "episode_index": episode_index,
                    "tactile_streams": list(self.tactile_recorder.stream_names),
                    "timestamp_files": [path.name for path in sidecar_paths],
                },
                indent=2,
            )
            + "\n"
        )
        self.tactile_recorder.publish_episode(
            episode_index,
            [*(extra_items or []), *((path, output_dir / path.name) for path in sidecar_paths)],
            commit_marker=(commit_path, output_dir / commit_path.name),
            defer_commit_marker=defer_commit,
        )

    def commit(self) -> None:
        if self.tactile_recorder is not None:
            self.tactile_recorder.confirm_publish(dataset_saved=True)

    def discard(self) -> None:
        """Discard staged auxiliary data for an episode that will not be saved."""
        if self.tactile_recorder is not None:
            self.tactile_recorder.discard()

    def statistics(self) -> dict:
        def stats(values):
            values = [v for v in values if v is not None]
            if not values:
                return None
            return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
                    "max": float(np.max(values))}

        cameras = {}
        pairs = []
        tactile_counts = {}
        tactile_spans = {}
        for row in self.frames:
            timings = json.loads(row["camera_timing_json"])
            for name, timing in timings.items():
                cameras.setdefault(name, []).append(timing.get("age_to_action_ms"))
            pair_values = [t["pair_skew_ms"] for t in timings.values() if t.get("pair_skew_ms") is not None]
            if pair_values:
                pairs.append(max(pair_values))
            for name, timing in json.loads(row["tactile_timing_json"]).items():
                tactile_counts.setdefault(name, []).append(timing["frame_count"])
                tactile_spans.setdefault(name, []).append(timing["span_ms"])
        return {
            "frames": len(self.frames),
            "timing_basis": (
                "action send-start anchor; ordinary camera and xArm times are host receipt, "
                "not exposure; Photon device time is mapped into the host monotonic domain; "
                "GELLO raw history is not a synchronized dataset observation"
            ),
            "action_state_age_ms": stats([r["action_state_age_ms"] for r in self.frames]),
            "state_age_ms": stats([r["state_age_ms"] for r in self.frames]),
            "camera_action_age_ms": {name: stats(values) for name, values in cameras.items()},
            # Backward-compatible key; values now have causal action-age semantics.
            "camera_state_abs_offset_ms": {name: stats(values) for name, values in cameras.items()},
            "camera_pair_skew_ms": stats(pairs),
            "tactile_frame_count": {name: stats(values) for name, values in tactile_counts.items()},
            "tactile_span_ms": {name: stats(values) for name, values in tactile_spans.items()},
            "camera_future_violation_count": 0,
            "state_future_violation_count": 0,
            "tactile_writer": (
                None
                if self.tactile_recorder is None
                else self.tactile_recorder.status()
            ),
        }

    def summary(self) -> str:
        if not self.frames:
            return "synchronization: no recorded frames"
        state_ages = [row["state_age_ms"] for row in self.frames]
        camera_ages: list[float] = []
        tactile_counts = []
        for row in self.frames:
            for timing in json.loads(row["camera_timing_json"]).values():
                if timing.get("age_to_action_ms") is not None:
                    camera_ages.append(timing["age_to_action_ms"])
            tactile_counts.extend(
                timing["frame_count"]
                for timing in json.loads(row["tactile_timing_json"]).values()
            )
        state_ages.sort()
        p95 = state_ages[min(len(state_ages) - 1, int(len(state_ages) * 0.95))]
        camera_text = "n/a" if not camera_ages else f"{max(camera_ages):.1f} ms max age"
        tactile_text = "n/a" if not tactile_counts else f"{sum(tactile_counts)} raw frames"
        return (
            f"synchronization: {len(self.frames)} frames, "
            f"state age p95={p95:.1f} ms, camera={camera_text}, tactile={tactile_text}; "
            f"camera ages={json.dumps(self.statistics()['camera_action_age_ms'])}, "
            f"pair errors={json.dumps(self.statistics()['camera_pair_skew_ms'])}"
        )


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

    def __init__(self, dataset, *, mesh_cameras=None, runtime_dir=None):
        self.dataset = dataset
        self.mesh_cameras = mesh_cameras or {}
        self.runtime_dir = runtime_dir
        self._queue = queue.Queue()
        self._total_cnts = 0
        self._finish_cnts = 0
        self._exception = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="uf-async-episode-saver", daemon=True)
        self._thread.start()

    def submit_current_episode(self, synchronization: EpisodeSynchronization | None = None):
        self._raise_if_failed()
        episode_buffer = _get_episode_buffer(self.dataset)
        if _episode_buffer_size(episode_buffer) == 0:
            raise RuntimeError("Cannot async save an empty episode buffer.")

        episode_index = _episode_buffer_index(episode_buffer)
        next_episode_buffer = _create_next_episode_buffer(self.dataset, episode_buffer)
        _set_episode_buffer(self.dataset, next_episode_buffer)
        self._queue.put((episode_index, episode_buffer, synchronization))
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
                episode_index, episode_buffer, synchronization = item
                print(f'[Async] saving episode {episode_index}')
                try:
                    validate_episode_images(self.dataset, episode_buffer)
                    if self.mesh_cameras:
                        from lerobot_robot_ufactory.tactile.deferred import compute_episode_mesh

                        compute_episode_mesh(
                            self.dataset, self.mesh_cameras, self.runtime_dir,
                            episode_index, episode_buffer=episode_buffer,
                        )
                    has_tactile_transaction = (
                        synchronization is not None
                        and synchronization.tactile_recorder is not None
                    )
                    if has_tactile_transaction:
                        synchronization.write(
                            Path(self.dataset.root),
                            episode_index,
                            defer_commit=True,
                        )
                    self.dataset.save_episode(episode_data=episode_buffer)
                except TypeError as exc:
                    if "episode_data" in str(exc):
                        raise RuntimeError(
                            "--async-save requires LeRobotDataset.save_episode(episode_data=...)."
                        ) from exc
                    raise
                if has_tactile_transaction:
                    synchronization.commit()
                self._delete_saved_image_dirs(episode_index)
                if synchronization is not None and not has_tactile_transaction:
                    synchronization.write(Path(self.dataset.root), episode_index)
                if synchronization is not None:
                    print(f"[Async] {synchronization.summary()}")
                print(f'[Async] save episode {episode_index} finish')
            except BaseException as exc:
                self._exception = exc
                if item is not self._STOP and synchronization is not None:
                    try:
                        synchronization.discard()
                    except BaseException:
                        logging.exception("Failed to discard tactile staging after save failure")
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


class _EpisodeSynchronizationOwner:
    """Own every uncommitted sidecar until save code explicitly transfers it."""

    def __init__(self):
        self._owned: dict[int, EpisodeSynchronization] = {}

    def __enter__(self):
        return self

    def track(self, synchronization: EpisodeSynchronization | None):
        if synchronization is not None:
            self._owned[id(synchronization)] = synchronization
        return synchronization

    def release(self, synchronization: EpisodeSynchronization | None) -> None:
        if synchronization is not None:
            self._owned.pop(id(synchronization), None)

    def discard(self, synchronization: EpisodeSynchronization | None) -> None:
        if synchronization is None:
            return
        try:
            synchronization.discard()
        finally:
            self.release(synchronization)

    def __exit__(self, exc_type, exc_value, traceback):
        errors = []
        for synchronization in tuple(self._owned.values()):
            try:
                synchronization.discard()
            except BaseException as exc:
                errors.append(exc)
        self._owned.clear()
        if errors and exc_type is None:
            raise errors[0]
        if errors:
            error = errors[0]
            logging.error(
                "Failed to discard uncommitted tactile staging during exception cleanup",
                exc_info=(type(error), error, error.__traceback__),
            )
        return False


def _disconnect_recording_resources(robot, teleop, listener):
    """Release recording devices while preserving cleanup after partial failures."""
    stop_current = getattr(teleop, "stop_current_control", None)
    if stop_current is not None:
        try:
            stop_current()
        except Exception:
            logging.exception("Failed to stop GELLO before disconnecting recording resources")
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
    def __init__(self, robot, teleop, listener, async_episode_saver, web_preview=None):
        self.robot = robot
        self.teleop = teleop
        self.listener = listener
        self.async_episode_saver = async_episode_saver
        self.web_preview = web_preview

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            try:
                # Release active leader output before potentially slow disk writes.
                stop_current = getattr(self.teleop, "stop_current_control", None)
                if stop_current is not None:
                    try:
                        stop_current()
                    except Exception:
                        logging.exception("Failed to stop GELLO compensation before save")
                if self.async_episode_saver is not None:
                    self.async_episode_saver.close()
            finally:
                if self.web_preview is not None:
                    self.web_preview.stop()
        finally:
            _disconnect_recording_resources(self.robot, self.teleop, self.listener)
        return False


class _RawDatasetFinalize:
    """Close writers without deleting committed raw PNGs on capture failure."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.dataset.stop_image_writer()
        finally:
            self.dataset.finalize()
        return False
    

def _safe_stop_recording_image_writer(func):
    """Keep the shared image writer alive after a recoverable capture timeout."""
    parameters = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TimeoutError:
            raise
        except Exception:
            dataset = parameters.bind(*args, **kwargs).arguments.get("dataset")
            image_writer = getattr(dataset, "image_writer", None)
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise

    return wrapper


def _discard_current_episode(dataset, async_episode_saver=None):
    """Discard only the active buffer, including incomplete first-frame writes."""
    if async_episode_saver is not None:
        async_episode_saver.wait_idle()
    episode_buffer = _get_episode_buffer(dataset)
    episode_index = _episode_buffer_index(episode_buffer)
    discard_episode_images(dataset, episode_index)
    _set_episode_buffer(
        dataset, _create_empty_episode_buffer(dataset, episode_index, episode_buffer)
    )


@_safe_stop_recording_image_writer
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
    web_preview: RecordingWebPreview | None = None,
    synchronize: bool = True,
    synchronization_owner: _EpisodeSynchronizationOwner | None = None,
    recording_control=None,
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
    episode_synchronization = None
    tactile_stream_names = ()
    if dataset is not None:
        get_tactile_stream_names = getattr(robot, "tactile_stream_names", None)
        if callable(get_tactile_stream_names):
            tactile_stream_names = tuple(get_tactile_stream_names())
    record_auxiliary_timing = synchronize or bool(tactile_stream_names)
    diagnostic_logs_enabled = _diagnostic_logs_enabled(robot)
    sync_log_file = None
    sync_log_writer = None
    sync_frame_index = 0
    record_loop_succeeded = False
    try:
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
                record_timing=record_auxiliary_timing,
            )
            if recording_control is not None:
                recording_control.realtime_controller = realtime_controller
                if events.get('pause_recording'):
                    realtime_controller.request_pause()
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
                        "action_index",
                        "state_sample_s",
                        "action_sent_s",
                        "action_send_start_s",
                        "action_send_end_s",
                        "action_send_latency_ms",
                        "action_age_ms",
                        "state_age_ms",
                        "tactile_counts",
                        "observation_end_s",
                        "state_to_observation_end_ms",
                        "camera_timings",
                        "preview_publish_ms",
                        "preview_clients",
                        "preview_source_generation",
                        "preview_encoded_frames",
                        "preview_last_encode_ms",
                        "preview_max_encode_ms",
                        "record_period_ms",
                        "frame_loop_ms",
                        "frame_budget_ms",
                        "frame_overrun_ms",
                    ],
                )
                sync_log_writer.writeheader()
                logging.info("Realtime dataset synchronization log: %s", sync_log_path)

        if record_auxiliary_timing:
            episode_synchronization = EpisodeSynchronization(
                realtime_controller,
                fps,
                dataset_root=(None if dataset is None else Path(dataset.root)),
                episode_index=(
                    None if dataset is None else _current_episode_index(dataset)
                ),
                tactile_stream_names=tactile_stream_names,
            )

        timestamp = 0
        start_episode_ns = time.perf_counter_ns()
        start_episode_t = start_episode_ns / 1_000_000_000
        previous_tactile_anchor_s = start_episode_t
        previous_tactile_anchor_ns = start_episode_ns
        last_consumed_action_index = -1
        previous_loop_start_t = None
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()
            record_period_ms = (
                0.0
                if previous_loop_start_t is None
                else (start_loop_t - previous_loop_start_t) * 1000
            )
            previous_loop_start_t = start_loop_t

            if recording_control is not None:
                recording_control.watchdog()
            if events["exit_early"]:
                events["exit_early"] = False
                break

            # Get robot observation
            if realtime_controller is not None:
                first_tick = last_consumed_action_index < 0
                action_sample = realtime_controller.latest_action_sample(
                    last_consumed_action_index,
                    not_before_s=(
                        start_episode_t
                        if first_tick
                        else None
                    ),
                    # The first tick tolerates the episode-start stall (blocking
                    # first arm move, video encoder spin-up); the causal
                    # constraint above is unchanged, so a TimeoutError here
                    # still means no post-episode-start action existed.
                    wait_s=(
                        1.0
                        if first_tick
                        else max(1.0 / fps, realtime_controller.period_s * 2)
                    ),
                )
                if first_tick:
                    # The first command's send-start is stamped before a
                    # blocking first move (mode/state switch plus the initial
                    # wait=True servo command), so the sample only becomes
                    # visible once that move finishes — long after its anchor.
                    # State and tactile pairing enforce ~70 ms max ages, so
                    # such a stale anchor is unusable. The staleness only
                    # materializes while waiting above, so re-check here and
                    # skip stale first commands like any other intermediate
                    # action until a fresh anchor appears.
                    first_tick_deadline = time.perf_counter() + 1.0
                    while (
                        action_sample.send_start_s
                        < time.perf_counter() - realtime_controller.period_s
                    ):
                        remaining_s = first_tick_deadline - time.perf_counter()
                        if remaining_s <= 0:
                            raise TimeoutError(
                                "No fresh action anchor within the first-tick budget"
                            )
                        action_sample = realtime_controller.latest_action_sample(
                            action_sample.action_index,
                            not_before_s=(
                                time.perf_counter() - realtime_controller.period_s
                            ),
                            wait_s=remaining_s,
                        )
                last_consumed_action_index = action_sample.action_index
                matched_action = dict(action_sample.command)
                action_index = action_sample.action_index
                action_send_start_s = action_sample.send_start_s
                action_send_end_s = action_sample.send_end_s
                action_send_start_ns = getattr(
                    action_sample,
                    "send_start_ns",
                    round(action_send_start_s * 1_000_000_000),
                )
                action_send_end_ns = getattr(
                    action_sample,
                    "send_end_ns",
                    round(action_send_end_s * 1_000_000_000),
                )
                # Action send-start, not the observation call time, drives all
                # realtime state/camera selection.
                obs = robot.get_realtime_observation(action_send_start_s)
                observation_monotonic_s = getattr(robot, "_last_realtime_observation_monotonic_s", None)
                if observation_monotonic_s is None:
                    raise RuntimeError(
                        "Realtime robot did not expose its selected state timestamp"
                    )
                sync_timing = getattr(robot, "_last_realtime_sync_timing", {})
                state_sample_ns = sync_timing.get("state_sample_ns")
                state_rt_receive_s = sync_timing.get("state_rt_receive_s")
                state_rt_receive_ns = sync_timing.get("state_rt_receive_ns")
                state_anchor_s = state_rt_receive_s or observation_monotonic_s
                realtime_controller.update_observation(obs)
            else:
                action_sample = None
                obs = robot.get_observation()
                sync_timing = getattr(robot, "_last_observation_sync_timing", {})
                observation_monotonic_s = sync_timing.get("state_sample_s", time.perf_counter())
                state_sample_ns = sync_timing.get("state_sample_ns")
                state_rt_receive_s = sync_timing.get("state_rt_receive_s")
                state_rt_receive_ns = sync_timing.get("state_rt_receive_ns")
                state_anchor_s = state_rt_receive_s or observation_monotonic_s

            # Applies a pipeline to the raw robot observation, default is IdentityProcessor
            obs_processed = robot_observation_processor(obs)
            preview_publish_ms = 0.0
            if web_preview is not None:
                # This only replaces references in a latest-frame slot. All image
                # processing and network I/O remain on preview background threads.
                before_preview_publish_t = time.perf_counter()
                web_preview.publish(obs_processed)
                preview_publish_ms = (time.perf_counter() - before_preview_publish_t) * 1000

            if policy is not None or dataset is not None:
                obs_for_dataset = obs_processed
                convert_observation = getattr(robot, "convert_observation_for_recording", None)
                if convert_observation is not None:
                    obs_for_dataset = convert_observation(obs_processed)
                obs_for_dataset = dict(obs_for_dataset)
                for key, shape in getattr(dataset, '_offline_mesh_fields', {}).items():
                    obs_for_dataset[key.removeprefix('observation.')] = np.full(shape, np.nan, dtype=np.float32)
                observation_frame = build_dataset_frame(dataset.features, obs_for_dataset, prefix=OBS_STR)

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
                    apply_pending_gello_joint_mode(robot, teleop, obs)
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
                action_send_start_ns = time.perf_counter_ns()
                robot_action_to_send = apply_keyboard_gripper_stop(robot, teleop, robot_action_to_send)
                _sent_action = robot.send_action(robot_action_to_send)
                action_send_end_ns = time.perf_counter_ns()
                action_send_start_s = action_send_start_ns / 1_000_000_000
                action_send_end_s = action_send_end_ns / 1_000_000_000
                action_index = sync_frame_index
            else:
                _sent_action = matched_action
            # Robots may clamp or otherwise sanitize a command before sending it.
            # Store that effective command so demonstrations match the motion.
            if isinstance(_sent_action, dict):
                action_values = _sent_action

            convert_action = getattr(robot, "convert_action_for_recording", None)
            if convert_action is not None:
                action_values = convert_action(action_values)

            tactile_samples = {}
            get_tactile_window = getattr(robot, "get_tactile_samples_between", None)
            if tactile_stream_names and callable(get_tactile_window):
                tactile_samples = get_tactile_window(
                    previous_tactile_anchor_s,
                    action_send_start_s,
                )

            # Write to dataset
            if dataset is not None:
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": single_task}
                if frame_callback is not None:
                    frame = frame_callback(frame)
                dataset.add_frame(frame)

            if episode_synchronization is not None:
                episode_synchronization.add_frame(
                    frame_index=sync_frame_index,
                    state_sample_s=observation_monotonic_s,
                    state_rt_receive_s=state_rt_receive_s,
                    camera_timing=sync_timing.get("camera", {}),
                    state_age_ms=sync_timing.get("state_age_ms"),
                    action_send_start_s=action_send_start_s,
                    action_send_end_s=action_send_end_s,
                    action_index=action_index,
                    state_sample_ns=state_sample_ns,
                    state_rt_receive_ns=state_rt_receive_ns,
                    action_send_start_ns=action_send_start_ns,
                    action_send_end_ns=action_send_end_ns,
                    tactile_window_start_s=previous_tactile_anchor_s,
                    tactile_window_start_ns=previous_tactile_anchor_ns,
                    tactile_samples=tactile_samples,
                )

            previous_tactile_anchor_s = action_send_start_s
            previous_tactile_anchor_ns = action_send_start_ns

            if sync_log_writer is not None:
                observation_end_s = getattr(
                    robot, "_last_realtime_observation_end_monotonic_s", observation_monotonic_s
                )
                camera_timings = getattr(robot, "_last_realtime_camera_timings", {})
                preview_stats = web_preview.timing_stats() if web_preview is not None else {}
                frame_loop_ms = (time.perf_counter() - start_loop_t) * 1000
                frame_budget_ms = 1000 / fps
                sync_log_writer.writerow(
                    {
                        "frame": sync_frame_index,
                        "action_index": action_index,
                        "state_sample_s": f"{observation_monotonic_s:.9f}",
                        # Legacy column now follows the primary send-start anchor.
                        "action_sent_s": f"{action_send_start_s:.9f}",
                        "action_send_start_s": f"{action_send_start_s:.9f}",
                        "action_send_end_s": f"{action_send_end_s:.9f}",
                        "action_send_latency_ms": f"{(action_send_end_s - action_send_start_s) * 1000:.3f}",
                        "action_age_ms": f"{(action_send_start_s - state_anchor_s) * 1000:.3f}",
                        "state_age_ms": f"{(action_send_start_s - state_anchor_s) * 1000:.3f}",
                        "tactile_counts": repr(
                            {name: len(samples) for name, samples in tactile_samples.items()}
                        ),
                        "observation_end_s": f"{observation_end_s:.9f}",
                        "state_to_observation_end_ms": f"{(observation_end_s - observation_monotonic_s) * 1000:.3f}",
                        "camera_timings": repr(camera_timings),
                        "preview_publish_ms": f"{preview_publish_ms:.6f}",
                        "preview_clients": preview_stats.get("preview_clients", 0),
                        "preview_source_generation": preview_stats.get(
                            "preview_source_generation", 0
                        ),
                        "preview_encoded_frames": preview_stats.get("preview_encoded_frames", 0),
                        "preview_last_encode_ms": f'{preview_stats.get("preview_last_encode_ms", 0.0):.3f}',
                        "preview_max_encode_ms": f'{preview_stats.get("preview_max_encode_ms", 0.0):.3f}',
                        "record_period_ms": f"{record_period_ms:.3f}",
                        "frame_loop_ms": f"{frame_loop_ms:.3f}",
                        "frame_budget_ms": f"{frame_budget_ms:.3f}",
                        "frame_overrun_ms": f"{max(0.0, frame_loop_ms - frame_budget_ms):.3f}",
                    }
                )

            sync_frame_index += 1
            if recording_control is not None:
                recording_control.progress(frames=sync_frame_index,
                    elapsed=time.perf_counter() - start_episode_t, has_unsaved=True)

            if display_data:
                log_rerun_data(
                    observation=obs_processed, action=action_values, compress_images=display_compressed_images
                )

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))

            timestamp = time.perf_counter() - start_episode_t

        record_loop_succeeded = True
        if synchronization_owner is not None:
            synchronization_owner.track(episode_synchronization)
    except InterruptedError:
        if not events.get('pause_recording'):
            raise
        record_loop_succeeded = True
        if synchronization_owner is not None:
            synchronization_owner.track(episode_synchronization)
    finally:
        try:
            if realtime_controller is not None:
                if events.get('pause_recording'):
                    realtime_controller.request_pause()
                # Preserve the capture exception instead of replacing it with
                # the same latched control fault during worker cleanup.
                if sys.exc_info()[0] is None:
                    realtime_controller.stop()
                else:
                    realtime_controller.stop(raise_on_fault=False)
        finally:
            if recording_control is not None:
                recording_control.realtime_controller = None
            if realtime_controller is None and events.get('pause_recording'):
                robot.pause_motion()
            if sync_log_file is not None:
                sync_log_file.close()
            if not record_loop_succeeded and episode_synchronization is not None:
                episode_synchronization.discard()
    return episode_synchronization


def _reset_recording_robot(robot, *, open_gripper_first=False, cancel_check=None):
    if cancel_check is not None and cancel_check():
        robot.pause_motion()
        raise InterruptedError('Reset cancelled before motion')
    if open_gripper_first:
        open_gripper = getattr(robot, "open_gripper", None)
        if open_gripper is not None:
            # The gripper command must finish successfully before arm motion.
            open_gripper()
    reset = getattr(robot, "reset_to_initial", None)
    if reset is None:
        reset = robot.configure
    if cancel_check is None:
        reset()
    else:
        reset(cancel_check=cancel_check)


def _prepare_recording_episode(robot, teleop, is_uf_teleop, manual_mode, *, reset_robot=True, cancel_check=None):
    if is_uf_teleop:
        # Stop teleop output before handing control to the xArm reset motion.
        teleop.set_teleop_enabled(False)

    if reset_robot and (is_uf_teleop or manual_mode):
        _reset_recording_robot(robot, cancel_check=cancel_check)

    if is_uf_teleop:
        obs = robot.get_observation()
        teleop.set_teleop_enabled(True, obs)


def _print_record_controls(is_recorded, manual_mode, teleop=None):
    if is_recorded:
        controls = '[ESC] Exit  [←] Reset  [→] Save'
    else:
        start_label = 'Reset / Start' if manual_mode else 'Start'
        controls = f'[ESC] Exit  [Space] {start_label}  [←] Reset  [→] Save'
    if manual_mode:
        controls += '  [C] Close  [O] Open'
    if getattr(getattr(teleop, "config", None), "joint7_only_mode_enabled", False):
        controls += '  [S] J7 only / All joints'
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
    recover_postprocessing(root)
    existed = root.exists()

    if not existed:
        if cfg.resume:
            raise RuntimeError(f"Cannot resume because the dataset directory does not exist: {root}")
        return

    missing = _missing_dataset_files(root)
    has_raw = bool(RawEpisodeStore.checkpoints(root)) and (root / "meta/info.json").is_file()
    if missing and not has_raw:
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


@exclusive_recording
def record(cfg: UFRecordConfig, async_save: bool = False, postprocess_only: bool = False, recording_control=None) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    # Encode each completed episode immediately.
    cfg.dataset.video_encoding_batch_size = 1
    if postprocess_only:
        cfg.resume = True

    if cfg.offline_mesh3dflow:
        if not cfg.dataset.video:
            raise ValueError('offline_mesh3dflow requires video recording')
        cfg.dataset.video_encoding_batch_size = 1
        from lerobot_robot_ufactory.tactile import TactileCameraConfig

        tactile_configs = [
            camera for camera in cfg.robot.cameras.values()
            if isinstance(camera, TactileCameraConfig)
        ]
        if not tactile_configs:
            raise ValueError('offline_mesh3dflow requires at least one tactile sensor')
        for camera in tactile_configs:
            camera.configure_deferred_processing()
    if recording_control is not None:
        recording_control.prepare_dataset()
    _prepare_dataset_root(cfg)

    if cfg.resume and not postprocess_only:
        root = Path(cfg.dataset.root)
        info = json.loads((root / "meta/info.json").read_text())
        pending = [
            path for path in RawEpisodeStore.checkpoints(root)
            if json.loads(path.read_text())["episode_index"] >= info["total_episodes"]
        ]
        if pending:
            raise RuntimeError(
                f"Cannot resume recording: {len(pending)} raw episode(s) "
                f"still need processing in {root}. Run --postprocess-only with the recording "
                "configuration first."
            )

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
    # Keep 3D tactile displacement tensors out of the generic camera feature
    # pipeline, which treats every HxWx3 shape as an image/video stream.
    dataset_features.update(getattr(robot, "tactile_observation_features", {}))
    offline_mesh_fields = {}
    if cfg.offline_mesh3dflow:
        from lerobot_robot_ufactory.tactile import TactileCamera
        for name, camera in robot.cameras.items():
            if isinstance(camera, TactileCamera):
                for suffix, shape in camera.deferred_feature_shapes.items():
                    key = f'observation.{name}.{suffix}'
                    offline_mesh_fields[key] = shape
                    dataset_features[key] = {
                        'dtype': 'float32', 'shape': shape, 'names': None
                    }

    if cfg.resume:
        dataset = open_recording_dataset(
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
            robot_type=_dataset_robot_type(robot),
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    dataset._offline_mesh_fields = offline_mesh_fields
    # A previous crash may leave only private transaction directories. Recover
    # an interrupted publish when a journal exists, then remove stale staging.
    from lerobot_robot_ufactory.tactile.persistence import cleanup_stale_tactile_staging

    cleanup_stale_tactile_staging(Path(dataset.root))

    if postprocess_only:
        from lerobot_robot_ufactory.tactile import TactileCamera
        cameras = {name: cam for name, cam in robot.cameras.items() if isinstance(cam, TactileCamera)}
        with _RawDatasetFinalize(dataset):
            result = postprocess_raw_episodes(dataset, cameras)
        return result

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

    # Every recording session gets its own reference/config snapshot. Resuming
    # must not overwrite runtime files belonging to previous episodes.
    from lerobot_robot_ufactory.tactile import TactileCamera

    tactile_cameras = {
        name: cam for name, cam in robot.cameras.items()
        if isinstance(cam, TactileCamera)
    }
    runtime_dir = None
    if tactile_cameras:
        runtime_dir = Path(dataset.root) / "runtime" / f"session_{uuid4().hex}"
        for cam in tactile_cameras.values():
            cam.runtime_export_dir = runtime_dir

    web_preview = None
    try:
        if recording_control is None:
            robot.connect()
        else:
            robot.connect(defer_motion=True)
        if runtime_dir is not None:
            manifest = {
                "created_unix_s": time.time(),
                "first_episode_index": dataset.num_episodes,
                "cameras": {
                    name: cam.runtime_manifest()
                    for name, cam in tactile_cameras.items()
                },
            }
            with (runtime_dir / "manifest.json").open("x") as stream:
                json.dump(manifest, stream, indent=2)
            logging.info("Photon runtime configurations saved: %s", runtime_dir)
        if teleop is not None:
            teleop.connect()
            if getattr(teleop.config, "gripper_control_mode", "gello") == "keyboard":
                speed, stroke = robot.get_gripper_motion_parameters()
                teleop.set_gripper_motion_parameters(speed, stroke)
        if recording_control is not None:
            web_preview = recording_control.preview
            web_preview.start_encoder()
        elif cfg.web_preview.enabled:
            web_preview = RecordingWebPreview(cfg.web_preview,
                camera_types={name: camera.type for name, camera in cfg.robot.cameras.items()})
            web_preview.start()
            print(f"Camera web preview: {web_preview.url}")
            if cfg.web_preview.host == "0.0.0.0":
                print(
                    f"From another machine: http://<recorder-ip>:{cfg.web_preview.port}/"
                )
    except BaseException:
        try:
            if web_preview is not None:
                web_preview.stop()
            _disconnect_recording_resources(robot, teleop, None)
        except BaseException:
            logging.exception("Failed to clean up after recording device connection failure")
        raise

    if recording_control is not None:
        from lerobot_robot_ufactory.utils.webapp.recording_control import controlled_recording
        return controlled_recording(cfg, robot, teleop, dataset, recording_control, web_preview,
            teleop_action_processor, robot_action_processor, robot_observation_processor,
            runtime_dir, tactile_cameras)

    is_evt = not is_headless()
    is_uf_teleop = isinstance(teleop, UFBaseTeleop)
    is_recorded = False
    robot_reset_after_discard = False
    waiting_for_retry = False
    wait_for_start_release = False
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
            update_gello_joint_mode_key(teleop, key, True)
            _update_manual_gripper_key_state(key, True, manual_gripper_keys)
            if getattr(teleop, "config", None) is not None and getattr(teleop.config, "gripper_control_mode", "gello") == "keyboard":
                teleop.set_gripper_keyboard_state(**manual_gripper_keys)
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
            update_gello_joint_mode_key(teleop, key, False)
            _update_manual_gripper_key_state(key, False, manual_gripper_keys)
            if getattr(teleop, "config", None) is not None and getattr(teleop.config, "gripper_control_mode", "gello") == "keyboard":
                teleop.set_gripper_keyboard_state(**manual_gripper_keys)
            try:
                if key == keyboard.Key.enter:
                    _print_record_controls(is_recorded, manual_mode, teleop)
                    # is_recorded = True
            except Exception as e:
                print(f"Error handling key release: {e}")
            if key in key_dict:
                key_dict[key] = False

        listener, events = init_keyboard_listener(events=events, on_press=on_press, on_release=on_release)
        print("\n********** Episode Record Loop Start **********")
        _print_record_controls(is_recorded, manual_mode, teleop)
    else:
        input('⌨   Press Enter to start record >>> ')
        is_recorded = True
        print('\n********** Episode Record Loop Start **********')

    frame_callback = None
    mesh_cameras = tactile_cameras if cfg.offline_mesh3dflow else {}
    async_episode_saver = (
        AsyncEpisodeSaver(dataset, mesh_cameras=mesh_cameras, runtime_dir=runtime_dir)
        if async_save else None
    )
    if async_episode_saver is not None:
        print('Async episode saving is enabled.')

    episode_owner = _EpisodeSynchronizationOwner()
    # Close pending async saves before VideoEncodingManager finalizes Parquet
    # writers, including when capture or device cleanup raises an exception.
    dataset_cleanup = VideoEncodingManager(dataset)
    with dataset_cleanup, _RecordingCleanup(
        robot, teleop, listener, async_episode_saver, web_preview
    ), episode_owner:
        # num_episodes is a dataset-wide limit.  Count existing episodes so a
        # resumed recording cannot exceed it by recording another full batch.
        recorded_episodes = dataset.num_episodes
        if recorded_episodes >= cfg.dataset.num_episodes:
            print(
                f"Episode limit already reached ({recorded_episodes}/"
                f"{cfg.dataset.num_episodes}); nothing to record."
            )
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            time.sleep(0.01)
            if is_evt:
                if wait_for_start_release:
                    wait_for_start_release = bool(key_dict[keyboard.Key.space])
                elif not is_recorded and key_dict[keyboard.Key.space]:
                    is_recorded = True

            if teleop is not None and isinstance(teleop, UFMockTeleop):
                if waiting_for_retry and not is_recorded:
                    continue
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
                waiting_for_retry = False
                events["rerecord_episode"] = False
                events["exit_early"] = False
                episode_synchronization = None
                episode_timed_out = False
                try:
                    if is_uf_teleop or manual_mode:
                        _prepare_recording_episode(
                            robot, teleop, is_uf_teleop, manual_mode,
                            reset_robot=not robot_reset_after_discard,
                        )
                        robot_reset_after_discard = False
                    log_say(f"Recording episode {_current_episode_index(dataset)}", cfg.play_sounds)
                    episode_synchronization = record_loop(
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
                        web_preview=web_preview,
                        synchronize=cfg.synchronize,
                        synchronization_owner=episode_owner,
                    )
                except TimeoutError as exc:
                    logging.warning(
                        "Episode %s synchronization timed out; discarding this episode: %s",
                        _current_episode_index(dataset), exc,
                    )
                    episode_timed_out = True
                    events["rerecord_episode"] = True
                episode_owner.track(episode_synchronization)
            else:
                continue
            if events['stop_recording'] and not events["rerecord_episode"]:
                episode_owner.discard(episode_synchronization)
                _discard_current_episode(dataset, async_episode_saver)
                break
            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if is_uf_teleop:
                    teleop.set_teleop_enabled(False)
                episode_owner.discard(episode_synchronization)
                _discard_current_episode(dataset, async_episode_saver)
                is_recorded = False
                if events["stop_recording"]:
                    break
                if is_uf_teleop or manual_mode:
                    _reset_recording_robot(robot, open_gripper_first=True)
                    robot_reset_after_discard = True
                if is_evt:
                    wait_for_start_release = bool(key_dict[keyboard.Key.space])
                    if episode_timed_out:
                        waiting_for_retry = True
                    _print_record_controls(is_recorded, manual_mode, teleop)
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
                    try:
                        validate_episode_images(dataset, _get_episode_buffer(dataset))
                        if mesh_cameras:
                            from lerobot_robot_ufactory.tactile.deferred import compute_episode_mesh

                            compute_episode_mesh(
                                dataset, mesh_cameras, runtime_dir, episode_index
                            )
                        has_tactile_transaction = (
                            episode_synchronization is not None
                            and episode_synchronization.tactile_recorder is not None
                        )
                        if has_tactile_transaction:
                            episode_synchronization.write(
                                Path(dataset.root),
                                episode_index,
                                defer_commit=True,
                            )
                        dataset.save_episode()
                        if has_tactile_transaction:
                            episode_synchronization.commit()
                        elif episode_synchronization is not None:
                            episode_synchronization.write(Path(dataset.root), episode_index)
                        if episode_synchronization is not None:
                            log_say(episode_synchronization.summary(), cfg.play_sounds)
                            episode_owner.release(episode_synchronization)
                        log_say(f"[Finish] Save episode {episode_index}", cfg.play_sounds)
                    except BaseException:
                        episode_owner.discard(episode_synchronization)
                        raise
                else:
                    queued_episode_index = async_episode_saver.submit_current_episode(
                        episode_synchronization
                    )
                    episode_owner.release(episode_synchronization)
                    if queued_episode_index is not None:
                        log_say(f"[Queued] Save episode {queued_episode_index}", cfg.play_sounds)

                recorded_episodes += 1
                is_recorded = False
                if recorded_episodes >= cfg.dataset.num_episodes:
                    print(f"Episode limit reached ({recorded_episodes}/{cfg.dataset.num_episodes}).")
                    break
                if is_evt:
                    _print_record_controls(is_recorded, manual_mode, teleop)
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
    parser.add_argument('--postprocess-only', '--postprocess_only', action='store_true',
                       help='Process saved raw episodes without connecting recording devices')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    if args.r:
        cfg.resume = True
    cfg.play_sounds = False
    record(cfg, async_save=args.async_save, postprocess_only=args.postprocess_only)


if __name__ == "__main__":
    main()
