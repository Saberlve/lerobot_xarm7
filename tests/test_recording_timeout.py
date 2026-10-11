"""Exercise timeout recovery through the recorder without connected hardware."""

import copy
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.datasets.image_writer import AsyncImageWriter

from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording
from lerobot_robot_ufactory.datasets.deferred_mesh import compute_episode_mesh as real_compute_episode_mesh
from lerobot_robot_ufactory.tactile.photon.camera import XensePhotonSample
from lerobot_robot_ufactory.tactile.photon.config import XensePhotonCameraConfig


class Dataset:
    """Use real image writes and tactile transactions, keeping saves in memory."""

    def __init__(self, root, features, fps):
        self.root = root
        self.features = features
        image_keys = [k for k, v in features.items() if v["dtype"] in ("image", "video")]
        self.meta = SimpleNamespace(features=features, image_keys=image_keys, video_keys=image_keys)
        self.fps = fps
        self.num_episodes = 0
        self.episodes_since_last_encoding = 0
        self.image_writer = AsyncImageWriter(num_threads=1)
        self.episode_buffer = self.create_episode_buffer()
        self.saved = []
        self.events = None
        self.finalized = False

    def create_episode_buffer(self, episode_index=None):
        return {
            **{key: [] for key in self.features},
            "task": [],
            "episode_index": self.num_episodes if episode_index is None else episode_index,
            "size": 0,
            "frames": [],
        }

    def _get_image_file_dir(self, episode_index, key):
        return self.root / "images" / key / f"episode_{episode_index:06d}"

    def _get_image_file_path(self, episode_index, image_key, frame_index):
        return self._get_image_file_dir(episode_index, image_key) / f"frame-{frame_index:06d}.png"

    def _wait_image_writer(self):
        self.image_writer.wait_until_done()

    def add_frame(self, frame):
        buffer = self.episode_buffer
        for key in self.meta.image_keys:
            path = self._get_image_file_path(buffer["episode_index"], key, buffer["size"])
            path.parent.mkdir(parents=True, exist_ok=True)
            self.image_writer.save_image(frame[key], path)
            buffer[key].append(str(path))
        for key in self.features:
            if key not in self.meta.image_keys:
                buffer[key].append(frame[key])
        buffer["task"].append(frame["task"])
        buffer["frames"].append(frame)
        buffer["size"] += 1
        if buffer["size"] == 2:
            self.events["exit_early"] = True

    def save_episode(self, episode_data=None):
        buffer = self.episode_buffer if episode_data is None else episode_data
        if not buffer.get("frames"):
            buffer["frames"] = [
                {key: buffer[key][i] for key in self.features} for i in range(buffer["size"])
            ]
        recording.validate_episode_images(self, buffer)
        assert buffer["episode_index"] == self.num_episodes
        self.saved.append(copy.deepcopy(buffer))
        self.num_episodes += 1
        if episode_data is None:
            self.episode_buffer = self.create_episode_buffer()

    def finalize(self):
        self.image_writer.stop()
        self.finalized = True

    def stop_image_writer(self):
        self.image_writer.stop()


@pytest.fixture
def session(tmp_path, monkeypatch):
    import lerobot_robot_ufactory.tactile as tactile
    import lerobot_robot_ufactory.datasets.deferred_mesh as deferred

    class Camera:
        deferred_feature_shapes = {"mesh_motion_3d": (2, 2, 3)}

        def runtime_manifest(self):
            return {}

    class Robot:
        name = robot_type = "test_robot"
        action_features = {"J1.pos": float}
        observation_features = {"J1.pos": float, "photon": (2, 3, 3)}
        enable_logs = False
        _is_connected = False
        cameras = {"photon": Camera()}
        attempt = -1
        observation_count = 0
        failure_frames = 1
        failure = TimeoutError
        # Successful first episode, two failed attempts, successful retry.
        outcomes = [False, True, True, False]

        def connect(self):
            self._is_connected = True
            self.cameras["photon"].runtime_export_dir.mkdir(parents=True)

        def disconnect(self):
            self._is_connected = False

        def reset_to_initial(self):
            self.attempt += 1
            self.observation_count = 0

        def get_observation(self):
            self.observation_count += 1
            # One observation initializes record_loop; subsequent ones record.
            if self.outcomes[self.attempt] and self.observation_count == self.failure_frames + 2:
                raise self.failure("camera synchronization failed")
            return {
                "J1.pos": float(self.attempt),
                "photon": np.full((2, 3, 3), self.attempt + 1, dtype=np.uint8),
            }

        def send_action(self, action):
            return action

        def tactile_stream_names(self):
            return ("photon",)

        def get_tactile_samples_between(self, start, end):
            return {
                "photon": (
                    XensePhotonSample(
                        frame_bgr=np.full((2, 3, 3), self.attempt + 1, dtype=np.uint8),
                        marker_motion_3d=None,
                        sensor_timestamp_s=1.0,
                        capture_monotonic_s=(start + end) / 2,
                    ),
                )
            }

    robot = Robot()
    cfg = SimpleNamespace(
        robot=SimpleNamespace(
            manual_mode=True, cameras={"photon": XensePhotonCameraConfig(serial_number="test")}
        ),
        teleop=None,
        policy=None,
        resume=False,
        offline_mesh3dflow=False,
        web_preview=recording.WebPreviewConfig(),
        synchronize=True,
        play_sounds=False,
        display_data=False,
        dataset=SimpleNamespace(
            repo_id="test/timeout",
            root=tmp_path / "dataset",
            fps=200,
            video=False,
            video_encoding_batch_size=1,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=1,
            num_episodes=2,
            episode_time_s=5,
            single_task="test",
            push_to_hub=False,
        ),
    )
    state = SimpleNamespace(
        robot=robot,
        cfg=cfg,
        dataset=None,
        stop_after_timeout=False,
        retry_prompts=0,
        mesh_episodes=[],
    )

    def create_dataset(*args, **kwargs):
        state.dataset = Dataset(kwargs["root"], kwargs["features"], args[1])
        return state.dataset

    original_loop = recording.record_loop

    def loop(**kwargs):
        state.dataset.events = kwargs["events"]
        return original_loop(**kwargs)

    def prompt(message):
        if "rerecord" in message:
            state.retry_prompts += 1
            dataset = state.dataset
            assert dataset.episode_buffer["episode_index"] == 1
            assert dataset.episode_buffer["size"] == 0
            assert not dataset._get_image_file_dir(1, "observation.images.photon").exists()
            assert not dataset.image_writer._stopped
            if cfg.offline_mesh3dflow:
                if "observation.photon.tactile_range" not in dataset.features:
                    assert state.mesh_episodes == [0]
                staging = dataset.root / "tactile_streams/.staging"
                assert not staging.exists() or not any(staging.iterdir())
                assert dataset.num_episodes == 1
                assert (
                    dataset.root / "tactile_streams/photon/episode_000000/samples.parquet"
                ).exists()
            if state.stop_after_timeout:
                dataset.events["stop_recording"] = True
        return ""

    def compute_mesh(dataset, cameras, runtime_dir, episode_index, episode_buffer=None, synchronization=None):
        buffer = dataset.episode_buffer if episode_buffer is None else episode_buffer
        assert buffer["episode_index"] == episode_index
        assert dataset.num_episodes == episode_index
        assert robot._is_connected
        buffer["observation.photon.mesh_motion_3d"] = [
            np.full((2, 2, 3), episode_index + 1, dtype=np.float32)
            for _ in range(buffer["size"])
        ]
        state.mesh_episodes.append(episode_index)

    def postprocess(dataset, cameras, *, progress=None):
        store = recording.RawEpisodeStore(dataset)
        for path in store.checkpoints(dataset.root):
            buffer, manifest = store.load(path)
            if manifest["episode_index"] < dataset.num_episodes:
                continue
            if cfg.offline_mesh3dflow:
                state.mesh_episodes.append(manifest["episode_index"])
            dataset.save_episode(episode_data=buffer)
        return dataset

    monkeypatch.setattr(tactile, "TactileCamera", Camera)
    monkeypatch.setattr(deferred, "deferred_sessions", lambda *args: nullcontext())
    monkeypatch.setattr(deferred, "compute_episode_mesh", compute_mesh)
    monkeypatch.setattr(recording, "_prepare_dataset_root", lambda cfg: None)
    monkeypatch.setattr(recording, "asdict", lambda cfg: {})
    monkeypatch.setattr(recording, "init_logging", lambda: None)
    monkeypatch.setattr(recording, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(recording.LeRobotDataset, "create", create_dataset)
    monkeypatch.setattr(recording, "is_headless", lambda: True)
    monkeypatch.setattr(recording, "record_loop", loop)
    monkeypatch.setattr(recording, "postprocess_raw_episodes", postprocess)
    monkeypatch.setattr("builtins.input", prompt)
    yield state
    if state.dataset is not None:
        state.dataset.image_writer.stop()


@pytest.mark.parametrize("mode", ["sync", "async", "offline", "offline_async"])
@pytest.mark.parametrize("failure_frames", [0, 1])
@pytest.mark.parametrize("stop_after_timeout", [False, True])
def test_timeout_preserves_previous_episode(
    session, mode, failure_frames, stop_after_timeout, caplog
):
    session.cfg.offline_mesh3dflow = mode.startswith("offline")
    session.cfg.dataset.video = mode.startswith("offline")
    session.robot.failure_frames = failure_frames
    session.stop_after_timeout = stop_after_timeout

    dataset = recording.record(session.cfg, async_save=mode in ("async", "offline_async"))

    expected = 1 if stop_after_timeout else 2
    assert dataset.num_episodes == expected
    assert [row["episode_index"] for row in dataset.saved] == list(range(expected))
    assert [row["size"] for row in dataset.saved] == [2] * expected
    assert session.retry_prompts == (1 if stop_after_timeout else 2)
    assert dataset.saved[0]["frames"][0]["observation.state"].tolist() == [0.0]
    if not stop_after_timeout:
        assert dataset.saved[1]["frames"][0]["observation.state"].tolist() == [3.0]
    assert session.mesh_episodes == (list(range(expected)) if mode.startswith("offline") else [])
    if mode.startswith("offline"):
        for index, buffer in enumerate(dataset.saved):
            np.testing.assert_array_equal(
                buffer["observation.photon.mesh_motion_3d"],
                np.full((2, 2, 2, 3), index + 1, dtype=np.float32),
            )
    assert session.cfg.dataset.video_encoding_batch_size == 1
    assert dataset.finalized
    assert not session.robot._is_connected
    assert "Episode 1 synchronization timed out" in caplog.text
    for episode in range(expected):
        assert (dataset.root / "timestamps" / f"episode_{episode:06d}.parquet").is_file()
        assert (
            dataset.root / "tactile_streams/photon" / f"episode_{episode:06d}/samples.parquet"
        ).is_file()
    staging = dataset.root / "tactile_streams/.staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_non_timeout_recording_failure_still_exits(session):
    session.robot.outcomes = [True]
    session.robot.failure = RuntimeError
    with pytest.raises(RuntimeError, match="camera synchronization failed"):
        recording.record(session.cfg)
    assert session.dataset.image_writer._stopped
    assert session.dataset.finalized
    assert session.retry_prompts == 0
    assert session.dataset.episode_buffer["size"] == 0
    assert not session.dataset._get_image_file_dir(0, "observation.images.photon").exists()


def test_timeout_is_discarded_before_caller_recovers(session, monkeypatch):
    session.robot.outcomes = [False, True, False]
    original_loop = recording.record_loop
    discarded = []

    def loop(**kwargs):
        try:
            return original_loop(**kwargs)
        except TimeoutError:
            dataset = session.dataset
            assert dataset.episode_buffer["episode_index"] == 1
            assert dataset.episode_buffer["size"] == 0
            assert not dataset._get_image_file_dir(1, "observation.images.photon").exists()
            assert not dataset.image_writer._stopped
            discarded.append(1)
            raise

    monkeypatch.setattr(recording, "record_loop", loop)
    dataset = recording.record(session.cfg)
    assert discarded == [1]
    assert dataset.num_episodes == 2


def test_capture_cleanup_failure_still_clears_buffer_and_preserves_error(session, monkeypatch, caplog):
    session.robot.outcomes = [True]
    session.robot.failure = ValueError

    def fail_cleanup(*args):
        raise OSError("cannot remove temporary images")

    monkeypatch.setattr(recording, "discard_episode_images", fail_cleanup)
    with pytest.raises(ValueError, match="camera synchronization failed"):
        recording.record(session.cfg)
    assert session.dataset.episode_buffer["size"] == 0
    assert session.dataset.num_episodes == 0
    assert session.dataset.finalized and not session.robot._is_connected
    assert "cannot remove temporary images" in caplog.text


def test_save_timeout_is_not_recovered_as_recording_timeout(session, monkeypatch):
    def fail_save(self, *args, **kwargs):
        raise TimeoutError("disk save timed out")

    monkeypatch.setattr(Dataset, "save_episode", fail_save)
    with pytest.raises(TimeoutError, match="disk save timed out"):
        recording.record(session.cfg)
    assert session.retry_prompts == 0


@pytest.mark.parametrize("async_save", [False, True])
def test_device_error_preserves_previously_saved_episode(session, async_save, monkeypatch):
    original_close = recording.AsyncEpisodeSaver.close

    def close_before_finalize(saver):
        assert not saver.dataset.finalized
        return original_close(saver)

    monkeypatch.setattr(recording.AsyncEpisodeSaver, "close", close_before_finalize)
    session.robot.failure = RuntimeError
    with pytest.raises(RuntimeError, match="camera synchronization failed"):
        recording.record(session.cfg, async_save=async_save)
    assert session.dataset.num_episodes == 1
    assert session.dataset.saved[0]["episode_index"] == 0
    assert session.dataset.saved[0]["size"] == 2
    assert (
        session.dataset.root / "tactile_streams/photon/episode_000000/samples.parquet"
    ).is_file()
    assert session.dataset.finalized

    assert session.dataset.episode_buffer["size"] == 0
    assert not session.dataset._get_image_file_dir(1, "observation.images.photon").exists()
    assert not (session.dataset.root / "timestamps/episode_000001.parquet").exists()
    staging = session.dataset.root / "tactile_streams/.staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_keyboard_timeout_requires_release_and_new_start(session, monkeypatch):
    import time

    keys = SimpleNamespace(space="space", enter="enter", right="right", left="left", esc="esc")
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=SimpleNamespace(Key=keys)))
    monkeypatch.setattr(recording, "is_headless", lambda: False)
    session.robot.outcomes = [False, True, False]
    steps = []
    listener = SimpleNamespace(stop=lambda: steps.append("stop"))
    callbacks = {}
    sleeps_after_timeout = 0

    def init_listener(events, on_press, on_release):
        callbacks.update(press=on_press, release=on_release)
        on_press(keys.space)
        return listener, events

    def main_loop_sleep(_duration):
        nonlocal sleeps_after_timeout
        if session.robot.attempt == 2:
            # Keep Space held for two main-loop ticks. Neither may restart.
            assert session.dataset.episode_buffer["size"] == 0
            sleeps_after_timeout += 1
            if sleeps_after_timeout == 3:
                callbacks["release"](keys.space)
            elif sleeps_after_timeout == 4:
                callbacks["press"](keys.space)
            elif sleeps_after_timeout > 4:
                pytest.fail("recorder did not resume on a new Space press")

    monkeypatch.setattr(recording, "init_keyboard_listener", init_listener)
    monkeypatch.setattr(
        recording,
        "time",
        SimpleNamespace(
            sleep=main_loop_sleep,
            perf_counter=time.perf_counter,
            perf_counter_ns=time.perf_counter_ns,
            time=time.time,
        ),
    )
    dataset = recording.record(session.cfg)
    assert sleeps_after_timeout == 4
    assert dataset.num_episodes == 2
    assert steps == ["stop"]


@pytest.mark.parametrize("async_save", [False, True])
def test_keyboard_discard_opens_and_resets_before_waiting_for_start(session, monkeypatch, async_save):
    import time

    session.robot.outcomes = [False, False, False]
    keys = SimpleNamespace(space="space", enter="enter", right="right", left="left", esc="esc")
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=SimpleNamespace(Key=keys)))
    monkeypatch.setattr(recording, "is_headless", lambda: False)
    callbacks = {}
    calls = []
    wait_ticks = 0
    original_reset = session.robot.reset_to_initial
    original_add = Dataset.add_frame

    def init_listener(events, on_press, on_release):
        callbacks.update(press=on_press, release=on_release)
        on_press(keys.space)
        return SimpleNamespace(stop=lambda: None), events

    def add_frame(dataset, frame):
        original_add(dataset, frame)
        if session.robot.attempt == 1 and dataset.episode_buffer["size"] == 2:
            callbacks["press"](keys.left)

    def open_gripper():
        dataset = session.dataset
        assert dataset.episode_buffer["episode_index"] == 1
        assert dataset.episode_buffer["size"] == 0
        assert not dataset._get_image_file_dir(1, "observation.images.photon").exists()
        assert dataset.saved[0]["size"] == 2
        calls.append("open")

    def reset():
        calls.append("reset")
        original_reset()

    def main_loop_sleep(_duration):
        nonlocal wait_ticks
        if session.robot.attempt == 2:
            assert calls == ["reset", "reset", "open", "reset"]
            assert session.robot.observation_count == 0
            assert session.dataset.episode_buffer["size"] == 0
            wait_ticks += 1
            if wait_ticks == 3:
                callbacks["release"](keys.space)
            elif wait_ticks == 4:
                callbacks["press"](keys.space)
            elif wait_ticks > 4:
                pytest.fail("discarded episode did not restart on a fresh Space press")

    monkeypatch.setattr(recording, "init_keyboard_listener", init_listener)
    monkeypatch.setattr(session.robot, "open_gripper", open_gripper, raising=False)
    monkeypatch.setattr(session.robot, "reset_to_initial", reset)
    monkeypatch.setattr(Dataset, "add_frame", add_frame)
    monkeypatch.setattr(recording, "time", SimpleNamespace(
        sleep=main_loop_sleep, perf_counter=time.perf_counter,
        perf_counter_ns=time.perf_counter_ns, time=time.time,
    ))
    dataset = recording.record(session.cfg, async_save=async_save)
    assert wait_ticks == 4
    assert calls == ["reset", "reset", "open", "reset"]
    assert [row["episode_index"] for row in dataset.saved] == [0, 1]
    assert dataset.saved[1]["frames"][0]["observation.state"].tolist() == [2.0]
    assert session.retry_prompts == 0


def test_timeout_with_exit_request_still_discards_current_episode(session, monkeypatch):
    original_observation = session.robot.get_observation

    def observation():
        try:
            return original_observation()
        except TimeoutError:
            session.dataset.events["stop_recording"] = True
            raise

    monkeypatch.setattr(session.robot, "get_observation", observation)
    dataset = recording.record(session.cfg)
    assert dataset.num_episodes == 1
    assert dataset.episode_buffer["size"] == 0
    assert session.retry_prompts == 0
    assert not dataset._get_image_file_dir(1, "observation.images.photon").exists()


def test_timeout_while_preparing_next_episode_is_recoverable(session, monkeypatch):
    original_prepare = recording._prepare_recording_episode

    def prepare(robot, teleop, is_uf_teleop, manual_mode, **kwargs):
        original_prepare(robot, teleop, is_uf_teleop, manual_mode, **kwargs)
        if robot.outcomes[robot.attempt]:
            raise TimeoutError("no synchronized observation while enabling teleop")

    monkeypatch.setattr(recording, "_prepare_recording_episode", prepare)
    dataset = recording.record(session.cfg)
    assert dataset.num_episodes == 2
    assert session.retry_prompts == 2
    assert dataset.saved[1]["frames"][0]["observation.state"].tolist() == [3.0]


def test_discard_zero_frame_episode_waits_for_partial_image_writes(session):
    dataset = Dataset(session.cfg.dataset.root, {"rgb": {"dtype": "image"}}, 200)
    session.dataset = dataset
    directory = dataset._get_image_file_dir(0, "rgb")
    directory.mkdir(parents=True)
    dataset.image_writer.save_image(
        np.ones((2, 3, 3), dtype=np.uint8), directory / "frame-000000.png"
    )
    recording._discard_current_episode(dataset)
    assert not directory.exists()
    assert dataset.episode_buffer["episode_index"] == 0
    assert dataset.episode_buffer["size"] == 0
    assert not dataset.image_writer._stopped


@pytest.mark.parametrize("async_save", [False, True])
def test_device_error_preserves_completed_offline_episode(session, async_save):
    session.cfg.offline_mesh3dflow = True
    session.cfg.dataset.video = True
    session.robot.failure = RuntimeError
    with pytest.raises(RuntimeError, match="camera synchronization failed"):
        recording.record(session.cfg, async_save=async_save)
    assert session.dataset.num_episodes == 1
    assert session.mesh_episodes == [0]
    assert session.dataset.saved[0]["size"] == 2


@pytest.mark.parametrize("async_save", [False, True])
def test_mesh_failure_stops_save_and_preserves_previous_episode(session, monkeypatch, async_save):
    from lerobot_robot_ufactory.datasets import deferred_mesh as deferred

    session.cfg.offline_mesh3dflow = True
    session.cfg.dataset.video = True
    session.robot.outcomes = [False, False]
    compute_mesh = deferred.compute_episode_mesh

    def fail_second(dataset, cameras, runtime_dir, episode_index, **kwargs):
        if episode_index == 1:
            raise RuntimeError("mesh inference failed")
        return compute_mesh(dataset, cameras, runtime_dir, episode_index, **kwargs)

    monkeypatch.setattr(deferred, "compute_episode_mesh", fail_second)
    message = "Async episode save failed" if async_save else "mesh inference failed"
    with pytest.raises(RuntimeError, match=message):
        recording.record(session.cfg, async_save=async_save)
    assert session.dataset.num_episodes == 1
    assert session.mesh_episodes == [0]
    assert session.dataset.finalized
    assert not session.robot._is_connected


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("camera_fps", [15, 60])
@pytest.mark.parametrize("offline", [False, True])
def test_native_tactile_video_survives_real_record_and_retry(
    session, async_save, camera_fps, offline
):
    from dataclasses import replace
    camera = session.robot.cameras["photon"]
    camera.fps = camera_fps
    camera.samples_between = lambda start, end, wait=0: (
        tuple(replace(sample, frame_bgr=np.full((64, 64, 3), 7, dtype=np.uint8)) for sample in session.robot.get_tactile_samples_between(start, end)["photon"])
    )
    session.cfg.dataset.fps = 15
    session.cfg.offline_mesh3dflow = offline
    session.cfg.dataset.video = True
    dataset = recording.record(session.cfg, async_save=async_save)
    assert dataset.fps == 15
    assert dataset.features["observation.images.photon"]["dtype"] == "video"
    assert dataset.num_episodes == 2
    for index in range(2):
        root = dataset.root / f"tactile_streams/photon/episode_{index:06d}"
        import pyarrow.parquet as pq
        rows = pq.read_table(root / "samples.parquet").to_pylist()
        assert len(rows) == 2
        assert all(row["camera_fps"] == camera_fps for row in rows)
        assert all((dataset.root / row["video_path"]).is_file() for row in rows)
    assert len(list(dataset.root.rglob("*.mp4"))) == 2


@pytest.mark.parametrize("legacy_plan", [None, {"dataset_fps": 15, "cameras": {}}])
def test_legacy_dataset_resume_is_rejected_without_changing_files(session, legacy_plan):
    import hashlib
    import json

    root = session.cfg.dataset.root
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes": 0}')
    (root / "original.png").write_bytes(b"original dataset sentinel")
    if legacy_plan is not None:
        (root / "meta/camera_streams.json").write_text(json.dumps(legacy_plan))
    snapshot = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    camera = session.robot.cameras["photon"]
    camera.fps = 60
    camera.samples_between = lambda *args: ()
    session.cfg.resume = True
    session.cfg.dataset.video = True
    session.cfg.dataset.fps = 15
    with pytest.raises(ValueError, match="preserve it and use a new dataset root"):
        recording.record(session.cfg)
    assert not session.robot._is_connected
    assert session.dataset is None
    assert snapshot == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("online_requested", [False, True])
def test_native_mesh_action_rows_contain_only_ranges(session, monkeypatch, async_save, online_requested):
    from dataclasses import replace
    import pyarrow.parquet as pq
    from lerobot_robot_ufactory.datasets import deferred_mesh as deferred

    camera = session.robot.cameras["photon"]
    camera.fps = 60
    camera.config = SimpleNamespace(motion_3d_output="Mesh3DFlow")
    camera.samples_between = lambda start, end, wait=0: tuple(
        replace(sample, frame_bgr=np.full((64, 64, 3), 7, dtype=np.uint8))
        for sample in session.robot.get_tactile_samples_between(start, end)["photon"]
    )
    camera.compute_deferred_features = lambda image, runtime: {
        "mesh_motion_3d": np.full((2, 2, 3), 1.234567, dtype=np.float32)
    }
    monkeypatch.setattr(deferred, "compute_episode_mesh", real_compute_episode_mesh)
    session.cfg.dataset.fps = 15
    session.cfg.offline_mesh3dflow = False
    sensor_config = session.cfg.robot.cameras["photon"]
    sensor_config.save_marker_motion_3d = online_requested
    sensor_config.disable_infer = not online_requested
    session.cfg.dataset.video = True
    dataset = recording.record(session.cfg, async_save=async_save)
    assert session.cfg.offline_mesh3dflow
    assert sensor_config.disable_infer and not sensor_config.save_marker_motion_3d
    key = "observation.photon.tactile_range"
    assert dataset.features[key]["dtype"] == "int64"
    assert "observation.photon.mesh_motion_3d" not in dataset.features
    for index, buffer in enumerate(dataset.saved):
        assert np.asarray(buffer[key]).dtype == np.int64
        assert np.asarray(buffer[key]).tolist() == [[0, 1], [1, 2]]
        assert "observation.photon.mesh_motion_3d" not in buffer
        stream = dataset.root / f"tactile_streams/photon/episode_{index:06d}"
        values = np.load(stream / "mesh3dflow.npy", allow_pickle=False)
        assert values.shape == (2, 2, 2, 3)
        rows = pq.read_table(stream / "samples.parquet").to_pylist()
        assert [row["mesh_frame_index"] for row in rows] == [0, 1]


@pytest.mark.parametrize("async_save", [False, True])
def test_record_seeds_prestart_frame_once_per_episode_and_retry(session, monkeypatch, async_save):
    import time
    import json
    import pyarrow.parquet as pq
    from lerobot_robot_ufactory.datasets import deferred_mesh as deferred

    camera = session.robot.cameras["photon"]
    camera.fps = 60
    camera.config = SimpleNamespace(motion_3d_output="Mesh3DFlow")
    camera.samples_between = lambda *args: ()
    camera.sync_samples = lambda: (camera.seed,)
    camera.compute_deferred_features = lambda image, runtime: {
        "mesh_motion_3d": np.full((2, 2, 3), 1.234567, dtype=np.float32)
    }
    original_observation = session.robot.get_observation

    def observation():
        value = original_observation()
        if session.robot.observation_count == 1:
            captured = time.perf_counter() - 0.001
            camera.seed = XensePhotonSample(
                frame_bgr=np.full((64, 64, 3), 7, dtype=np.uint8),
                capture_monotonic_s=captured, capture_monotonic_ns=round(captured * 1e9),
                sensor_timestamp_s=None, marker_motion_3d=None,
            )
        session.robot._last_observation_sync_timing = {"camera": {"photon": {
            "capture_monotonic_s": camera.seed.capture_monotonic_s,
            "capture_monotonic_ns": camera.seed.capture_monotonic_ns,
        }}}
        return value

    monkeypatch.setattr(session.robot, "get_observation", observation)
    monkeypatch.setattr(deferred, "compute_episode_mesh", real_compute_episode_mesh)
    session.cfg.dataset.fps = 15
    session.cfg.dataset.video = True
    dataset = recording.record(session.cfg, async_save=async_save)
    assert dataset.num_episodes == 2
    for episode, buffer in enumerate(dataset.saved):
        assert np.asarray(buffer["observation.photon.tactile_range"]).tolist() == [[1, 1], [1, 1]]
        stream = dataset.root / f"tactile_streams/photon/episode_{episode:06d}"
        assert pq.read_table(stream / "samples.parquet").num_rows == 1
        assert np.load(stream / "mesh3dflow.npy").shape[0] == 1
        timing = pq.read_table(dataset.root / f"timestamps/episode_{episode:06d}.parquet").to_pylist()
        for row in timing:
            interval = json.loads(row["camera_intervals_json"])["photon"]
            assert interval["frame_count"] == 0
            assert interval["representative_tactile_index"] == 0
