"""Exercise timeout recovery through the recorder without connected hardware."""

import copy
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.datasets.image_writer import AsyncImageWriter

from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording
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
    import lerobot_robot_ufactory.tactile.deferred as deferred

    class Camera:
        deferred_feature_shapes = {}

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
        defer_processing=False,
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
                assert dataset._get_image_file_dir(0, "observation.images.photon").exists()
                staging = dataset.root / "tactile_streams/.staging"
                assert not staging.exists() or not any(staging.iterdir())
                assert (dataset.root / "raw_episodes/episode_000000/manifest.json").exists()
                assert (
                    dataset.root / "tactile_streams/photon/episode_000000/samples.parquet"
                ).exists()
            if state.stop_after_timeout:
                dataset.events["stop_recording"] = True
        return ""

    def compute_mesh(dataset, cameras, runtime_dir, episode_index, **kwargs):
        state.mesh_episodes.append(episode_index)

    def postprocess(dataset, cameras):
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


@pytest.mark.parametrize("mode", ["sync", "async", "offline", "raw", "raw_async"])
@pytest.mark.parametrize("failure_frames", [0, 1])
@pytest.mark.parametrize("stop_after_timeout", [False, True])
def test_timeout_preserves_previous_episode(
    session, mode, failure_frames, stop_after_timeout, caplog
):
    session.cfg.offline_mesh3dflow = mode == "offline"
    session.cfg.defer_processing = mode in ("raw", "raw_async")
    session.cfg.dataset.video = mode == "offline"
    session.robot.failure_frames = failure_frames
    session.stop_after_timeout = stop_after_timeout

    dataset = recording.record(session.cfg, async_save=mode in ("async", "raw_async"))

    expected = 1 if stop_after_timeout else 2
    assert dataset.num_episodes == expected
    assert [row["episode_index"] for row in dataset.saved] == list(range(expected))
    assert [row["size"] for row in dataset.saved] == [2] * expected
    assert session.retry_prompts == (1 if stop_after_timeout else 2)
    assert dataset.saved[0]["frames"][0]["observation.state"].tolist() == [0.0]
    if not stop_after_timeout:
        assert dataset.saved[1]["frames"][0]["observation.state"].tolist() == [3.0]
    assert session.mesh_episodes == (list(range(expected)) if mode == "offline" else [])
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
        if session.robot.attempt == 1:
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

    def prepare(robot, teleop, is_uf_teleop, manual_mode):
        original_prepare(robot, teleop, is_uf_teleop, manual_mode)
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
def test_device_error_leaves_completed_raw_episode_recoverable(session, async_save):
    session.cfg.defer_processing = True
    session.robot.failure = RuntimeError
    with pytest.raises(RuntimeError, match="camera synchronization failed"):
        recording.record(session.cfg, async_save=async_save)
    store = recording.RawEpisodeStore(session.dataset)
    paths = store.checkpoints(session.dataset.root)
    assert len(paths) == 1
    buffer, metadata = store.load(paths[0])
    assert metadata["episode_index"] == 0
    assert buffer["size"] == 2
    assert session.dataset.num_episodes == 0  # No video/inference attempted.
    np.testing.assert_array_equal(buffer["observation.state"], [[0.0], [0.0]])
