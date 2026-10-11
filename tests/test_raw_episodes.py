import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_ufactory.scripts.uf_lerobot_record import (
    EpisodeSynchronization,
    _prepare_dataset_root,
    _RawDatasetFinalize,
)
from lerobot_robot_ufactory.datasets import raw_episodes
from lerobot_robot_ufactory.datasets.raw_episodes import (
    RawEpisodeStore,
    open_recording_dataset,
    postprocess_raw_episodes,
)


@pytest.fixture
def dataset(tmp_path, monkeypatch, request):
    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    use_videos = getattr(request, "param", True)
    dataset = LeRobotDataset.create(
        "test/raw-episodes",
        fps=30,
        root=tmp_path / "dataset",
        use_videos=use_videos,
        vcodec="h264",
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "observation.images.photon": {
                "dtype": "video" if use_videos else "image",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.photon.mesh_motion_3d": {
                "dtype": "float32",
                "shape": (2, 2, 3),
                "names": None,
            },
        },
        image_writer_threads=1,
    )
    yield dataset
    dataset.stop_image_writer()
    dataset.finalize()


def checkpoint(dataset, index=0, *, mesh=False):
    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=index)
    for i in range(3):
        dataset.add_frame(
            {
                "action": np.array([i, -i], dtype=np.float32),
                "observation.state": np.array([i + 0.5, i + 1], dtype=np.float32),
                "observation.images.photon": np.full((64, 64, 3), i * 20, dtype=np.uint8),
                "observation.photon.mesh_motion_3d": np.full(
                    (2, 2, 3), np.nan if mesh else i, dtype=np.float32
                ),
                "task": "test raw episode",
            }
        )
    runtime = dataset.root / "runtime/session-test"
    runtime.mkdir(parents=True, exist_ok=True)
    store = RawEpisodeStore(
        dataset,
        runtime_dir=runtime,
        offline_mesh_fields={"observation.photon.mesh_motion_3d": (2, 2, 3)} if mesh else {},
    )
    sync = EpisodeSynchronization(None, dataset.fps)
    for i in range(3):
        sync.add_frame(i, 1 + i / 30, None, action_send_start_s=1 + i / 30)
    path = store.save(dataset.episode_buffer, sync)
    return store, path


def test_checkpoint_survives_exception_and_reopens_without_encoding(dataset):
    with pytest.raises(RuntimeError, match="device disconnected"):
        with _RawDatasetFinalize(dataset):
            store, manifest = checkpoint(dataset)
            assert dataset.num_episodes == 0
            assert not list((dataset.root / "videos").rglob("*.mp4"))
            raise RuntimeError("device disconnected")
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=dataset.root), resume=True)
    _prepare_dataset_root(cfg)
    reopened = open_recording_dataset(dataset.repo_id, dataset.root, batch_encoding_size=1)
    restored, metadata = RawEpisodeStore(reopened).load(manifest)
    assert store.next_episode_index() == 1
    assert metadata["size"] == 3
    np.testing.assert_array_equal(restored["action"], [[0, 0], [1, -1], [2, -2]])
    np.testing.assert_array_equal(restored["observation.state"], [[0.5, 1], [1.5, 2], [2.5, 3]])
    assert len(restored["observation.images.photon"]) == 3
    assert (dataset.root / "timestamps/episode_000000.parquet").exists()
    reopened.finalize()


def test_checkpoint_images_survive_source_directory_cleanup(dataset):
    import shutil

    store, manifest = checkpoint(dataset)
    dataset.stop_image_writer()
    shutil.rmtree(dataset.root / "images")
    restored, metadata = store.load(manifest)
    assert metadata["size"] == 3
    from PIL import Image
    for index, path in enumerate(restored["observation.images.photon"]):
        np.testing.assert_array_equal(
            np.asarray(Image.open(path)), np.full((64, 64, 3), index * 20, dtype=np.uint8)
        )


def test_deferred_web_resume_counts_pending_checkpoints(dataset, monkeypatch):
    from lerobot_robot_ufactory.utils.webapp.recording_web_config import dataset_status
    from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording

    checkpoint(dataset)
    checkpoint(dataset, 1)
    monkeypatch.setattr(recording, "_missing_dataset_files", lambda root: [])
    raw = {"dataset": {"root": str(dataset.root)}, "defer_processing": True}
    status = dataset_status(dataset.root.parent, raw)
    assert status["resumable"]
    assert status["episodes"] == 2
    raw["defer_processing"] = False
    assert not dataset_status(dataset.root.parent, raw)["resumable"]
    raw["defer_processing"] = True
    monkeypatch.setattr(recording, "_missing_dataset_files", lambda root: ["data/file.parquet"])
    assert not dataset_status(dataset.root.parent, raw)["resumable"]


def test_web_postprocess_detection_counts_pending_and_rejects_incompatible_data(dataset):
    from lerobot_robot_ufactory.utils.webapp.recording_web_config import postprocess_status

    raw = {"dataset": {"root": str(dataset.root), "fps": dataset.fps}}
    assert not postprocess_status(dataset.root.parent, raw)["ready"]
    checkpoint(dataset)
    status = postprocess_status(dataset.root.parent, raw)
    assert status["ready"] and status["processed_episodes"] == 0
    assert status["pending_episodes"] == 1 and status["pending_frames"] == 3
    raw["dataset"]["fps"] += 1
    status = postprocess_status(dataset.root.parent, raw)
    assert not status["ready"] and "FPS" in status["reason"]
    raw["dataset"]["fps"] = dataset.fps
    checkpoint(dataset, 2)
    status = postprocess_status(dataset.root.parent, raw)
    assert not status["ready"] and "contiguous" in status["reason"]


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("processed_first", [False, True])
def test_immediate_resume_rejects_pending_raw_before_creating_robot(
    dataset, monkeypatch, resume, processed_first
):
    from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording

    checkpoint(dataset)
    if processed_first:
        dataset = postprocess_raw_episodes(dataset, {})
        checkpoint(dataset, 1)
    dataset.stop_image_writer()
    dataset.finalize()
    originals = {
        path: path.read_bytes() for path in dataset.root.rglob("*") if path.is_file()
    }
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(root=dataset.root),
        resume=resume,
        offline_mesh3dflow=False,
        display_data=False,
        robot=None,
    )
    monkeypatch.setattr(recording, "asdict", lambda cfg: {})
    monkeypatch.setattr(recording, "init_logging", lambda: None)
    monkeypatch.setattr(recording.sys.stdin, "isatty", lambda: False)

    def unexpected_robot(_cfg):
        pytest.fail("Must reject unsafe resume before creating recording devices")

    monkeypatch.setattr(recording, "make_robot_from_config", unexpected_robot)
    with pytest.raises(RuntimeError, match=r"1 raw episode.*--postprocess-only"):
        recording.record(cfg)
    assert cfg.resume  # Also protect non-interactive automatic resume.
    assert originals == {
        path: path.read_bytes() for path in dataset.root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("mode", ["processed", "postprocess_only"])
def test_raw_resume_guard_allows_safe_modes(dataset, monkeypatch, mode):
    from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording

    checkpoint(dataset)
    if mode == "processed":
        dataset = postprocess_raw_episodes(dataset, {})
    dataset.stop_image_writer()
    dataset.finalize()
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(root=dataset.root),
        resume=True,
        offline_mesh3dflow=False,
        display_data=False,
        robot=None,
    )
    monkeypatch.setattr(recording, "asdict", lambda cfg: {})
    monkeypatch.setattr(recording, "init_logging", lambda: None)

    class GuardPassed(Exception):
        pass

    def robot_factory(_cfg):
        raise GuardPassed

    monkeypatch.setattr(recording, "make_robot_from_config", robot_factory)
    with pytest.raises(GuardPassed):
        recording.record(cfg, postprocess_only=mode == "postprocess_only")


@pytest.mark.parametrize("dataset", [True, False], indirect=True)
def test_normal_postprocessing_saves_all_episodes_and_removes_pngs(dataset):
    _, first = checkpoint(dataset)
    _, second = checkpoint(dataset, 1)
    progress = []
    result = postprocess_raw_episodes(dataset, {}, progress=progress.append)
    assert progress[0]["stage"] == "preparing"
    assert [p["episode_index"] for p in progress if p["stage"] == "loading"] == [0, 1]
    assert {p["stage"] for p in progress} >= {"loading", "saving", "publishing", "complete"}
    assert progress[-1]["total_episodes"] == progress[-1]["completed_episodes"] == 2
    assert progress[-1]["stage"] == "complete"
    assert all(p["elapsed_s"] >= 0 for p in progress)
    assert result.num_episodes == 2
    assert result.num_frames == 6
    assert len(list((result.root / "videos").rglob("*.mp4"))) == int(bool(result.meta.video_keys))
    assert not list((result.root / "images").rglob("*.png"))
    assert first.exists() and second.exists()
    assert postprocess_raw_episodes(result, {}) is result  # Already processed.
    values = result.hf_dataset.with_format("numpy")["action"]
    np.testing.assert_array_equal(values[:3], [[0, 0], [1, -1], [2, -2]])
    if not result.meta.video_keys:
        image = result.hf_dataset.with_format(None)[1]["observation.images.photon"]
        np.testing.assert_array_equal(np.asarray(image), np.full((64, 64, 3), 20, dtype=np.uint8))
    result.finalize()


def test_failed_conversion_keeps_original_dataset_and_can_retry(dataset, monkeypatch):
    _, first = checkpoint(dataset)
    completed = postprocess_raw_episodes(dataset, {})
    old_info = (completed.root / "meta/info.json").read_bytes()
    _, second = checkpoint(completed, 1)
    original_save = LeRobotDataset.save_episode

    def fail_save(self, *args, **kwargs):
        original_save(self, *args, **kwargs)
        raise RuntimeError("encoding failed after writing output")

    monkeypatch.setattr(LeRobotDataset, "save_episode", fail_save)
    progress = []
    with pytest.raises(RuntimeError, match="encoding failed"):
        postprocess_raw_episodes(completed, {}, progress=progress.append)
    assert progress[-1]["stage"] == "failed"
    assert "encoding failed" in progress[-1]["error"]
    assert progress[-1]["completed_episodes"] == 0
    from lerobot_robot_ufactory.utils.webapp.recording_web_config import postprocess_status
    status = postprocess_status(completed.root.parent,
                                {"dataset": {"root": str(completed.root), "fps": completed.fps}})
    assert status["ready"] and status["pending_episodes"] == 1
    assert (completed.root / "meta/info.json").read_bytes() == old_info
    assert first.exists() and second.exists()
    assert len(list((completed.root / "images").rglob("*.png"))) == 3
    monkeypatch.setattr(LeRobotDataset, "save_episode", original_save)
    result = postprocess_raw_episodes(completed, {})
    assert result.num_episodes == 2
    assert not list((result.root / "images").rglob("*.png"))
    result.finalize()
    completed.finalize()


def test_interrupted_png_cleanup_keeps_published_data_and_can_resume(dataset, monkeypatch):
    checkpoint(dataset)
    checkpoint(dataset, 1)
    active = dataset._get_image_file_path(2, "observation.images.photon", 0)
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(b"active episode")
    tactile = dataset.root / "tactile_streams/photon/episode_000000/frames/frame_000000.png"
    tactile.parent.mkdir(parents=True)
    tactile.write_bytes(b"full-rate tactile sample")
    original_discard = raw_episodes.discard_episode_images

    def interrupt_cleanup(published, index):
        assert published.num_episodes == 2
        assert list((published.root / "videos").rglob("*.mp4"))
        if index == 1:
            raise OSError("cleanup interrupted")
        original_discard(published, index)

    monkeypatch.setattr(raw_episodes, "discard_episode_images", interrupt_cleanup)
    with pytest.raises(OSError, match="cleanup interrupted"):
        postprocess_raw_episodes(dataset, {})
    reopened = open_recording_dataset(dataset.repo_id, dataset.root, vcodec="h264")
    assert reopened.num_frames == 6
    assert len(list((reopened.root / "images").rglob("*.png"))) == 4
    monkeypatch.setattr(raw_episodes, "discard_episode_images", original_discard)
    assert postprocess_raw_episodes(reopened, {}) is reopened
    assert list((reopened.root / "images").rglob("*.png")) == [active]
    assert tactile.read_bytes() == b"full-rate tactile sample"
    reopened.finalize()


def test_dataset_reopen_failure_preserves_pngs_until_retry(dataset, monkeypatch):
    checkpoint(dataset)
    original_init = LeRobotDataset.__init__

    def fail_reopen(self, *args, **kwargs):
        if kwargs.get("root") == dataset.root:
            raise RuntimeError("cannot reopen published dataset")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(LeRobotDataset, "__init__", fail_reopen)
    with pytest.raises(RuntimeError, match="cannot reopen"):
        postprocess_raw_episodes(dataset, {})
    assert len(list((dataset.root / "images").rglob("*.png"))) == 3
    monkeypatch.setattr(LeRobotDataset, "__init__", original_init)
    reopened = open_recording_dataset(dataset.repo_id, dataset.root, vcodec="h264")
    assert reopened.num_frames == 3
    postprocess_raw_episodes(reopened, {})
    assert not list((dataset.root / "images").rglob("*.png"))
    reopened.finalize()


def test_mesh_fields_are_computed_from_checkpoint_after_recording(dataset):
    _, path = checkpoint(dataset, mesh=True)
    assert (
        "observation.photon.mesh_motion_3d"
        not in pq.read_table(path.parent / "frames.parquet").column_names
    )
    calls = []

    class Camera:
        deferred_feature_shapes = {"mesh_motion_3d": (2, 2, 3)}

        @contextmanager
        def deferred_session(self, runtime):
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        def compute_deferred_features(self, image, runtime):
            assert runtime == dataset.root / "runtime/session-test"
            return {"mesh_motion_3d": np.full((2, 2, 3), 7, dtype=np.float32)}

    result = postprocess_raw_episodes(dataset, {"photon": Camera()})
    assert calls.count("enter") == calls.count("exit")
    values = result.hf_dataset.with_format("numpy")["observation.photon.mesh_motion_3d"]
    np.testing.assert_array_equal(values, np.full((3, 2, 2, 3), 7, dtype=np.float32))
    assert json.loads(path.read_text())["offline_mesh_fields"]
    assert not list((result.root / "images").rglob("*.png"))
    result.finalize()


def test_publish_failure_rolls_back_converted_directories(dataset, monkeypatch):
    _, path = checkpoint(dataset)
    original_info = (dataset.root / "meta/info.json").read_bytes()
    replace = raw_episodes.os.replace

    def fail_publish(source, destination):
        if destination == dataset.root / "meta" and source.parent.name == "dataset":
            raise OSError("failed to publish metadata")
        return replace(source, destination)

    monkeypatch.setattr(raw_episodes.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish metadata"):
        postprocess_raw_episodes(dataset, {})
    assert (dataset.root / "meta/info.json").read_bytes() == original_info
    assert not (dataset.root / "data").exists()
    assert not (dataset.root / "videos").exists()
    assert path.exists()
    assert len(list((dataset.root / "images").rglob("*.png"))) == 3
    assert not (dataset.root / ".raw_postprocessing_transaction.json").exists()


def test_interrupted_publication_is_recovered_on_startup(tmp_path):
    root = tmp_path / "dataset"
    work = root / "raw_episodes/.processing_test"
    output, backup = work / "dataset/meta", work / "backup/meta"
    backup.mkdir(parents=True)
    (backup / "old.json").write_text("previous dataset")
    (root / "meta").mkdir()
    (root / "meta/new.json").write_text("partial conversion")
    journal = {
        "work": str(work.relative_to(root)),
        "items": [
            {
                "source": str(output.relative_to(root)),
                "destination": "meta",
                "backup": str(backup.relative_to(root)),
                "had_destination": True,
            }
        ],
    }
    (root / ".raw_postprocessing_transaction.json").write_text(json.dumps(journal))
    raw_episodes.recover_postprocessing(root)
    assert (root / "meta/old.json").read_text() == "previous dataset"
    assert not (root / "meta/new.json").exists()
    assert not work.exists()


def test_checkpoint_rejects_image_path_outside_dataset(dataset, tmp_path):
    store, path = checkpoint(dataset)
    table = pq.read_table(path.parent / "frames.parquet").to_pydict()
    table["observation.images.photon"][0] = "../../outside.png"
    import pyarrow as pa

    pq.write_table(pa.table(table), path.parent / "frames.parquet")
    with pytest.raises(ValueError, match="outside the dataset"):
        store.load(path)


def test_missing_offline_camera_keeps_raw_episode_for_retry(dataset):
    store, path = checkpoint(dataset, mesh=True)
    original_info = (dataset.root / "meta/info.json").read_bytes()
    with pytest.raises(ValueError, match="Missing or incompatible tactile"):
        postprocess_raw_episodes(dataset, {})
    assert (dataset.root / "meta/info.json").read_bytes() == original_info
    assert store.load(path)[0]["size"] == 3


@pytest.mark.parametrize(
    ("fail_recording", "recovery_mode"),
    [(False, None), (True, "postprocess_only"), (True, "resume")],
)
def test_recording_saves_video_before_next_episode_and_survives_failure(
    tmp_path, monkeypatch, fail_recording, recovery_mode
):
    import datasets.config

    from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    state = SimpleNamespace(events=None, connections=0, recovering=False)

    class Robot:
        name = robot_type = "test_record_robot"
        action_features = {"J1.pos": float, "J2.pos": float}
        observation_features = {**action_features, "rgb": (64, 64, 3)}
        cameras = {"rgb": object()}
        _is_connected = False

        def connect(self):
            state.connections += 1
            self._is_connected = True

        def disconnect(self):
            self._is_connected = False

        def reset_to_initial(self):
            pass

        def get_observation(self):
            return {"J1.pos": 1.0, "J2.pos": 2.0, "rgb": np.zeros((64, 64, 3), dtype=np.uint8)}

        def send_action(self, action):
            return action

    robot = Robot()
    root = tmp_path / "recording"
    cfg = SimpleNamespace(
        robot=SimpleNamespace(manual_mode=True, cameras=robot.cameras),
        teleop=None,
        policy=None,
        resume=False,
        offline_mesh3dflow=False,
        web_preview=recording.WebPreviewConfig(),
        synchronize=True,
        play_sounds=False,
        display_data=False,
        dataset=SimpleNamespace(
            repo_id="test/raw-recording",
            root=root,
            fps=30,
            video=True,
            video_encoding_batch_size=8,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=1,
            num_episodes=3,
            episode_time_s=0.001,
            single_task="test",
            push_to_hub=False,
        ),
    )
    original_loop = recording.record_loop
    original_create = LeRobotDataset.create

    def create(*args, **kwargs):
        assert kwargs["batch_encoding_size"] == 1
        kwargs["vcodec"] = "h264"
        return original_create(*args, **kwargs)

    def loop(**kwargs):
        state.events = kwargs["events"]
        return original_loop(**kwargs)

    def prompt(message):
        if "next episode" in message:
            assert not (root / "raw_episodes").exists()
            assert not list((root / "images").rglob("*.png"))
            assert list((root / "videos").rglob("*.mp4"))
            if fail_recording and not state.recovering:
                raise RuntimeError("GELLO disconnected")
            state.events["stop_recording"] = True  # Same event as Esc.
        return ""

    monkeypatch.setattr(recording, "asdict", lambda cfg: {})
    monkeypatch.setattr(recording, "init_logging", lambda: None)
    monkeypatch.setattr(recording, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(recording, "is_headless", lambda: True)
    monkeypatch.setattr(recording, "record_loop", loop)
    monkeypatch.setattr(LeRobotDataset, "create", create)
    monkeypatch.setattr("builtins.input", prompt)

    if fail_recording:
        with pytest.raises(RuntimeError, match="GELLO disconnected"):
            recording.record(cfg)
        assert list((root / "videos").rglob("*.mp4"))
        state.recovering = True
        if recovery_mode == "resume":
            cfg.resume = True
            result = recording.record(cfg)
        else:
            result = recording.record(cfg, postprocess_only=True)
    else:
        result = recording.record(cfg)
    expected = 2 if recovery_mode == "resume" else 1
    assert result.num_episodes == expected
    assert result.num_frames == expected
    assert state.connections == expected  # Postprocess-only never connects hardware.
    assert not list((root / "images").rglob("*.png"))
    assert (root / "timestamps/episode_000000.parquet").exists()
    assert list((root / "videos").rglob("*.mp4"))
    result.finalize()
