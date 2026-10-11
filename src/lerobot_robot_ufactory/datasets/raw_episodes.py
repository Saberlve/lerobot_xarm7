"""Checkpoint complete raw episodes before any video encoding or inference."""

import copy
import errno
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .episode_images import discard_episode_images, validate_episode_images


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _within_root(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Raw episode path is outside the dataset: {relative}")
    return path


class RawEpisodeStore:
    def __init__(self, dataset, *, runtime_dir=None, offline_mesh_fields=None):
        self.dataset = dataset
        self.root = Path(dataset.root).resolve()
        self.directory = self.root / "raw_episodes"
        self.runtime_dir = runtime_dir
        self.offline_mesh_fields = offline_mesh_fields or {}

    @staticmethod
    def checkpoints(root):
        return sorted(
            (Path(root) / "raw_episodes").glob(
                "episode_[0-9][0-9][0-9][0-9][0-9][0-9]/manifest.json"
            )
        )

    def next_episode_index(self):
        indices = [
            json.loads(path.read_text())["episode_index"] for path in self.checkpoints(self.root)
        ]
        return max([self.dataset.num_episodes - 1, *indices]) + 1

    def save(self, buffer, synchronization=None):
        """Publish numeric data, image references and sidecars as one raw episode."""
        size = int(buffer["size"])
        index = int(buffer["episode_index"])
        if size <= 0:
            raise ValueError("Cannot checkpoint an empty episode")
        destination = self.directory / f"episode_{index:06d}"
        if destination.exists():
            raise FileExistsError(f"Raw episode already saved: {destination}")
        validate_episode_images(self.dataset, buffer)
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=self.directory))
        try:
            columns = {}
            for key in [*self.dataset.features, "task"]:
                if key == "episode_index" or key in self.offline_mesh_fields:
                    continue
                values = buffer.get(key, [])
                if len(values) == 0:
                    continue  # LeRobot generates global indices during conversion.
                if len(values) != size:
                    raise ValueError(f"Raw episode {index}: incorrect frame count for {key}")
                feature = self.dataset.features.get(key, {})
                if feature.get("dtype") in ("image", "video"):
                    retained = []
                    for frame_index, path in enumerate(values):
                        source = Path(path).resolve()
                        source.relative_to(self.root)
                        relative = Path("images") / key / f"frame-{frame_index:06d}.png"
                        target = staging / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source, target)
                        retained.append(str((destination / relative).relative_to(self.root)))
                    values = retained
                elif key != "task":
                    values = np.asarray(values).tolist()
                columns[key] = pa.array(values)
            if len(columns.get("task", [])) != size:
                raise ValueError("Raw episode must include one task per frame")
            data_path = staging / "frames.parquet"
            pq.write_table(pa.table(columns), data_path)
            with data_path.open("rb") as stream:
                os.fsync(stream.fileno())
            manifest = {
                "version": 1,
                "episode_index": index,
                "size": size,
                "fps": self.dataset.fps,
                "features": self.dataset.features,
                "offline_mesh_fields": self.offline_mesh_fields,
                "runtime_dir": (
                    None
                    if self.runtime_dir is None
                    else str(Path(self.runtime_dir).resolve().relative_to(self.root))
                ),
            }
            recorder = getattr(synchronization, "tactile_recorder", None)
            if recorder is not None:
                manifest["native_streams"] = {
                    "names": list(recorder.stream_names),
                    "fps": recorder.stream_fps,
                    "videos": recorder.video_streams,
                    "crf": recorder.video_crf,
                    "mesh": sorted(recorder.required_mesh_streams),
                }
                recorder.raw_checkpoint = True
            _write_json(staging / "manifest.json", manifest)
            if synchronization is not None:
                # Include the checkpoint in the existing tactile transaction so
                # startup recovery preserves either the whole episode or none.
                synchronization.write(self.root, index, extra_items=[(staging, destination)])
            else:
                os.replace(staging, destination)
            logging.info("[RawSaved] Episode %s: %s frames; video/inference deferred", index, size)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination / "manifest.json"

    def load(self, manifest_path):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest["version"] != 1 or manifest["fps"] != self.dataset.fps:
            raise ValueError(f"Incompatible raw episode: {manifest_path}")
        expected = json.loads(json.dumps(self.dataset.features))
        if manifest["features"] != expected:
            raise ValueError(f"Raw episode features do not match the dataset: {manifest_path}")
        index, size = manifest["episode_index"], manifest["size"]
        table = pq.read_table(manifest_path.parent / "frames.parquet")
        if table.num_rows != size:
            raise ValueError(f"Raw episode frame count mismatch: {manifest_path}")
        buffer = {key: [] for key in self.dataset.features}
        buffer.update(size=size, episode_index=index)
        for key in table.column_names:
            values = table[key].to_pylist()
            feature = self.dataset.features.get(key, {})
            if feature.get("dtype") in ("image", "video"):
                values = [str(_within_root(self.root, path)) for path in values]
                if any(not Path(path).is_file() for path in values):
                    raise FileNotFoundError(f"Missing raw images for episode {index}: {key}")
            elif key != "task":
                values = list(np.asarray(values, dtype=feature["dtype"]))
            buffer[key] = values
        for key, shape in manifest["offline_mesh_fields"].items():
            buffer[key] = [np.full(shape, np.nan, dtype=np.float32) for _ in range(size)]
        return buffer, manifest


def _rollback_publication(root, journal):
    for item in reversed(journal["items"]):
        source = _within_root(root, item["source"])
        destination = _within_root(root, item["destination"])
        backup = _within_root(root, item["backup"])
        if backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        elif not item["had_destination"] and not source.exists() and destination.exists():
            shutil.rmtree(destination)


def recover_postprocessing(root):
    """Roll back a conversion interrupted while replacing dataset directories."""
    root = Path(root).resolve()
    path = root / ".raw_postprocessing_transaction.json"
    if not path.exists():
        return
    journal = json.loads(path.read_text())
    _rollback_publication(root, journal)
    path.unlink()
    work = _within_root(root, journal["work"])
    if work.exists():
        shutil.rmtree(work)
    logging.warning("Recovered interrupted postprocessing; raw episodes are available for retry")


def open_recording_dataset(repo_id, root, **kwargs):
    """Open a raw-only recording locally without attempting a Hub download."""
    from .native_dataset import NativeRateLeRobotDataset as LeRobotDataset
    from lerobot.datasets.utils import load_info

    root = Path(root)
    info = load_info(root)
    if info["total_episodes"] or not RawEpisodeStore.checkpoints(root):
        return LeRobotDataset(repo_id, root=root, **kwargs)
    with tempfile.TemporaryDirectory(prefix=".template_", dir=root / "raw_episodes") as temporary:
        dataset = LeRobotDataset.create(
            repo_id,
            info["fps"],
            root=Path(temporary) / "dataset",
            features=info["features"],
            robot_type=info.get("robot_type"),
            use_videos=bool(info.get("video_path")),
            **kwargs,
        )
        dataset.root = root
        dataset.meta.root = root
        return dataset


def _publish_output(root, output, work):
    items = []
    for name in ("data", "videos", "meta", "tactile_streams"):
        source, destination, backup = output / name, root / name, work / "backup" / name
        if source.exists():
            items.append(
                {
                    "source": str(source.relative_to(root)),
                    "destination": str(destination.relative_to(root)),
                    "backup": str(backup.relative_to(root)),
                    "had_destination": destination.exists(),
                }
            )
    journal = {"work": str(work.relative_to(root)), "items": items}
    journal_path = root / ".raw_postprocessing_transaction.json"
    _write_json(journal_path, journal)
    try:
        for item in items:
            source, destination, backup = (
                root / item[key] for key in ("source", "destination", "backup")
            )
            if destination.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            os.replace(source, destination)
    except BaseException:
        _rollback_publication(root, journal)
        journal_path.unlink()
        raise
    journal_path.unlink()


def _cleanup_processed_images(dataset):
    """Remove temporary camera PNGs only for published, checkpointed episodes."""
    for path in RawEpisodeStore.checkpoints(dataset.root):
        index = json.loads(path.read_text())["episode_index"]
        if index < dataset.num_episodes:
            discard_episode_images(dataset, index)


def postprocess_raw_episodes(dataset, cameras, *, progress=None):
    """Keep raw inputs until conversion is published and can be reopened."""
    from .native_dataset import NativeRateLeRobotDataset as LeRobotDataset

    from lerobot_robot_ufactory.datasets.deferred_mesh import compute_episode_mesh, deferred_sessions

    store = RawEpisodeStore(dataset)
    pending = [
        path
        for path in store.checkpoints(store.root)
        if json.loads(path.read_text())["episode_index"] >= dataset.num_episodes
    ]
    started = time.monotonic()
    progress_lock = threading.RLock()
    state = {"total_episodes": len(pending), "completed_episodes": 0,
             "episode_index": None, "streams": {}, "stage": "preparing"}

    def report(stage=None, **updates):
        if progress is None:
            return
        with progress_lock:
            if stage is not None:
                state["stage"] = stage
            state.update(updates)
            progress({**state, "streams": {key: dict(value) for key, value in state["streams"].items()},
                      "elapsed_s": time.monotonic() - started})

    def stream_progress(update):
        with progress_lock:
            state["streams"][update["camera"]] = update
            report()

    report()
    if not pending:
        # Also finish cleanup interrupted after a previous successful publication.
        _cleanup_processed_images(dataset)
        report("complete")
        return dataset
    dataset._wait_image_writer()
    dataset.finalize()
    work = store.root / "raw_episodes" / f".processing_{uuid4().hex}"
    work.mkdir(parents=True)
    output = work / "dataset"
    converted = None
    try:
        if dataset.num_episodes:
            output.mkdir()
            for name in ("meta", "data", "videos"):
                if (store.root / name).exists():
                    shutil.copytree(store.root / name, output / name)
            converted = LeRobotDataset(
                dataset.repo_id, root=output, batch_encoding_size=1, vcodec=dataset.vcodec
            )
        else:
            converted = LeRobotDataset.create(
                dataset.repo_id,
                dataset.fps,
                root=output,
                features=copy.deepcopy(dataset.features),
                robot_type=dataset.meta.info.get("robot_type"),
                use_videos=bool(dataset.meta.video_keys),
                batch_encoding_size=1,
                vcodec=dataset.vcodec,
            )
        # Reuse each session's offline solvers across its episodes.
        # Private hard links preserve the complete native inputs on failure.
        for name in ("tactile_streams", "timestamps"):
            source = store.root / name
            if source.is_dir():
                shutil.copytree(source, output / name, copy_function=os.link)
        converted._native_stream_root = output
        converted._camera_stream_plan = getattr(dataset, "_camera_stream_plan", None)
        if converted._camera_stream_plan is None:
            plan_path = store.root / "meta/camera_streams.json"
            converted._camera_stream_plan = json.loads(plan_path.read_text())["cameras"] if plan_path.is_file() else {}
        active_runtime = None
        with ExitStack() as sessions:
            for ordinal, path in enumerate(pending, 1):
                report("loading", episode_index=json.loads(path.read_text())["episode_index"], streams={})
                buffer, manifest = store.load(path)
                index = manifest["episode_index"]
                if index != converted.num_episodes:
                    raise ValueError(
                        "Raw episodes must be contiguous; "
                        f"expected {converted.num_episodes}, got {index}"
                    )
                logging.info("[Postprocess] Episode %s (%s/%s)", index, ordinal, len(pending))
                native = manifest.get("native_streams")
                if native:
                    from .stream_recorder import TactileStreamRecorder

                    recorder = TactileStreamRecorder.__new__(TactileStreamRecorder)
                    recorder._staging_root = output / "tactile_streams"
                    # Methods operate on <staging>/<camera>, so use private links
                    # to this episode without creating capture threads.
                    stage = work / f"native_{index:06d}"
                    stage.mkdir()
                    for name in native["names"]:
                        shutil.copytree(output / "tactile_streams" / name / f"episode_{index:06d}",
                                        stage / name, copy_function=os.link)
                    recorder._staging_root = stage
                    recorder._base = f"episode_{index:06d}"
                    recorder.episode_index = index
                    recorder.stream_names = native["names"]
                    recorder.stream_fps = native["fps"]
                    recorder.video_streams = native["videos"]
                    recorder.video_crf = native["crf"]
                    recorder.required_mesh_streams = set(native["mesh"])
                    recorder._published = recorder._prepared = False
                    recorder._close_writer = lambda: None
                    recorder._rows = {
                        name: pq.read_table(stage / name / "samples.parquet").to_pylist()
                        for name in native["names"]
                    }
                    if native["mesh"]:
                        if manifest["runtime_dir"] is None:
                            raise ValueError("Raw native episode has no saved runtime")
                        report("mesh")
                        recorder.compute_mesh(
                            {name: cameras[name] for name in native["mesh"]},
                            _within_root(store.root, manifest["runtime_dir"]),
                            progress=stream_progress if progress is not None else None,
                        )
                    report("encoding")
                    recorder.prepare(index, progress=stream_progress if progress is not None else None)
                    for name in native["names"]:
                        destination = output / "tactile_streams" / name / recorder._base
                        shutil.rmtree(destination)
                        os.replace(stage / name, destination)
                    stage.rmdir()
                if manifest["offline_mesh_fields"]:
                    report("mesh")
                    available = {
                        f"observation.{name}.{suffix}": list(shape)
                        for name, camera in cameras.items()
                        for suffix, shape in camera.deferred_feature_shapes.items()
                    }
                    if any(
                        available.get(key) != shape
                        for key, shape in manifest["offline_mesh_fields"].items()
                    ):
                        raise ValueError(
                            "Missing or incompatible tactile cameras for saved offline features"
                        )
                    if manifest["runtime_dir"] is None:
                        raise ValueError("Raw offline episode has no saved runtime configuration")
                    runtime = _within_root(store.root, manifest["runtime_dir"])
                    if runtime != active_runtime:
                        sessions.close()
                        sessions.enter_context(deferred_sessions(cameras, runtime))
                        active_runtime = runtime
                    compute_episode_mesh(
                        dataset, cameras, runtime, index, episode_buffer=buffer,
                        progress=stream_progress if progress is not None else None,
                    )
                # LeRobot's encoder deletes its input directory. Give it private
                # hard links, never a directory symlink to the raw originals.
                for key in converted.meta.video_keys:
                    for frame_index, source in enumerate(buffer[key]):
                        destination = converted._get_image_file_path(index, key, frame_index)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(source, destination)
                        except OSError as exc:
                            if exc.errno != errno.EXDEV:
                                raise
                            shutil.copyfile(source, destination)
                report("saving")
                converted.save_episode(episode_data=buffer)
                if native:
                    converted.finish_native_episode(index)
                report(completed_episodes=ordinal)
        report("publishing")
        converted.finalize()
        plan_path = store.root / "meta/camera_streams.json"
        if plan_path.is_file():
            shutil.copyfile(plan_path, output / "meta/camera_streams.json")
        _publish_output(store.root, output, work)
        result = LeRobotDataset(
            dataset.repo_id, root=store.root, batch_encoding_size=1, vcodec=dataset.vcodec
        )
        _cleanup_processed_images(result)
        report("complete")
        return result
    except BaseException as exc:
        report("failed", error=str(exc))
        raise
    finally:
        if converted is not None:
            converted.finalize()
        # A failed rollback keeps its journal and backups for startup recovery.
        if not (store.root / ".raw_postprocessing_transaction.json").exists():
            shutil.rmtree(work)
