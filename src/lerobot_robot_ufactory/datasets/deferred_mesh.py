"""Compute deferred Parquet fields from lossless PNGs after recording."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from lerobot_robot_ufactory.datasets.tactile_indices import tactile_range_fields


@contextmanager
def deferred_sessions(cameras, runtime_dir):
    """Own offline solvers across all episodes; release on success or failure."""
    with ExitStack() as stack:
        for camera in cameras.values():
            session = getattr(camera, "deferred_session", None)
            if session is not None:
                stack.enter_context(session(Path(runtime_dir)))
        yield


def compute_episode_mesh(
    dataset, cameras, runtime_dir, episode_index, episode_buffer=None, synchronization=None,
    *, progress=None,
):
    buffer = dataset.episode_buffer if episode_buffer is None else episode_buffer
    if int(buffer["episode_index"]) != episode_index:
        raise RuntimeError("Offline mesh episode mismatch")
    dataset._wait_image_writer()
    native = {}
    if synchronization is not None and synchronization.tactile_recorder is not None:
        native = synchronization.tactile_recorder.compute_mesh(cameras, runtime_dir, progress=progress)
    results = {}
    range_keys = {name: key for key, name in tactile_range_fields(buffer).items()}
    for name, camera in cameras.items():
        range_key = range_keys.get(name)
        if range_key in buffer:
            # The row already contains the complete capture-window reference.
            # Never replace that reference with a representative numeric array.
            if synchronization is None or name not in native:
                raise RuntimeError(
                    "Native Mesh index processing requires the full capture transaction"
                )
            ranges = buffer[range_key]
            if len(ranges) != int(buffer["size"]):
                raise RuntimeError(f"Mesh index row count mismatch: {name}")
            for indices in ranges:
                indices = np.asarray(indices)
                if indices.dtype != np.int64 or indices.shape != (2,):
                    raise RuntimeError(f"Mesh ranges must be int64 pairs: {name}")
                start, end = indices
                if not 0 <= start <= end <= len(native[name]):
                    raise RuntimeError(f"Mesh range outside full native array: {name}")
            continue
        paths = buffer[f"observation.images.{name}"]
        if len(paths) != int(buffer["size"]):
            raise RuntimeError(f"Offline mesh frame count mismatch: {name}")
        values = {suffix: [] for suffix in camera.deferred_feature_shapes}
        reported_at = monotonic()
        if progress is not None:
            progress({"camera": name, "stage": "mesh", "completed_frames": 0,
                      "total_frames": len(paths)})
        with deferred_sessions({name: camera}, runtime_dir):
            for frame_index, path in enumerate(paths):
                if name in native:
                    import json

                    timing = json.loads(
                        synchronization.frames[frame_index]["camera_intervals_json"]
                    )[name]
                    index = timing.get("representative_tactile_index")
                    if index is not None:
                        values["mesh_motion_3d"].append(native[name][index].copy())
                        continue
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot read tactile image: {path}")
                computed = camera.compute_deferred_features(image, Path(runtime_dir))
                if set(computed) != set(values):
                    raise RuntimeError(f"Unexpected deferred tactile features: {name}")
                for suffix, value in computed.items():
                    values[suffix].append(np.asarray(value, dtype=np.float32).copy())
                now = monotonic()
                if progress is not None and now - reported_at >= 0.2:
                    progress({"camera": name, "stage": "mesh", "completed_frames": frame_index + 1,
                              "total_frames": len(paths)})
                    reported_at = now
        for suffix, feature_values in values.items():
            results[f"observation.{name}.{suffix}"] = feature_values
        if progress is not None:
            progress({"camera": name, "stage": "mesh", "completed_frames": len(paths),
                      "total_frames": len(paths)})
    buffer.update(results)
