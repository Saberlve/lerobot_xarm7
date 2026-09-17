"""Compute deferred Parquet fields from lossless PNGs after recording."""
from pathlib import Path
import cv2
import numpy as np


def compute_episode_mesh(dataset, cameras, runtime_dir, episode_index, episode_buffer=None):
    buffer = dataset.episode_buffer if episode_buffer is None else episode_buffer
    if int(buffer['episode_index']) != episode_index:
        raise RuntimeError('Offline mesh episode mismatch')
    dataset._wait_image_writer()
    results = {}
    for name, camera in cameras.items():
        paths = buffer[f'observation.images.{name}']
        if len(paths) != int(buffer['size']):
            raise RuntimeError(f'Offline mesh frame count mismatch: {name}')
        values = {suffix: [] for suffix in camera.deferred_feature_shapes}
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Cannot read tactile image: {path}')
            computed = camera.compute_deferred_features(image, Path(runtime_dir))
            if set(computed) != set(values):
                raise RuntimeError(f'Unexpected deferred tactile features: {name}')
            for suffix, value in computed.items():
                values[suffix].append(np.asarray(value, dtype=np.float32).copy())
        for suffix, feature_values in values.items():
            results[f'observation.{name}.{suffix}'] = feature_values
    buffer.update(results)
