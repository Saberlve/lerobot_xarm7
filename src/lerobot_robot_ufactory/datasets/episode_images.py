"""Validate and discard only the active episode's temporary camera images."""

from pathlib import Path
import shutil


def episode_image_dirs(dataset, episode_index):
    root = Path(dataset.root).resolve()
    for key, feature in dataset.meta.features.items():
        if feature['dtype'] not in ('image', 'video'):
            continue
        directory = Path(dataset._get_image_file_dir(episode_index, key))
        resolved = directory.resolve()
        if resolved == root or root not in resolved.parents:
            raise RuntimeError(f'Unsafe episode image directory: {directory}')
        yield directory


def discard_episode_images(dataset, episode_index):
    # Complete queued writes before deleting files or reusing frame names.
    dataset._wait_image_writer()
    for directory in episode_image_dirs(dataset, episode_index):
        if directory.is_dir():
            shutil.rmtree(directory)


def validate_episode_images(dataset, episode_buffer):
    dataset._wait_image_writer()
    episode_index = int(episode_buffer['episode_index'])
    count = int(episode_buffer['size'])
    expected = {f'frame-{i:06d}.png' for i in range(count)}
    for directory in episode_image_dirs(dataset, episode_index):
        actual = {path.name for path in directory.glob('frame-*.png')}
        if actual != expected:
            raise RuntimeError(
                f'Episode {episode_index}: image frames do not match {count} records '
                f'in {directory} (missing={len(expected-actual)}, extra={len(actual-expected)}). '
                'Refusing to encode inconsistent video.'
            )
