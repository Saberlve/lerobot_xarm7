from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.utils.episode_images import discard_episode_images, validate_episode_images


@pytest.fixture
def dataset(tmp_path):
    def directory(episode, key):
        return tmp_path / 'images' / key / f'episode_{episode:06d}'

    return SimpleNamespace(
        root=tmp_path,
        meta=SimpleNamespace(features={'rgb': {'dtype': 'video'}, 'still': {'dtype': 'image'}}),
        _get_image_file_dir=directory,
        _wait_image_writer=lambda: None,
    )


def populate(dataset, episode, count):
    for key in dataset.meta.features:
        directory = dataset._get_image_file_dir(episode, key)
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (directory / f'frame-{i:06d}.png').touch()


def test_rerecord_shorter_cleans_video_and_image_only_for_target(dataset):
    populate(dataset, 0, 261)
    populate(dataset, 1, 2)
    discard_episode_images(dataset, 0)
    populate(dataset, 0, 93)
    validate_episode_images(dataset, {'episode_index': 0, 'size': 93})
    validate_episode_images(dataset, {'episode_index': 1, 'size': 2})


@pytest.mark.parametrize('count', [92, 94, 261])
def test_reject_mismatched_count(dataset, count):
    populate(dataset, 0, count)
    with pytest.raises(RuntimeError, match='Refusing'):
        validate_episode_images(dataset, {'episode_index': 0, 'size': 93})


def test_waits_for_pending_writes_before_delete(dataset):
    dataset._wait_image_writer = lambda: populate(dataset, 0, 3)
    discard_episode_images(dataset, 0)
    assert not dataset._get_image_file_dir(0, 'rgb').exists()


def test_refuses_broad_directory(dataset):
    dataset._get_image_file_dir = lambda episode, key: dataset.root
    with pytest.raises(RuntimeError, match='Unsafe'):
        discard_episode_images(dataset, 0)
