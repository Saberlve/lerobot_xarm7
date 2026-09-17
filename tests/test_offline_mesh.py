from types import SimpleNamespace as NS
import numpy as np
import pytest
from lerobot_robot_ufactory.tactile import deferred as offline_mesh


class FakeTactileCamera:
    deferred_feature_shapes = {'mesh_motion_3d': (35, 20, 3)}

    def __init__(self, released):
        self.released = released

    def compute_deferred_features(self, image, runtime_dir):
        assert image.shape == (4, 4, 3)
        self.released.append(True)
        return {'mesh_motion_3d': np.ones((35, 20, 3), dtype=np.float32)}


@pytest.mark.parametrize('length', [2, 3])
def test_post_save_mesh(tmp_path, monkeypatch, length):
    released = []
    monkeypatch.setattr(offline_mesh.cv2, 'imread', lambda *args: np.zeros((4,4,3),np.uint8))
    buffer = {'episode_index':0, 'size':length,
              'observation.images.photon':['first.png','second.png']}
    dataset = NS(episode_buffer=buffer, _wait_image_writer=lambda:None)
    cameras = {'photon': FakeTactileCamera(released)}
    if length == 3:
        with pytest.raises(RuntimeError, match='count mismatch'):
            offline_mesh.compute_episode_mesh(dataset,cameras,tmp_path,0)
        assert not (tmp_path/'offline_mesh3dflow/episode_000000/manifest.json').exists()
    else:
        offline_mesh.compute_episode_mesh(dataset,cameras,tmp_path,0)
        assert np.asarray(buffer['observation.photon.mesh_motion_3d']).shape == (2,35,20,3)
        assert not list(tmp_path.iterdir())
    assert released == ([] if length == 3 else [True, True])


def test_mesh_can_use_a_detached_episode_buffer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        offline_mesh.cv2, 'imread', lambda *args: np.zeros((4, 4, 3), np.uint8)
    )
    active = {'episode_index': 2, 'size': 0}
    detached = {
        'episode_index': 1,
        'size': 1,
        'observation.images.photon': ['frame.png'],
    }
    dataset = NS(episode_buffer=active, _wait_image_writer=lambda: None)
    cameras = {'photon': FakeTactileCamera([])}

    offline_mesh.compute_episode_mesh(
        dataset, cameras, tmp_path, 1, episode_buffer=detached
    )

    assert active == {'episode_index': 2, 'size': 0}
    assert np.asarray(detached['observation.photon.mesh_motion_3d']).shape == (1, 35, 20, 3)
