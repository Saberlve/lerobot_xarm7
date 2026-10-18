"""xArm7 FTP1 EEF client. Geometry is independent of model/robot packages.

Wire actions are already unnormalized by the server. FTP1 pose9 uses ROWS;
the common xArm EEF executor consumes COLUMNS and metres in a 32D envelope.
"""
from __future__ import annotations
import numpy as np

CONTRACT = 'raw_ftp1_state_native_actions_v1'
MAPPING = 'xarm7_ftp1_eef_command_actions_v2'
TACTILE = 'right_tactile_gripper'
SCHEMA = {TACTILE: dict(type='image', sensor='photon', shape=[1, 2, 224, 224, 3], function_areas=[0, 1])}


def rotation6d(values, *, rows):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError('Invalid rotation6D')
    a, b = values[..., :3], values[..., 3:]
    size = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(size < 1e-8): raise ValueError('Degenerate first rotation axis')
    a = a / size
    b = b - (a * b).sum(axis=-1, keepdims=True) * a
    size = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(size < 1e-8): raise ValueError('Degenerate second rotation axis')
    b = b / size
    return np.stack([a, b, np.cross(a, b)], axis=-2 if rows else -1)


def recorded_matrix(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (10,) or not np.isfinite(raw).all():
        raise ValueError('Expected recording EEF/gripper 10D state')
    mat = np.eye(4)
    mat[:3, :3] = rotation6d(raw[3:9], rows=False)
    mat[:3, 3] = raw[:3] * .001
    return mat


def pose9(mat):
    return np.concatenate([mat[..., :3, 3], mat[..., :2, :3].reshape(mat.shape[:-2] + (6,))], axis=-1)


def encode_state(raw, representation):
    base = recorded_matrix(raw)
    if representation == 'abs':
        state_mat = base
    elif representation in ('relative', 'rel', 'delta'):
        state_mat = np.eye(4)  # One current observation, relative to itself.
    else:
        raise ValueError('Unsupported proprioception pose representation')
    state = np.zeros((1, 120), dtype=np.float64)
    state[0, :9] = pose9(state_mat)
    state[0, 44] = raw[9]
    return state, base


def restore_actions(native, base, representation):
    native = np.asarray(native, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 120 or not np.isfinite(native).all():
        raise ValueError('Expected finite (horizon, 120) native actions')
    rel = np.tile(np.eye(4), (len(native), 1, 1))
    rel[:, :3, 3] = native[:, :3]
    rel[:, :3, :3] = rotation6d(native[:, 3:9], rows=True)
    if representation == 'relative':
        absolute = base @ rel
    elif representation == 'abs':
        absolute = rel
    elif representation == 'rel':
        absolute = rel.copy()
        absolute[:, :3, 3] += base[:3, 3]
        absolute[:, :3, :3] = rel[:, :3, :3] @ base[:3, :3]
    elif representation == 'delta':
        absolute = rel.copy()
        absolute[:, :3, 3] = np.cumsum(rel[:, :3, 3], axis=0) + base[:3, 3]
        rotation = base[:3, :3]
        for i in range(len(rel)):
            rotation = rel[i, :3, :3] @ rotation
            absolute[i, :3, :3] = rotation
    else:
        raise ValueError('Unsupported action pose representation')
    result = np.zeros((len(native), 32), dtype=np.float64)
    result[:, :3] = absolute[:, :3, 3]
    result[:, 3:6] = absolute[:, :3, 0]
    result[:, 6:9] = absolute[:, :3, 1]
    result[:, 9] = native[:, 44]
    return result


class FTP1PolicyClient:
    def __init__(self, *, host='127.0.0.1', port=10093, unnorm_key=None, transport=None):
        if transport is None:
            from .websocket_policy_client import WebsocketClientPolicy
            transport = WebsocketClientPolicy(host=host, port=port)
        self.transport = transport
        try:
            metadata = transport.get_server_metadata()
            if metadata.get('ftp1_observation_contract') != CONTRACT:
                raise ValueError('FTP1 raw-state/native-action server required')
            profiles = metadata.get('ftp1_profiles', {})
            self.unnorm_key = unnorm_key or metadata.get('default_unnorm_key')
            if self.unnorm_key is None and len(profiles) == 1:
                self.unnorm_key = next(iter(profiles))
            profile = profiles[self.unnorm_key]
            mapping = profile.get('robot_profile', {})
            if mapping.get('mapping_version') != MAPPING:
                raise ValueError('Checkpoint must advertise the xArm7 EEF v2 training mapping')
            self.options = mapping['options']
            if set(self.options) != {'proprioception_pose_rep', 'action_pose_rep'} or any(v not in ('abs','relative','rel','delta') for v in self.options.values()):
                raise ValueError('Invalid FTP1 pose representation metadata')
            if profile['state_dim'] != 120 or profile['action_dim'] != 120 or profile['action_representation'] != 'native_training_representation':
                raise ValueError('Expected native 120D FTP1 checkpoint')
            if profile['tactile_input_config'].get(self.unnorm_key) != SCHEMA:
                raise ValueError('Checkpoint tactile schema differs from current left/right Photon frames')
            self.chunk_size = int(metadata['action_chunk_size'])
            if self.chunk_size < 1 or self.chunk_size != profile['action_horizon']:
                raise ValueError('Invalid FTP1 chunk size')
            self.action_dim = 32  # Output of this client, after native action recovery.
            self.native_dim = 120
        except Exception:
            transport.close()
            raise

    def predict_action(self, observation):
        from PIL import Image
        import cv2
        raw = observation['eef_record']
        state, base = encode_state(raw, self.options['proprioception_pose_rep'])
        def frame(value):
            arr = np.asarray(value)
            if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError('Expected RGB uint8 HWC camera frame')
            return arr
        # Match Xarm7FTP1Dataset.canonical_sample: PIL for RGB, cv2 for tactile.
        images = {name: np.asarray(Image.fromarray(frame(value)).resize((224,224)))
                  for name,value in observation['image'].items()}
        expected = {'camera_main_rgb_0', 'right_wrist_camera_rgb_0'}
        if set(images) != expected: raise ValueError('FTP1 needs external and right-wrist RGB cameras')
        tactile = observation['tactile']
        if len(tactile) != 2: raise ValueError('Expected left/right current tactile frames')
        tactile = np.stack([cv2.resize(frame(v), (224,224)) for v in tactile])[None]
        prompt = observation['lang'].strip()
        if not prompt: raise ValueError('Empty task instruction')
        if not prompt.endswith('.'): prompt += '.'
        example = dict(state=state, state_mask=np.ones(1, dtype=bool), image=images,
                       image_mask={name:True for name in images}, lang=prompt,
                       tactile={TACTILE:tactile}, tactile_sensor={TACTILE:'photon'},
                       tactile_function_area={TACTILE:np.array([0,1],np.int32)})
        reply = self.transport.predict_action(dict(examples=[example], unnorm_key=self.unnorm_key,
                                                  observation_contract=CONTRACT))
        if not reply.get('ok'): raise RuntimeError(f'FTP1 inference failed: {reply.get("error")}')
        native = np.asarray(reply['data']['actions'])
        if native.shape != (1,self.chunk_size,120): raise ValueError('Invalid FTP1 action chunk shape')
        return {'actions':restore_actions(native[0],base,self.options['action_pose_rep'])}

    def reset(self):
        self.transport.reset()

    def close(self):
        self.transport.close()
