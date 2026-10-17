"""N0 raw-observation WebSocket client; no checkpoint or model imports."""
from __future__ import annotations

import time
import numpy as np
from .temporal_ensemble import TemporalEnsembleClientWrapper
from .websocket_policy_client import WebsocketClientPolicy


class N0PolicyClient:
    def __init__(self, *, host="127.0.0.1", port=10093, unnorm_key=None, transport=None):
        self.transport = transport if transport is not None else WebsocketClientPolicy(host=host, port=port)
        self._baselines = {}
        try:
            metadata = self.transport.get_server_metadata()
            if metadata.get("n0_observation_contract") != "raw_state_absolute_actions_v1":
                raise ValueError("Server must expose the N0 raw-state/absolute-action contract")
            self.unnorm_key = unnorm_key or metadata.get("default_unnorm_key")
            profiles = metadata["n0_profiles"]
            if self.unnorm_key is None and len(profiles) == 1:
                self.unnorm_key = next(iter(profiles))
            if self.unnorm_key not in profiles:
                raise ValueError("Select an available N0 unnorm_key")
            self.profile = profiles[self.unnorm_key]
            self.chunk_size = int(metadata["action_chunk_size"])
            self.state_dim = int(self.profile["state_dim"])
            self.action_dim = int(self.profile["action_dim"])
            self.model_dim = int(self.profile["model_dim"])
            if self.chunk_size < 1 or self.state_dim < 1 or self.action_dim < 1:
                raise ValueError("Invalid server dimensions")
        except Exception:
            self.transport.close()
            raise

    def predict_action(self, observation, **kwargs):
        example = {key: observation[key] for key in
                   ("image", "image_mask", "lang", "state", "tactile_mask") if key in observation}
        tactile = {}
        for key, value in observation.get("tactile", {}).items():
            if not observation.get("tactile_mask", {}).get(key, True):
                continue
            if isinstance(value, (list, tuple)) and len(value) == 2:
                baseline, current = value
                self._baselines[key] = np.asarray(baseline).copy()
            else:
                current = value
                self._baselines.setdefault(key, np.asarray(current).copy())
            tactile[key] = [self._baselines[key].copy(), current]
        example["tactile"] = tactile
        response = self.transport.predict_action({
            **kwargs, "examples": [example], "unnorm_key": self.unnorm_key,
            "observation_contract": "raw_state_absolute_actions_v1",
        })
        if not response.get("ok", False):
            raise RuntimeError(f"N0 inference failed: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"])
        if actions.shape != (1, self.chunk_size, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid server action shape/values: {actions.shape}")
        return {"actions": actions[0].copy()}

    def reset(self):
        self.transport.reset()
        self._baselines.clear()

    def close(self):
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class N0TemporalEnsembleClientWrapper(TemporalEnsembleClientWrapper):
    """Fuse absolute single-arm EEF coordinates; index 9 is the gripper.

    The server restores each chunk using its request observation before fusion.
    Missing-arm/padding coordinates never participate in the ensemble.
    """

    def __init__(self, client, query_period, action_horizon, coeff=0.01, ensemble_gripper=True):
        if not 1 <= query_period <= action_horizon:
            raise ValueError("TE query period must be between 1 and action horizon")
        if not np.isfinite(coeff):
            raise ValueError("TE coefficient must be finite")
        super().__init__(client, query_period, action_horizon, coeff, ensemble_gripper)

    def _discard_pending(self):
        # The existing wrapper does not drain requests before resetting its transport.
        if self._pending is not None:
            self._pending[0].join()
            self._pending = None

    def reset(self):
        self._discard_pending()
        super().reset()

    def close(self):
        self._discard_pending()
        self._client.close()

    def _infer(self, observation):
        start = time.perf_counter()
        actions = np.asarray(self._client.predict_action(observation)["actions"])
        if actions.shape != (self._horizon, 32) or not np.isfinite(actions).all():
            raise ValueError("N0 TE requires finite canonical 32D action chunks")
        return actions[:, :10], (time.perf_counter() - start) * 1000

    def get_action(self, observation):
        active = super().get_action(observation)
        return np.pad(active, (0, 22))
