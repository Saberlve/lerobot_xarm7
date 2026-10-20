"""Model-independent policy transport and action scheduling."""
from __future__ import annotations
import time
import numpy as np
from .temporal_ensemble import TemporalEnsembleClientWrapper

class PolicyClient:
    def __init__(self, *, host="127.0.0.1", port=10093, unnorm_key=None, transport=None):
        if transport is None:
            from .websocket_policy_client import WebsocketClientPolicy
            transport = WebsocketClientPolicy(host=host, port=port)
        self.transport = transport
        try:
            metadata = transport.get_server_metadata()
            profiles = metadata.get("policy_profiles", {})
            self.unnorm_key = unnorm_key or metadata.get("default_unnorm_key")
            if self.unnorm_key is None and len(profiles) == 1:
                self.unnorm_key = next(iter(profiles))
            if self.unnorm_key not in profiles:
                raise ValueError("Select an available policy normalization domain")
            self.profile = profiles[self.unnorm_key]
            profile = self.profile
            self.action_space = profile.get("action_space")
            self.state_dim, self.action_dim = profile.get("state_dim"), profile.get("action_dim")
            if (self.action_space not in ("eef10", "joint")
                    or type(self.state_dim) is not int or self.state_dim < 1
                    or type(self.action_dim) is not int or self.action_dim < 1):
                raise ValueError("Invalid policy dimensions or action_space")
            self.chunk_size = int(metadata["action_chunk_size"])
            if self.chunk_size < 1 or self.chunk_size != profile["action_horizon"]:
                raise ValueError("Invalid policy action horizon")
        except Exception:
            transport.close()
            raise

    def predict_action(self, observation):
        reply = self.transport.predict_action(dict(
            examples=[observation], unnorm_key=self.unnorm_key))
        if not reply.get("ok"):
            raise RuntimeError(f"Policy inference failed: {reply.get('error')}")
        actions = np.asarray(reply["data"]["actions"])
        if actions.shape != (1, self.chunk_size, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError("Invalid policy action chunk")
        return {"actions": actions[0].copy()}

    def reset(self):
        self.transport.reset()

    def close(self):
        self.transport.close()


class PolicyTemporalEnsembleClientWrapper(TemporalEnsembleClientWrapper):
    """Fuse the action coordinates returned by the server."""

    def __init__(self, client, query_period, action_horizon, coeff=0.01, ensemble_gripper=True):
        if not 1 <= query_period <= action_horizon:
            raise ValueError("TE query period must be between 1 and action horizon")
        if not np.isfinite(coeff):
            raise ValueError("TE coefficient must be finite")
        self._action_dim = client.action_dim
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
        if actions.shape != (self._horizon, self._action_dim) or not np.isfinite(actions).all():
            raise ValueError("Policy TE requires a finite action chunk")
        return actions, (time.perf_counter() - start) * 1000

