"""Vendored starVLA WebSocket policy client.

Copied from the starVLA repository (deployment/model_server/tools/) so the
real-robot eval script can talk to a starVLA policy server without adding a
dependency on the starVLA package itself. TE uses the single implementation
in VLArmory deployment.model_server.tools and requires VLArmory on the Python path.
"""

from .rtc_policy_client_wrapper import RTCPolicyClientWrapper
from deployment.model_server.tools.temporal_ensemble import TemporalEnsembleClientWrapper, TemporalEnsembler
from .websocket_policy_client import WebsocketClientPolicy

__all__ = [
    "RTCPolicyClientWrapper",
    "TemporalEnsembleClientWrapper",
    "TemporalEnsembler",
    "WebsocketClientPolicy",
]
