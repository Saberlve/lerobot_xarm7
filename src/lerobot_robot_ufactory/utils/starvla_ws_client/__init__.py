"""Vendored starVLA WebSocket policy client.

Copied from the starVLA repository (deployment/model_server/tools/) so the
real-robot eval script can talk to a starVLA policy server without adding a
dependency on the starVLA package itself.
"""

from .websocket_policy_client import WebsocketClientPolicy

__all__ = ["WebsocketClientPolicy"]
