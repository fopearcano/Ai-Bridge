"""Houdini integration layer.

Bridge / transports for executing generated Python against Houdini without
requiring the `hou` module to be importable in the host process.
"""

from aibridge_houdini.houdini.bridge import HoudiniBridge, TransportName
from aibridge_houdini.houdini.inspection import inspect_scene
from aibridge_houdini.houdini.scene_client import (
    SceneClientError,
    fetch_scene_context,
)
from aibridge_houdini.houdini.transport import (
    ExecutionResult,
    HythonTransport,
    SocketTransport,
    Transport,
    TransportError,
)

__all__ = [
    "ExecutionResult",
    "HoudiniBridge",
    "HythonTransport",
    "SceneClientError",
    "SocketTransport",
    "Transport",
    "TransportError",
    "TransportName",
    "fetch_scene_context",
    "inspect_scene",
]
