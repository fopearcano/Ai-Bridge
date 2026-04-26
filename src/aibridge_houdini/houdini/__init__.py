"""Houdini integration layer.

Bridge / transports for executing generated Python against Houdini without
requiring the `hou` module to be importable in the host process.
"""

from aibridge_houdini.houdini.bridge import HoudiniBridge, TransportName
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
    "SocketTransport",
    "Transport",
    "TransportError",
    "TransportName",
]
