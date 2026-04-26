from __future__ import annotations

import logging
from typing import Literal

from aibridge_houdini.config import Settings
from aibridge_houdini.houdini.transport import (
    ExecutionResult,
    HythonTransport,
    SocketTransport,
    Transport,
    TransportError,
)


log = logging.getLogger("aibridge.houdini.bridge")


TransportName = Literal["hython", "socket"]


class HoudiniBridge:
    """Send Houdini Python to a configured transport and return its result.

    The bridge does not decide *whether* to run code — that's the execution
    layer's job (e.g. checking MODE=safe vs direct, asking the user). Once
    called, it forwards the code to the chosen transport.
    """

    def __init__(
        self,
        settings: Settings,
        transport: TransportName | Transport = "hython",
    ) -> None:
        self._settings = settings
        if isinstance(transport, str):
            self._transport: Transport = self._build_transport(transport)
        else:
            self._transport = transport
        log.info("HoudiniBridge ready (transport=%s)", self._transport.name)

    @property
    def transport(self) -> Transport:
        return self._transport

    def execute(self, code: str) -> ExecutionResult:
        if not code.strip():
            return ExecutionResult(
                transport=self._transport.name,
                success=False,
                error="empty code — nothing to execute",
            )
        log.info(
            "executing %d bytes via %s", len(code), self._transport.name
        )
        return self._transport.execute(code)

    def _build_transport(self, name: TransportName) -> Transport:
        if name == "hython":
            return HythonTransport(self._settings)
        if name == "socket":
            return SocketTransport(self._settings)
        raise TransportError(f"unknown transport: {name}")
