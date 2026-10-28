"""stdio transport for Amadeus, with one compatibility shim for clients that ping without an envelope.

On a 2026-07-28 ("modern") connection every request carries a per-request `_meta` envelope, and `ping` no longer
exists: an enveloped ping gets MethodNotFound. Some clients (Crush, through the Go MCP SDK) still send a bare `ping`
before every tool call. The python SDK rejects that with INVALID_PARAMS (no envelope), and Crush, which only excuses
MethodNotFound, then kills the server and starts a new one for every call. The shim gives such a ping the envelope of
the connection's last enveloped request, so it gets the same MethodNotFound an enveloped ping gets. Nothing else is
changed; legacy (initialize) connections, where `ping` is valid, never see an envelope and are left alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anyio
from mcp.server.stdio import stdio_server
from mcp.shared.inbound import CLIENT_CAPABILITIES_META_KEY, CLIENT_INFO_META_KEY, PROTOCOL_VERSION_META_KEY
from mcp.shared.message import SessionMessage
from mcp_types import JSONRPCRequest

ENVELOPE_KEYS = (PROTOCOL_VERSION_META_KEY, CLIENT_CAPABILITIES_META_KEY, CLIENT_INFO_META_KEY)


class BarePingShim:
    """Read stream wrapper: adds the connection's envelope to a bare `ping` once the client has sent one."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._envelope: dict[str, Any] | None = None

    @property
    def last_context(self) -> Any:
        return getattr(self._inner, 'last_context', None)

    def patch(self, item: Any) -> Any:
        if not (isinstance(item, SessionMessage) and isinstance(item.message, JSONRPCRequest)):
            return item
        req = item.message
        params = req.params if isinstance(req.params, Mapping) else {}
        meta = params.get('_meta')
        meta = meta if isinstance(meta, Mapping) else None
        if meta is not None and PROTOCOL_VERSION_META_KEY in meta and CLIENT_CAPABILITIES_META_KEY in meta:
            self._envelope = {k: meta[k] for k in ENVELOPE_KEYS if k in meta}
        elif req.method == 'ping' and self._envelope and (meta is None or PROTOCOL_VERSION_META_KEY not in meta):
            new = {**params, '_meta': {**(meta or {}), **self._envelope}}
            return SessionMessage(message=req.model_copy(update={'params': new}), metadata=item.metadata)
        return item

    async def receive(self) -> Any:
        return self.patch(await self._inner.receive())

    def __aiter__(self) -> BarePingShim:
        return self

    async def __anext__(self) -> Any:
        return self.patch(await self._inner.__anext__())

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> BarePingShim:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> bool | None:
        return await self._inner.__aexit__(*exc)


async def serve_stdio(server: Any) -> None:
    """Run an MCPServer over stdio, like `server.run('stdio')`, with the bare-ping shim on the read side."""
    low = server._lowlevel_server
    async with stdio_server() as (read_stream, write_stream):
        await low.run(BarePingShim(read_stream), write_stream, low.create_initialization_options())


def run_stdio(server: Any) -> None:
    anyio.run(serve_stdio, server)
