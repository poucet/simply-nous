"""MCP client for tool server communication.

This module provides MCPClient, which manages connections to MCP servers
and provides a unified interface for tool discovery and execution.

Example:
    >>> from nous.mcp import MCPClient, MCPServerConfig
    >>> async with MCPClient() as client:
    ...     await client.connect(MCPServerConfig(
    ...         name="tools",
    ...         url="http://localhost:8080/mcp",
    ...     ))
    ...     tools = await client.list_tools()
    ...     result = await client.call_tool("search", {"query": "test"})
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anyio
from pydantic import BaseModel

from nous.types import ToolDefinition

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection via streaming HTTP."""

    name: str
    url: str
    headers: dict[str, str] = {}


class _ServerConnection:
    """Internal: holds connection state for a single MCP server.

    Manages its own transport and session lifecycle for clean
    per-server connect/disconnect.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Any = None
        self._transport_ctx: Any = None
        self._session_ctx: Any = None

    async def connect(self) -> None:
        """Establish transport and session."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._transport_ctx = streamablehttp_client(
            self.config.url, headers=self.config.headers or None
        )
        read, write, _ = await self._transport_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()

    async def disconnect(self) -> None:
        """Clean disconnect: session first, then transport."""
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
            self.session = None

        if self._transport_ctx:
            try:
                await self._transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_ctx = None


class MCPClient:
    """Client for MCP server communication.

    Manages connections to multiple MCP servers and provides unified
    access to tools across all connected servers.
    """

    def __init__(self) -> None:
        self._connections: dict[str, _ServerConnection] = {}
        self._tool_to_server: dict[str, str] = {}

    @property
    def connected_servers(self) -> list[str]:
        """Names of currently connected servers."""
        return list(self._connections.keys())

    def server_for_tool(self, tool_name: str) -> str | None:
        """Get the server name that provides a given tool."""
        return self._tool_to_server.get(tool_name)

    async def connect(
        self,
        config: MCPServerConfig,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        """Connect to an MCP server via streaming HTTP.

        Retries with exponential backoff on failure.

        Args:
            config: Server configuration with name, URL, and optional headers.
            max_retries: Maximum connection attempts.
            base_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.

        Raises:
            ValueError: If already connected to a server with this name.
            ImportError: If mcp package is not installed.
        """
        if config.name in self._connections:
            raise ValueError(f"Already connected to server: {config.name}")

        try:
            from mcp import ClientSession  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mcp package required for MCP support. "
                "Install with: pip install simply-nous[mcp]"
            ) from e

        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                conn = _ServerConnection(config)
                await conn.connect()
                self._connections[config.name] = conn
                await self._refresh_tools(config.name)
                logger.info(f"Connected to MCP server: {config.name}")
                return
            except Exception as e:
                last_error = e
                # Clean up failed attempt
                try:
                    await conn.disconnect()
                except Exception:
                    pass

                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Connection to {config.name} failed (attempt "
                        f"{attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Failed to connect to {config.name} after "
                        f"{max_retries} attempts: {e}"
                    )

        raise RuntimeError(
            f"Failed to connect to {config.name} after {max_retries} attempts"
        ) from last_error

    async def _refresh_tools(self, server_name: str) -> None:
        """Refresh tool cache for a specific server."""
        conn = self._connections[server_name]
        result = await conn.session.list_tools()

        for tool in result.tools:
            if tool.name in self._tool_to_server:
                existing = self._tool_to_server[tool.name]
                logger.warning(
                    f"Tool '{tool.name}' from '{server_name}' shadows "
                    f"existing tool from '{existing}'"
                )
            self._tool_to_server[tool.name] = server_name

    async def disconnect(self, name: str) -> None:
        """Disconnect from a specific MCP server.

        Args:
            name: Name of the server to disconnect from.
        """
        conn = self._connections.pop(name, None)
        if conn:
            await conn.disconnect()

        # Remove tool mappings for this server
        self._tool_to_server = {
            tool: server
            for tool, server in self._tool_to_server.items()
            if server != name
        }

        logger.info(f"Disconnected from MCP server: {name}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and clean up resources."""
        for conn in self._connections.values():
            await conn.disconnect()
        self._connections.clear()
        self._tool_to_server.clear()
        logger.info("Disconnected from all MCP servers")

    async def _reconnect(self, server_name: str) -> None:
        """Reconnect to a server that has lost connection.

        Args:
            server_name: Name of the server to reconnect.
        """
        conn = self._connections.get(server_name)
        if not conn:
            return

        config = conn.config
        logger.info(f"Reconnecting to MCP server: {server_name}")

        # Disconnect old connection
        await self.disconnect(server_name)

        # Reconnect (retry logic built into connect)
        await self.connect(config)

    async def _ensure_connected(self, server_name: str) -> None:
        """Ensure a server connection is alive, reconnecting if needed.

        Args:
            server_name: Name of the server to check.

        Raises:
            RuntimeError: If reconnection fails.
        """
        if server_name not in self._connections:
            raise RuntimeError(f"Not connected to server: {server_name}")

        conn = self._connections[server_name]
        if not conn.session:
            logger.warning(f"Connection to {server_name} has no session, reconnecting...")
            try:
                await self._reconnect(server_name)
            except Exception as e:
                raise RuntimeError(f"Failed to reconnect to {server_name}: {e}") from e

    def _build_tool_definition(self, tool: Any, server_name: str) -> ToolDefinition:
        """Convert MCP tool to ToolDefinition."""
        return ToolDefinition(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema,
            server_name=server_name,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        """Get all available tools from connected servers.

        Returns:
            List of tool definitions from all connected servers.
        """
        tools: list[ToolDefinition] = []

        for server_name, conn in list(self._connections.items()):
            try:
                result = await conn.session.list_tools()
                for tool in result.tools:
                    tools.append(self._build_tool_definition(tool, server_name))
            except anyio.ClosedResourceError:
                logger.warning(f"Connection to {server_name} closed, reconnecting...")
                await self._reconnect(server_name)
                conn = self._connections.get(server_name)
                if conn:
                    result = await conn.session.list_tools()
                    for tool in result.tools:
                        tools.append(self._build_tool_definition(tool, server_name))

        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool and return its result.

        Args:
            name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's result content.

        Raises:
            ValueError: If the tool is not found on any connected server.
        """
        server_name = self._tool_to_server.get(name)
        if not server_name:
            available = list(self._tool_to_server.keys())
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {available}"
            )

        conn = self._connections[server_name]

        try:
            result = await conn.session.call_tool(name, arguments)
            return result.content
        except anyio.ClosedResourceError:
            logger.warning(f"Connection to {server_name} closed during tool call, reconnecting...")
            await self._reconnect(server_name)

            conn = self._connections.get(server_name)
            if not conn:
                raise RuntimeError(f"Failed to reconnect to {server_name}")

            result = await conn.session.call_tool(name, arguments)
            return result.content

    def is_connected(self, name: str) -> bool:
        """Check if a server connection exists.

        Args:
            name: Name of the server.

        Returns:
            True if connected, False otherwise.
        """
        return name in self._connections

    async def tools_for_server(self, server_name: str) -> list[ToolDefinition]:
        """Get tools available on a specific server.

        Args:
            server_name: Name of the server to list tools for.

        Returns:
            List of tool definitions from the specified server.

        Raises:
            RuntimeError: If not connected to the specified server.
        """
        await self._ensure_connected(server_name)
        conn = self._connections[server_name]

        try:
            result = await conn.session.list_tools()
        except anyio.ClosedResourceError:
            logger.warning(f"Connection to {server_name} closed, reconnecting...")
            await self._reconnect(server_name)
            conn = self._connections.get(server_name)
            if not conn:
                raise RuntimeError(f"Failed to reconnect to {server_name}")
            result = await conn.session.list_tools()

        return [
            self._build_tool_definition(tool, server_name)
            for tool in result.tools
        ]

    async def call_tool_on_server(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Execute a tool on a specific server.

        Unlike call_tool() which routes by tool name, this method targets
        a specific server directly.

        Args:
            server_name: Name of the server to call the tool on.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's result content.

        Raises:
            RuntimeError: If not connected to the specified server.
        """
        await self._ensure_connected(server_name)
        conn = self._connections[server_name]

        try:
            result = await conn.session.call_tool(tool_name, arguments)
            return result.content
        except anyio.ClosedResourceError:
            logger.warning(
                f"Connection to {server_name} closed during tool call, reconnecting..."
            )
            await self._reconnect(server_name)
            conn = self._connections.get(server_name)
            if not conn:
                raise RuntimeError(f"Failed to reconnect to {server_name}")
            result = await conn.session.call_tool(tool_name, arguments)
            return result.content

    async def shutdown(self) -> None:
        """Disconnect all servers and clean up resources."""
        await self.disconnect_all()

    async def __aenter__(self) -> "MCPClient":
        """Support async context manager usage."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Disconnect all servers on context exit."""
        await self.disconnect_all()
