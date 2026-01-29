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

import logging
from contextlib import AsyncExitStack
from typing import Any

import anyio
from pydantic import BaseModel

from nous.types import ToolDefinition

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection via streaming HTTP."""

    name: str
    url: str


class _ServerConnection:
    """Internal: holds connection state for a single MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Any = None  # mcp.ClientSession
        self.read: Any = None
        self.write: Any = None


class MCPClient:
    """Client for MCP server communication.

    Manages connections to multiple MCP servers and provides unified
    access to tools across all connected servers.
    """

    def __init__(self) -> None:
        self._connections: dict[str, _ServerConnection] = {}
        self._exit_stack = AsyncExitStack()
        self._tool_to_server: dict[str, str] = {}

    @property
    def connected_servers(self) -> list[str]:
        """Names of currently connected servers."""
        return list(self._connections.keys())

    async def connect(self, config: MCPServerConfig) -> None:
        """Connect to an MCP server via streaming HTTP.

        Args:
            config: Server configuration with name and URL.

        Raises:
            ValueError: If already connected to a server with this name.
            ImportError: If mcp package is not installed.
        """
        if config.name in self._connections:
            raise ValueError(f"Already connected to server: {config.name}")

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:
            raise ImportError(
                "mcp package required for MCP support. "
                "Install with: pip install simply-nous[mcp]"
            ) from e

        conn = _ServerConnection(config)

        # Connect via streaming HTTP
        conn.read, conn.write, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(config.url)
        )

        conn.session = await self._exit_stack.enter_async_context(
            ClientSession(conn.read, conn.write)
        )
        await conn.session.initialize()

        self._connections[config.name] = conn

        # Discover and cache tool mappings
        await self._refresh_tools(config.name)

        logger.info(f"Connected to MCP server: {config.name}")

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

        Note:
            Due to AsyncExitStack limitations, individual disconnection
            is not fully supported. Use disconnect_all() for clean shutdown.
        """
        if name not in self._connections:
            return

        # Remove tool mappings for this server
        self._tool_to_server = {
            tool: server
            for tool, server in self._tool_to_server.items()
            if server != name
        }

        del self._connections[name]
        logger.info(f"Disconnected from MCP server: {name}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and clean up resources."""
        await self._exit_stack.aclose()
        self._connections.clear()
        self._tool_to_server.clear()
        self._exit_stack = AsyncExitStack()
        logger.info("Disconnected from all MCP servers")

    async def _is_connection_alive(self, server_name: str) -> bool:
        """Check if a server connection is still alive.

        Args:
            server_name: Name of the server to check.

        Returns:
            True if connection is alive, False otherwise.
        """
        conn = self._connections.get(server_name)
        if not conn or not conn.session:
            return False

        try:
            # Use list_tools as a lightweight ping
            await conn.session.list_tools()
            return True
        except Exception:
            return False

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

        # Remove old connection state
        del self._connections[server_name]

        # Remove tool mappings for this server
        self._tool_to_server = {
            tool: server
            for tool, server in self._tool_to_server.items()
            if server != server_name
        }

        # Reconnect
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

        if not await self._is_connection_alive(server_name):
            logger.warning(f"Connection to {server_name} appears dead, reconnecting...")
            try:
                await self._reconnect(server_name)
            except Exception as e:
                raise RuntimeError(f"Failed to reconnect to {server_name}: {e}") from e

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
                    tools.append(ToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema,
                    ))
            except anyio.ClosedResourceError:
                # Connection closed, try to reconnect
                logger.warning(f"Connection to {server_name} closed, reconnecting...")
                await self._reconnect(server_name)
                # Retry after reconnect
                conn = self._connections.get(server_name)
                if conn:
                    result = await conn.session.list_tools()
                    for tool in result.tools:
                        tools.append(ToolDefinition(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema,
                        ))

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
            # Connection closed, try to reconnect and retry
            logger.warning(f"Connection to {server_name} closed during tool call, reconnecting...")
            await self._reconnect(server_name)

            # Retry the call
            conn = self._connections.get(server_name)
            if not conn:
                raise RuntimeError(f"Failed to reconnect to {server_name}")

            result = await conn.session.call_tool(name, arguments)
            return result.content

    async def __aenter__(self) -> "MCPClient":
        """Support async context manager usage."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Disconnect all servers on context exit."""
        await self.disconnect_all()
