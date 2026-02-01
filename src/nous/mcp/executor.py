"""Tool executor for MCP tool calls.

This module provides ToolExecutor, which executes tool calls via MCPClient
and converts responses to ToolResult format.

Example:
    >>> from nous.mcp import MCPClient, ToolExecutor
    >>> from nous.types import ToolCall
    >>> client = MCPClient()
    >>> # ... connect to servers ...
    >>> executor = ToolExecutor(client)
    >>> call = ToolCall(name="read_file", input={"path": "/tmp/test.txt"})
    >>> result = await executor.execute(call)
    >>> print(result.content[0].text)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nous.types import TextContent, ImageContent, ToolCall, ToolResult

if TYPE_CHECKING:
    from nous.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tool calls via MCP and returns formatted results."""

    def __init__(self, client: "MCPClient", timeout: float | None = 30.0) -> None:
        """Initialize the executor.

        Args:
            client: MCPClient instance with connected servers.
            timeout: Default timeout in seconds for tool execution.
                     None means no timeout.
        """
        self.client = client
        self.timeout = timeout

    async def execute(
        self,
        tool_call: ToolCall,
        timeout: float | None = None,
    ) -> ToolResult:
        """Execute a tool call and return formatted result.

        Args:
            tool_call: The tool call to execute.
            timeout: Override the default timeout for this call.

        Returns:
            ToolResult with content from the tool execution.
        """
        effective_timeout = timeout if timeout is not None else self.timeout

        try:
            if effective_timeout is not None:
                content = await asyncio.wait_for(
                    self.client.call_tool(tool_call.name, tool_call.input),
                    timeout=effective_timeout,
                )
            else:
                content = await self.client.call_tool(tool_call.name, tool_call.input)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=self._convert_content(content),
                is_error=False,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Tool '{tool_call.name}' timed out after {effective_timeout}s")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=[TextContent(text=f"Tool execution timed out after {effective_timeout} seconds")],
                is_error=True,
            )

        except ValueError as e:
            # Tool not found
            logger.warning(f"Tool error: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=[TextContent(text=str(e))],
                is_error=True,
            )

        except Exception as e:
            logger.exception(f"Tool '{tool_call.name}' failed")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=[TextContent(text=f"Tool execution failed: {e}")],
                is_error=True,
            )

    def _convert_content(self, mcp_content: list) -> list[TextContent | ImageContent]:
        """Convert MCP content items to nous content types."""
        result: list[TextContent | ImageContent] = []

        for item in mcp_content:
            if hasattr(item, "type"):
                if item.type == "text":
                    result.append(TextContent(text=item.text))
                elif item.type == "image":
                    result.append(ImageContent(
                        mime_type=item.mimeType,
                        data=item.data,
                    ))
                else:
                    # Unknown type, convert to text representation
                    result.append(TextContent(text=str(item)))
            else:
                # Fallback for unexpected formats
                result.append(TextContent(text=str(item)))

        # Ensure we always have at least one content block
        if not result:
            result.append(TextContent(text="Tool executed successfully"))

        return result
