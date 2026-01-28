"""Engine - storage-agnostic conversation engine.

The engine orchestrates conversation turns:
1. Reads messages from ConversationView
2. Calls LLM client
3. Streams events back through view callbacks
4. Handles tool calls by delegating to view

Callback sequence:
- on_text_delta: As streaming text arrives
- on_content_block: When tool call blocks are ready
- add_message: To persist each message (view handles UX internally)
- on_turn_complete: When entire turn is done

Example:
    >>> from nous.engine import Engine
    >>>
    >>> engine = Engine()
    >>> response = await engine.run_turn(client, view)
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from nous.llm.events import TextDeltaEvent, ToolCallEvent, MessageCompleteEvent, StreamEvent
from nous.types import Message, ToolCall, ToolDefinition, ToolResultContent
from nous.view.protocol import ConversationView


class ModelClient(Protocol):
    """Protocol for LLM clients that can complete conversations.

    Messages may include a "system" role message which the client
    should extract and handle appropriately for its provider.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion."""
        ...


class Engine:
    """Storage-agnostic conversation engine.

    Reads from ConversationView, writes events back through callbacks.
    All storage and UI concerns are delegated to the view.
    """

    async def run_turn(
        self,
        client: ModelClient,
        view: ConversationView,
        tools: list[ToolDefinition] | None = None,
    ) -> Message:
        """Run one conversation turn.

        Reads messages from view, calls client, streams events back
        through view callbacks. Handles tool calls by delegating to
        the view and continuing until no more tool calls.

        Args:
            client: The instantiated LLM client to use.
            view: The conversation view to read from and write to.
            tools: Optional tool definitions available to the model.

        Returns:
            The final assistant message after all tool calls complete.
        """
        return await self._completion_loop(
            view=view,
            client=client,
            tools=tools,
        )

    async def _completion_loop(
        self,
        view: ConversationView,
        client,
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Run completion loop, handling tool calls.

        The view is the source of truth for messages. Each iteration
        re-reads from view.get_messages() after persisting via add_message.

        Args:
            view: The conversation view.
            client: The model client.
            tools: Available tools.

        Returns:
            Final assistant message.
        """
        while True:
            response = await self._stream_completion(
                view=view,
                client=client,
                tools=tools,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                await view.add_message(response)
                await view.on_turn_complete()
                return response

            await view.add_message(response)

            # Execute all tool calls in parallel
            async def execute_tool(tool_use) -> ToolResultContent:
                tool_call = ToolCall(
                    id=tool_use.id,
                    name=tool_use.name,
                    input=tool_use.input,
                )
                result = await view.call_tool(tool_call)
                return ToolResultContent(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                )

            tool_results = await asyncio.gather(
                *[execute_tool(tu) for tu in tool_uses]
            )

            tool_result_message = Message(role="user", content=list(tool_results))
            await view.add_message(tool_result_message)

    async def _stream_completion(
        self,
        view: ConversationView,
        client,
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Stream a single completion, calling view callbacks.

        Args:
            view: The conversation view (source of messages).
            client: The model client.
            tools: Available tools.

        Returns:
            The complete assistant message.
        """
        messages = await view.get_messages()
        stream: AsyncIterator[StreamEvent] = await client.complete(
            messages=messages,
            tools=tools,
            stream=True,
        )

        async for event in stream:
            match event:
                case TextDeltaEvent(text=text):
                    await view.on_text_delta(text)
                case ToolCallEvent(tool_call=tc):
                    await view.on_content_block(tc)
                case MessageCompleteEvent(message=msg):
                    return msg

        raise RuntimeError("Stream ended without MessageCompleteEvent")
