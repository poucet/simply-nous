"""Engine - storage-agnostic conversation engine.

The engine orchestrates conversation turns:
1. Reads state from ConversationView
2. Optionally retrieves RAG context via ContextBuilder
3. Calls LLM via ProviderHub
4. Streams events back through view callbacks

Example:
    >>> from nous.engine import Engine
    >>> from nous.llm import create_default_hub
    >>>
    >>> hub = create_default_hub()
    >>> engine = Engine(hub)
    >>> response = await engine.run_turn(view)
"""

from collections.abc import AsyncIterator

from nous.engine.context import ContextBuilder, DefaultContextBuilder
from nous.llm.hub import ProviderHub
from nous.llm.events import TextDeltaEvent, ToolCallEvent, MessageCompleteEvent, StreamEvent
from nous.types import Message, ToolCall, ToolDefinition, ToolResultContent
from nous.view.protocol import ConversationView


class Engine:
    """Storage-agnostic conversation engine.

    Reads from ConversationView, writes events back through callbacks.
    All storage and UI concerns are delegated to the view.
    """

    def __init__(
        self,
        hub: ProviderHub,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """Initialize engine with a provider hub.

        Args:
            hub: ProviderHub for accessing LLM providers.
            context_builder: Strategy for RAG context building.
                           Defaults to DefaultContextBuilder.
        """
        self.hub = hub
        self.context_builder = context_builder or DefaultContextBuilder()

    async def run_turn(
        self,
        view: ConversationView,
        tools: list[ToolDefinition] | None = None,
    ) -> Message:
        """Run one conversation turn.

        Builds context from the view, calls the LLM, and streams
        events back through view callbacks. Handles tool calls
        by delegating to the view and continuing.

        Args:
            view: The conversation view to read from and write to.
            tools: Optional tool definitions available to the model.

        Returns:
            The final assistant message after all tool calls complete.
        """
        client = await self.hub.client_for_model(view.model_id)

        system_prompt = view.get_system_prompt()
        messages = view.get_messages()

        # Apply RAG if context builder provides a query
        system_prompt, messages = await self._apply_knowledge(
            view, system_prompt, messages
        )

        return await self._completion_loop(
            view=view,
            client=client,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )

    async def _apply_knowledge(
        self,
        view: ConversationView,
        system_prompt: str | None,
        messages: list[Message],
    ) -> tuple[str | None, list[Message]]:
        """Apply RAG knowledge injection via context builder.

        Args:
            view: The conversation view.
            system_prompt: Current system prompt.
            messages: Conversation messages.

        Returns:
            Tuple of (system_prompt, messages) with knowledge injected.
        """
        query = self.context_builder.build_query(messages)
        if not query:
            return system_prompt, messages

        chunks = await view.on_knowledge_needed(query)
        if not chunks:
            return system_prompt, messages

        knowledge = self.context_builder.format_knowledge(chunks)
        if not knowledge:
            return system_prompt, messages

        return self.context_builder.inject_knowledge(
            system_prompt, messages, knowledge
        )

    async def _completion_loop(
        self,
        view: ConversationView,
        client,
        system_prompt: str | None,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Run completion loop, handling tool calls.

        Args:
            view: The conversation view.
            client: The model client.
            system_prompt: System prompt for the conversation.
            messages: Conversation messages.
            tools: Available tools.

        Returns:
            Final assistant message.
        """
        while True:
            response = await self._stream_completion(
                view=view,
                client=client,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                await view.on_message_complete(response)
                return response

            await view.on_message_complete(response)
            messages = [*messages, response]

            tool_results = []
            for tool_use in tool_uses:
                tool_call = ToolCall(
                    id=tool_use.id,
                    name=tool_use.name,
                    input=tool_use.input,
                )
                result = await view.on_tool_call(tool_call)
                tool_results.append(
                    ToolResultContent(
                        tool_use_id=tool_use.id,
                        content=result.content,
                        is_error=result.is_error,
                    )
                )

            messages = [
                *messages,
                Message(role="user", content=tool_results),
            ]

    async def _stream_completion(
        self,
        view: ConversationView,
        client,
        system_prompt: str | None,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Stream a single completion, calling view callbacks.

        Args:
            view: The conversation view.
            client: The model client.
            system_prompt: System prompt.
            messages: Conversation messages.
            tools: Available tools.

        Returns:
            The complete assistant message.
        """
        stream: AsyncIterator[StreamEvent] = await client.complete(
            messages=messages,
            system_prompt=system_prompt,
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
