"""Anthropic Claude provider implementation.

Example:
    >>> from nous.llm.providers import AnthropicProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = AnthropicProvider(model="claude-sonnet-4-20250514")
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await provider.complete(messages)
    >>> print(response.content[0].text)
"""

from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock

from nous.llm.events import (
    MessageCompleteEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolCallEvent,
)
from nous.types import (
    ContentBlock,
    ImageContent,
    Message,
    Provider,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)


class AnthropicProvider:
    """Anthropic Claude LLM provider.

    Implements the LLMProvider protocol for Claude models.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        """Initialize the Anthropic provider.

        Args:
            model: The Claude model ID to use.
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            base_url: Optional custom API base URL.
            max_tokens: Default max tokens for completions.
        """
        self._model = model
        self._max_tokens = max_tokens
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**client_kwargs)

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.ANTHROPIC

    async def list_models(self) -> list[str]:
        """Fetch available models from Anthropic API."""
        response = await self._client.models.list()
        return [model.id for model in response.data]

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message:
        """Generate a completion from Claude.

        Args:
            messages: Conversation history.
            system_prompt: Optional system prompt.
            tools: Optional tool definitions.
            stream: If True, return async iterator of events.

        Returns:
            Complete Message if stream=False, else AsyncIterator[StreamEvent].
        """
        if stream:
            return self._stream(messages, system_prompt, tools)
        return await self._complete(messages, system_prompt, tools)

    async def _complete(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Non-streaming completion."""
        request = self._build_request(messages, system_prompt, tools)
        response = await self._client.messages.create(**request)
        return self._parse_response(response)

    async def _stream(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion."""
        request = self._build_request(messages, system_prompt, tools)
        current_tool_call: dict[str, Any] | None = None

        async with self._client.messages.stream(**request) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if isinstance(event.content_block, ToolUseBlock):
                        current_tool_call = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                        }

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield TextDeltaEvent(text=delta.text)

                elif event.type == "content_block_stop":
                    if current_tool_call is not None:
                        snapshot = stream.current_message_snapshot
                        for block in snapshot.content:
                            if (
                                isinstance(block, ToolUseBlock)
                                and block.id == current_tool_call["id"]
                            ):
                                yield ToolCallEvent(
                                    tool_call=ToolCall(
                                        id=block.id,
                                        name=block.name,
                                        input=block.input,
                                    )
                                )
                                break
                        current_tool_call = None

                elif event.type == "message_stop":
                    final_message = self._parse_response(stream.current_message_snapshot)
                    yield MessageCompleteEvent(message=final_message)

    def _build_request(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> dict[str, Any]:
        """Build the Anthropic API request."""
        request: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(messages),
            "max_tokens": self._max_tokens,
        }
        if system_prompt:
            request["system"] = system_prompt
        if tools:
            request["tools"] = self._convert_tools(tools)
        return request

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert nous Messages to Anthropic format."""
        result = []
        for msg in messages:
            content_blocks = []
            for block in msg.content:
                converted = self._convert_content_block(block)
                if converted:
                    content_blocks.append(converted)
            result.append({"role": msg.role, "content": content_blocks})
        return result

    def _convert_content_block(self, block: ContentBlock) -> dict[str, Any] | None:
        """Convert a single content block to Anthropic format."""
        if isinstance(block, TextContent):
            return {"type": "text", "text": block.text}

        if isinstance(block, ImageContent):
            if block.data:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            return None

        if isinstance(block, ToolUseContent):
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }

        if isinstance(block, ToolResultContent):
            result_content = []
            for item in block.content:
                converted = self._convert_content_block(item)
                if converted:
                    result_content.append(converted)
            if not result_content:
                result_content.append({
                    "type": "text",
                    "text": "Tool execution failed" if block.is_error else "",
                })
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": result_content,
                "is_error": block.is_error,
            }

        return None

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert ToolDefinitions to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _parse_response(self, response: Any) -> Message:
        """Convert Anthropic response to nous Message."""
        content_blocks: list[ContentBlock] = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseContent(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return Message(
            role="assistant",
            content=content_blocks,
            provider=self.provider.value,
            model=self._model,
        )
