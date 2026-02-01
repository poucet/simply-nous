"""Anthropic Claude provider implementation.

Example:
    >>> from nous.llm.providers import AnthropicProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = AnthropicProvider()
    >>> client = provider.model("claude-sonnet-4-20250514")
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await client.complete(messages)
    >>> print(response.content[0].text)
"""

from typing import Any, AsyncIterator

from anthropic import APIError as AnthropicAPIError, AsyncAnthropic
from anthropic.types import ToolUseBlock

from nous.llm.capabilities import ModelCapabilities, ModelInfo
from nous.llm.events import (
    MessageCompleteEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolCallEvent,
)
from nous.llm.protocol import ProviderError
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


class AnthropicModelClient:
    """Model-specific client for Anthropic Claude.

    Created via AnthropicProvider.model() - do not instantiate directly.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model_id: str,
        max_tokens: int = 4096,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.ANTHROPIC

    @property
    def model_id(self) -> str:
        """The model identifier this client is configured for."""
        return self._model_id

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message:
        """Generate a completion from Claude.

        Args:
            messages: Conversation history (may include system role message).
            tools: Optional tool definitions.
            stream: If True, return async iterator of events.

        Returns:
            Complete Message if stream=False, else AsyncIterator[StreamEvent].
        """
        system_prompt, filtered_messages = self._extract_system(messages)
        if stream:
            return self._stream(filtered_messages, system_prompt, tools)
        return await self._complete(filtered_messages, system_prompt, tools)

    def _extract_system(
        self, messages: list[Message]
    ) -> tuple[str | None, list[Message]]:
        """Extract system message from messages list."""
        system_prompt = None
        filtered = []
        for msg in messages:
            if msg.role == "system":
                for block in msg.content:
                    if isinstance(block, TextContent):
                        system_prompt = block.text
                        break
            else:
                filtered.append(msg)
        return system_prompt, filtered

    def _map_error(self, exc: AnthropicAPIError) -> ProviderError:
        """Map an Anthropic SDK error to ProviderError.

        Anthropic's APIError has .status_code and .body with structure:
        {"type": "error", "error": {"type": "...", "message": "..."}}
        """
        status_code = getattr(exc, "status_code", None)
        detail = None

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", {})
            if isinstance(err, dict):
                detail = err.get("message")
                error_type = err.get("type")
                if error_type:
                    detail = f"{detail} ({error_type})" if detail else error_type

        return ProviderError(
            str(exc), status_code=status_code, detail=detail,
            provider=self.provider.value,
        )

    async def _complete(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Non-streaming completion."""
        request = self._build_request(messages, system_prompt, tools)
        try:
            response = await self._client.messages.create(**request)
        except AnthropicAPIError as exc:
            raise self._map_error(exc) from exc
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

        try:
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
        except AnthropicAPIError as exc:
            raise self._map_error(exc) from exc

    def _build_request(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> dict[str, Any]:
        """Build the Anthropic API request."""
        request: dict[str, Any] = {
            "model": self._model_id,
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
                "tool_use_id": block.tool_call_id,
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
            elif block.type == "image":
                content_blocks.append(ImageContent(
                    mime_type=block.source.media_type,
                    data=block.source.data,
                ))
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
            model=self._model_id,
        )


class AnthropicProvider:
    """Anthropic LLM provider.

    Handles connection configuration and produces model-specific clients.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            base_url: Optional custom API base URL.
            max_tokens: Default max tokens for completions.
        """
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

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from Anthropic API with capabilities."""
        response = await self._client.models.list()
        return [self._model_info(model.id, model.display_name) for model in response.data]

    def _model_info(self, model_id: str, display_name: str) -> ModelInfo:
        """Create ModelInfo with capabilities inferred from model ID."""
        # Claude 3+ models support vision, all Claude models support tools
        supports_vision = "claude-3" in model_id or "claude-4" in model_id
        # Determine context window from model family
        if "claude-3-5" in model_id or "claude-4" in model_id:
            context_window = 200000
            max_tokens = 8192
        elif "claude-3" in model_id:
            context_window = 200000
            max_tokens = 4096
        else:
            context_window = 100000
            max_tokens = 4096

        return ModelInfo(
            id=model_id,
            name=display_name,
            provider=self.provider.value,
            capabilities=ModelCapabilities(
                vision=supports_vision,
                tools=True,
                streaming=True,
                max_tokens=max_tokens,
                context_window=context_window,
            ),
        )

    def model(self, model_id: str) -> AnthropicModelClient:
        """Get a client configured for a specific model.

        Args:
            model_id: The Claude model ID (e.g., "claude-sonnet-4-20250514").

        Returns:
            An AnthropicModelClient for the specified model.
        """
        return AnthropicModelClient(
            client=self._client,
            model_id=model_id,
            max_tokens=self._max_tokens,
        )
