"""OpenAI provider implementation.

Example:
    >>> from nous.llm.providers import OpenAIProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = OpenAIProvider()
    >>> client = provider.model("gpt-4o")
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await client.complete(messages)
    >>> print(response.content[0].text)
"""

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

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


class OpenAIModelClient:
    """Model-specific client for OpenAI.

    Created via OpenAIProvider.model() - do not instantiate directly.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_id: str,
        max_tokens: int = 4096,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.OPENAI

    @property
    def model_id(self) -> str:
        """The model identifier this client is configured for."""
        return self._model_id

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message:
        """Generate a completion from OpenAI.

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
        response = await self._client.chat.completions.create(**request)
        return self._parse_response(response)

    async def _stream(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion."""
        request = self._build_request(messages, system_prompt, tools)
        request["stream"] = True

        accumulated_text = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        async with await self._client.chat.completions.create(**request) as stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content
                if delta.content:
                    accumulated_text += delta.content
                    yield TextDeltaEvent(text=delta.content)

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                accumulated_tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

                # Check if done
                if chunk.choices[0].finish_reason:
                    # Yield tool call events for completed calls
                    for tc_data in accumulated_tool_calls.values():
                        try:
                            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield ToolCallEvent(
                            tool_call=ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                input=args,
                            )
                        )

                    # Yield final message
                    final_message = self._build_final_message(
                        accumulated_text,
                        list(accumulated_tool_calls.values()),
                    )
                    yield MessageCompleteEvent(message=final_message)

    def _build_request(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> dict[str, Any]:
        """Build the OpenAI API request."""
        openai_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system_prompt:
            openai_messages.insert(0, {"role": "system", "content": system_prompt})

        request: dict[str, Any] = {
            "model": self._model_id,
            "messages": openai_messages,
            "max_tokens": self._max_tokens,
        }

        if tools:
            request["tools"] = self._convert_tools(tools)

        return request

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert nous Messages to OpenAI format."""
        result = []
        for msg in messages:
            converted = self._convert_message(msg)
            if converted:
                result.append(converted)
        return result

    def _convert_message(self, msg: Message) -> dict[str, Any] | None:
        """Convert a single message to OpenAI format."""
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        for block in msg.content:
            if isinstance(block, TextContent):
                content_parts.append({"type": "text", "text": block.text})

            elif isinstance(block, ImageContent) and block.data:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.mime_type};base64,{block.data}",
                    },
                })

            elif isinstance(block, ToolUseContent):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

            elif isinstance(block, ToolResultContent):
                # Tool results are separate messages in OpenAI
                content_text = self._extract_tool_result_text(block)
                return {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": content_text,
                }

        # Build message
        message: dict[str, Any] = {"role": msg.role}

        # Handle content
        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
            message["content"] = content_parts[0]["text"]
        elif content_parts:
            message["content"] = content_parts
        else:
            message["content"] = ""

        # Handle tool calls (only for assistant messages)
        if tool_calls and msg.role == "assistant":
            message["tool_calls"] = tool_calls

        return message

    def _extract_tool_result_text(self, block: ToolResultContent) -> str:
        """Extract text content from a tool result."""
        for item in block.content:
            if isinstance(item, TextContent):
                return item.text
        return "Tool execution failed" if block.is_error else ""

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert ToolDefinitions to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _parse_response(self, response: Any) -> Message:
        """Convert OpenAI response to nous Message."""
        choice = response.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return self._build_final_message(text, tool_calls)

    def _build_final_message(
        self,
        text: str,
        tool_calls: list[dict[str, Any]],
    ) -> Message:
        """Build a nous Message from accumulated response data."""
        content_blocks: list[ContentBlock] = []

        if text:
            content_blocks.append(TextContent(text=text))

        for tc in tool_calls:
            content_blocks.append(
                ToolUseContent(
                    id=tc.get("id", f"call_{len(content_blocks)}"),
                    name=tc.get("name", ""),
                    input=tc.get("arguments", tc.get("input", {})),
                )
            )

        return Message(
            role="assistant",
            content=content_blocks,
            provider=self.provider.value,
            model=self._model_id,
        )


class OpenAIProvider:
    """OpenAI LLM provider.

    Handles connection configuration and produces model-specific clients.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            base_url: Optional custom API base URL (for Azure, etc.).
            max_tokens: Default max tokens for completions.
        """
        self._max_tokens = max_tokens
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.OPENAI

    async def list_models(self) -> list[str]:
        """Fetch available models from OpenAI API."""
        response = await self._client.models.list()
        return [model.id for model in response.data]

    def model(self, model_id: str) -> OpenAIModelClient:
        """Get a client configured for a specific model.

        Args:
            model_id: The OpenAI model ID (e.g., "gpt-4o", "gpt-4-turbo").

        Returns:
            An OpenAIModelClient for the specified model.
        """
        return OpenAIModelClient(
            client=self._client,
            model_id=model_id,
            max_tokens=self._max_tokens,
        )
