"""Mistral provider implementation.

Example:
    >>> from nous.llm.providers import MistralProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = MistralProvider()
    >>> client = provider.model("mistral-large-latest")
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await client.complete(messages)
    >>> print(response.content[0].text)
"""

import json
import os
from typing import Any, AsyncIterator

from mistralai import Mistral

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


class MistralModelClient:
    """Model-specific client for Mistral.

    Created via MistralProvider.model() - do not instantiate directly.
    """

    def __init__(
        self,
        client: Mistral,
        model_id: str,
        max_tokens: int = 4096,
    ):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.MISTRAL

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
        """Generate a completion from Mistral.

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

    async def _complete(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> Message:
        """Non-streaming completion."""
        mistral_messages = self._convert_messages(messages, system_prompt)
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": mistral_messages,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await self._client.chat.complete_async(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), provider=self.provider.value) from exc
        return self._parse_response(response)

    async def _stream(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion."""
        mistral_messages = self._convert_messages(messages, system_prompt)
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": mistral_messages,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        accumulated_text = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in await self._client.chat.stream_async(**kwargs):
                if not chunk.data.choices:
                    continue

                delta = chunk.data.choices[0].delta

                # Handle text content
                if delta.content:
                    accumulated_text += delta.content
                    yield TextDeltaEvent(text=delta.content)

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, 'index') else 0
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
                if chunk.data.choices[0].finish_reason:
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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), provider=self.provider.value) from exc

    def _convert_messages(
        self,
        messages: list[Message],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        """Convert nous Messages to Mistral format."""
        result = []

        # Prepend system message if provided
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            converted = self._convert_message(msg)
            if converted:
                result.append(converted)
        return result

    def _convert_message(self, msg: Message) -> dict[str, Any] | None:
        """Convert a single message to Mistral format."""
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        for block in msg.content:
            if isinstance(block, TextContent):
                content_parts.append({"type": "text", "text": block.text})

            elif isinstance(block, ImageContent) and block.data:
                content_parts.append({
                    "type": "image_url",
                    "image_url": f"data:{block.mime_type};base64,{block.data}",
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
                # Tool results are separate messages in Mistral
                content_text = self._extract_tool_result_text(block)
                return {
                    "role": "tool",
                    "tool_call_id": block.tool_call_id,
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
        """Convert ToolDefinitions to Mistral format."""
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
        """Convert Mistral response to nous Message."""
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


class MistralProvider:
    """Mistral LLM provider.

    Handles connection configuration and produces model-specific clients.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        """Initialize the Mistral provider.

        Args:
            api_key: Mistral API key. If None, uses MISTRAL_API_KEY env var.
            max_tokens: Default max tokens for completions.
        """
        self._max_tokens = max_tokens
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY env var or pass api_key."
            )
        self._client = Mistral(api_key=key)

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.MISTRAL

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from Mistral API with capabilities."""
        response = await self._client.models.list_async()
        return [self._model_info(model) for model in response.data]

    def _model_info(self, model_data: Any) -> ModelInfo:
        """Create ModelInfo from Mistral model data."""
        model_id = model_data.id

        # Pixtral models support vision
        supports_vision = "pixtral" in model_id.lower()

        # Determine context window and tokens from model type
        if "large" in model_id:
            context_window = 128000
            max_tokens = 8192
        elif "medium" in model_id:
            context_window = 32000
            max_tokens = 4096
        else:
            context_window = 32000
            max_tokens = 4096

        return ModelInfo(
            id=model_id,
            name=getattr(model_data, "name", model_id),
            provider=self.provider.value,
            capabilities=ModelCapabilities(
                vision=supports_vision,
                tools=True,
                streaming=True,
                max_tokens=max_tokens,
                context_window=context_window,
            ),
        )

    def model(self, model_id: str) -> MistralModelClient:
        """Get a client configured for a specific model.

        Args:
            model_id: The Mistral model ID (e.g., "mistral-large-latest").

        Returns:
            A MistralModelClient for the specified model.
        """
        return MistralModelClient(
            client=self._client,
            model_id=model_id,
            max_tokens=self._max_tokens,
        )
