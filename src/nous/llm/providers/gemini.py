"""Google Gemini provider implementation.

Example:
    >>> from nous.llm.providers import GeminiProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = GeminiProvider()
    >>> client = provider.model("gemini-2.0-flash")
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await client.complete(messages)
    >>> print(response.content[0].text)
"""

import os
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from nous.llm.capabilities import ModelCapabilities, ModelInfo
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


class GeminiModelClient:
    """Model-specific client for Google Gemini.

    Created via GeminiProvider.model() - do not instantiate directly.
    """

    def __init__(
        self,
        client: genai.Client,
        model_id: str,
    ):
        self._client = client
        self._model_id = model_id

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.GOOGLE

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
        """Generate a completion from Gemini.

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
        config = self._build_config(system_prompt, tools)
        contents = self._convert_messages(messages)

        response = await self._client.aio.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=config,
        )
        return self._parse_response(response)

    async def _stream(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion."""
        config = self._build_config(system_prompt, tools)
        contents = self._convert_messages(messages)

        accumulated_text = ""
        accumulated_tool_calls: list[dict[str, Any]] = []

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model_id,
            contents=contents,
            config=config,
        ):
            # Handle text content
            if chunk.text:
                accumulated_text += chunk.text
                yield TextDeltaEvent(text=chunk.text)

            # Handle function calls
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.function_call:
                                fc = part.function_call
                                tool_call = ToolCall(
                                    id=f"call_{len(accumulated_tool_calls)}",
                                    name=fc.name,
                                    input=dict(fc.args) if fc.args else {},
                                )
                                accumulated_tool_calls.append({
                                    "name": fc.name,
                                    "args": dict(fc.args) if fc.args else {},
                                })
                                yield ToolCallEvent(tool_call=tool_call)

        # Yield final message
        final_message = self._build_final_message(accumulated_text, accumulated_tool_calls)
        yield MessageCompleteEvent(message=final_message)

    def _build_config(
        self,
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> types.GenerateContentConfig:
        """Build the Gemini generation config."""
        config_kwargs: dict[str, Any] = {}

        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        return types.GenerateContentConfig(**config_kwargs)

    def _convert_messages(self, messages: list[Message]) -> list[types.Content]:
        """Convert nous Messages to Gemini format."""
        contents: list[types.Content] = []

        for msg in messages:
            parts = self._convert_message_parts(msg)
            if parts:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(types.Content(role=role, parts=parts))

        return contents

    def _convert_message_parts(self, msg: Message) -> list[types.Part]:
        """Convert message content blocks to Gemini parts."""
        parts: list[types.Part] = []

        for block in msg.content:
            if isinstance(block, TextContent):
                parts.append(types.Part(text=block.text))

            elif isinstance(block, ImageContent) and block.data:
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=block.mime_type,
                            data=block.data,
                        )
                    )
                )

            elif isinstance(block, ToolUseContent):
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=block.name,
                            args=block.input,
                        )
                    )
                )

            elif isinstance(block, ToolResultContent):
                result_text = self._extract_tool_result_text(block)
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=block.tool_use_id,
                            response={"result": result_text},
                        )
                    )
                )

        return parts

    def _extract_tool_result_text(self, block: ToolResultContent) -> str:
        """Extract text content from a tool result."""
        for item in block.content:
            if isinstance(item, TextContent):
                return item.text
        return "Tool execution failed" if block.is_error else ""

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[types.Tool]:
        """Convert ToolDefinitions to Gemini format."""
        function_declarations = []
        for tool in tools:
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                )
            )
        return [types.Tool(function_declarations=function_declarations)]

    def _parse_response(self, response: Any) -> Message:
        """Convert Gemini response to nous Message."""
        text = ""
        tool_calls: list[dict[str, Any]] = []

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            text += part.text
                        if part.function_call:
                            fc = part.function_call
                            tool_calls.append({
                                "name": fc.name,
                                "args": dict(fc.args) if fc.args else {},
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

        for i, tc in enumerate(tool_calls):
            content_blocks.append(
                ToolUseContent(
                    id=f"call_{i}",
                    name=tc.get("name", ""),
                    input=tc.get("args", {}),
                )
            )

        return Message(
            role="assistant",
            content=content_blocks,
            provider=self.provider.value,
            model=self._model_id,
        )


class GeminiProvider:
    """Google Gemini LLM provider.

    Handles connection configuration and produces model-specific clients.
    """

    def __init__(
        self,
        api_key: str | None = None,
    ):
        """Initialize the Gemini provider.

        Args:
            api_key: Google API key. If None, uses GEMINI_API_KEY or GOOGLE_API_KEY env var.
        """
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY env var, "
                "or pass api_key to GeminiProvider()."
            )
        self._client = genai.Client(api_key=resolved_key)

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.GOOGLE

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from Gemini with capabilities."""
        models = []
        async for model in await self._client.aio.models.list():
            if model.name:
                # Strip "models/" prefix if present
                name = model.name
                if name.startswith("models/"):
                    name = name[7:]
                # Filter to generative models only
                if "gemini" in name.lower():
                    models.append(self._model_info(name, model))
        return models

    def _model_info(self, model_id: str, model_data: Any) -> ModelInfo:
        """Create ModelInfo from Gemini model data."""
        # All Gemini models support vision and tools
        # Gemini 2.0 supports audio
        supports_audio = "2.0" in model_id or "2.5" in model_id

        # Get context window from model data if available
        context_window = getattr(model_data, "input_token_limit", None)
        max_tokens = getattr(model_data, "output_token_limit", None)

        return ModelInfo(
            id=model_id,
            name=getattr(model_data, "display_name", model_id),
            provider=self.provider.value,
            capabilities=ModelCapabilities(
                vision=True,
                audio_input=supports_audio,
                audio_output=supports_audio,
                tools=True,
                streaming=True,
                max_tokens=max_tokens,
                context_window=context_window,
            ),
        )

    def model(self, model_id: str) -> GeminiModelClient:
        """Get a client configured for a specific model.

        Args:
            model_id: The Gemini model name (e.g., "gemini-2.0-flash").

        Returns:
            A GeminiModelClient for the specified model.
        """
        return GeminiModelClient(
            client=self._client,
            model_id=model_id,
        )
