"""Ollama provider implementation.

Example:
    >>> from nous.llm.providers import OllamaProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = OllamaProvider()
    >>> client = provider.model("llama3.2")
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await client.complete(messages)
    >>> print(response.content[0].text)
"""

import json
from typing import Any, AsyncIterator

import httpx

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


class OllamaModelClient:
    """Model-specific client for Ollama.

    Created via OllamaProvider.model() - do not instantiate directly.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_id: str,
    ):
        self._client = client
        self._base_url = base_url
        self._model_id = model_id

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.OLLAMA

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
        """Generate a completion from Ollama.

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

    def _map_error(self, exc: httpx.HTTPError) -> ProviderError:
        """Map an httpx error to ProviderError.

        HTTPStatusError has .response with .status_code and .text.
        Other HTTPError subclasses (connection errors, timeouts) have no response.
        """
        status_code = None
        detail = None

        response = getattr(exc, "response", None)
        if response is not None:
            status_code = response.status_code
            if response.text:
                detail = response.text[:500]

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
        request = self._build_request(messages, system_prompt, tools, stream=False)
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json=request,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._map_error(exc) from exc
        return self._parse_response(response.json())

    async def _stream(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion."""
        request = self._build_request(messages, system_prompt, tools, stream=True)
        accumulated_text = ""
        accumulated_tool_calls: list[dict[str, Any]] = []

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=request,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    chunk = json.loads(line)
                    msg = chunk.get("message", {})

                    # Handle text content
                    if content := msg.get("content"):
                        accumulated_text += content
                        yield TextDeltaEvent(text=content)

                    # Handle tool calls
                    if tool_calls := msg.get("tool_calls"):
                        for tc in tool_calls:
                            tool_call = ToolCall(
                                id=tc.get("id", f"call_{len(accumulated_tool_calls)}"),
                                name=tc["function"]["name"],
                                input=tc["function"].get("arguments", {}),
                            )
                            accumulated_tool_calls.append(tc)
                            yield ToolCallEvent(tool_call=tool_call)

                    # Check if done
                    if chunk.get("done"):
                        final_message = self._build_final_message(
                            accumulated_text, accumulated_tool_calls
                        )
                        yield MessageCompleteEvent(message=final_message)
        except httpx.HTTPError as exc:
            raise self._map_error(exc) from exc

    def _build_request(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the Ollama API request."""
        ollama_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system_prompt:
            ollama_messages.insert(0, {"role": "system", "content": system_prompt})

        request: dict[str, Any] = {
            "model": self._model_id,
            "messages": ollama_messages,
            "stream": stream,
        }

        if tools:
            request["tools"] = self._convert_tools(tools)

        return request

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert nous Messages to Ollama format."""
        result = []
        for msg in messages:
            converted = self._convert_message(msg)
            if converted:
                result.append(converted)
        return result

    def _convert_message(self, msg: Message) -> dict[str, Any] | None:
        """Convert a single message to Ollama format."""
        text_parts: list[str] = []
        images: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in msg.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ImageContent) and block.data:
                images.append(block.data)
            elif isinstance(block, ToolUseContent):
                tool_calls.append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                })
            elif isinstance(block, ToolResultContent):
                # Tool results become separate messages in Ollama
                content_text = self._extract_tool_result_text(block)
                return {
                    "role": "tool",
                    "content": content_text,
                }

        message: dict[str, Any] = {
            "role": msg.role,
            "content": "\n".join(text_parts) if text_parts else "",
        }

        if images:
            message["images"] = images

        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    def _extract_tool_result_text(self, block: ToolResultContent) -> str:
        """Extract text content from a tool result."""
        for item in block.content:
            if isinstance(item, TextContent):
                return item.text
        return "Tool execution failed" if block.is_error else ""

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert ToolDefinitions to Ollama format."""
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

    def _parse_response(self, response: dict[str, Any]) -> Message:
        """Convert Ollama response to nous Message."""
        msg = response.get("message", {})
        return self._build_final_message(
            msg.get("content", ""),
            msg.get("tool_calls", []),
        )

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
            func = tc.get("function", tc)
            content_blocks.append(
                ToolUseContent(
                    id=tc.get("id", f"call_{len(content_blocks)}"),
                    name=func.get("name", ""),
                    input=func.get("arguments", {}),
                )
            )

        return Message(
            role="assistant",
            content=content_blocks,
            provider=self.provider.value,
            model=self._model_id,
        )


class OllamaProvider:
    """Ollama LLM provider.

    Handles connection configuration and produces model-specific clients.
    Uses Ollama's HTTP API directly without an SDK.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        """Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL. Defaults to localhost.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def provider(self) -> Provider:
        """The provider identifier."""
        return Provider.OLLAMA

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from Ollama with capabilities."""
        response = await self._client.get(f"{self._base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [self._model_info(model) for model in data.get("models", [])]

    def _model_info(self, model_data: dict[str, Any]) -> ModelInfo:
        """Create ModelInfo from Ollama model data."""
        model_name = model_data["name"]
        details = model_data.get("details", {})

        # Infer vision support from model family
        family = details.get("family", "").lower()
        supports_vision = any(v in model_name.lower() or v in family for v in ("llava", "vision", "bakllava"))

        # Get context length from details if available
        context_window = details.get("parameter_size")
        if isinstance(context_window, str):
            # Parse strings like "7B" - not actually context window
            context_window = None

        return ModelInfo(
            id=model_name,
            name=model_name,
            provider=self.provider.value,
            capabilities=ModelCapabilities(
                vision=supports_vision,
                tools=True,
                streaming=True,
                context_window=context_window,
            ),
        )

    def model(self, model_id: str) -> OllamaModelClient:
        """Get a client configured for a specific model.

        Args:
            model_id: The Ollama model name (e.g., "llama3.2").

        Returns:
            An OllamaModelClient for the specified model.
        """
        return OllamaModelClient(
            client=self._client,
            base_url=self._base_url,
            model_id=model_id,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
