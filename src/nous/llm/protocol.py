"""LLM provider protocol - abstract interface for LLM providers.

The architecture separates connection handling from model-specific operations:
- LLMProvider: Handles connection/auth, provides .model() to get a client
- ModelClient: Model-specific client with complete() method

Example:
    >>> from nous.llm.providers import AnthropicProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> provider = AnthropicProvider()  # Connection only
    >>> client = provider.model("claude-sonnet-4-20250514")  # Model-specific client
    >>> messages = [Message(role="user", content=[TextContent(text="Hello!")])]
    >>> response = await client.complete(messages)
    >>> print(response.content[0].text)
"""

from __future__ import annotations

from typing import Protocol, AsyncIterator, runtime_checkable

from nous.types import Message, Provider, ToolDefinition
from nous.llm.capabilities import ModelInfo
from nous.llm.events import StreamEvent


class ProviderError(Exception):
    """Generic error from an LLM provider API call.

    Providers catch SDK-specific exceptions and raise this instead,
    giving the engine a uniform error type with structured HTTP details.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
        provider: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.provider = provider

    def format_detail(self) -> str:
        """Format status_code and detail into a bracketed suffix for log messages."""
        parts = []
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        # Skip detail if it just repeats the main message
        if self.detail and self.detail != str(self):
            parts.append(self.detail)
        return f" [{', '.join(parts)}]" if parts else ""


@runtime_checkable
class ModelClient(Protocol):
    """Model-specific client for LLM completions.

    Returned by LLMProvider.model() and tied to a specific model.
    """

    @property
    def provider(self) -> Provider:
        """The provider identifier for this client."""
        ...

    @property
    def model_id(self) -> str:
        """The model identifier this client is configured for."""
        ...

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message:
        """Generate a completion from the LLM.

        Args:
            messages: Conversation history to send to the model.
            system_prompt: Optional system prompt for the conversation.
            tools: Optional list of tool definitions available to the model.
            stream: If True, return an async iterator of streaming events.
                   If False, return the complete message.

        Returns:
            If stream=False: A complete Message with the model's response.
            If stream=True: An AsyncIterator yielding StreamEvent objects
                           (TextDeltaEvent, ToolCallEvent, MessageCompleteEvent).
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract LLM provider interface.

    Providers handle connection configuration (API keys, base URLs) and
    produce ModelClient instances for specific models.
    """

    @property
    def provider(self) -> Provider:
        """The provider identifier for this instance."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider.

        Returns:
            List of ModelInfo objects with capabilities.
            Results may be cached by the implementation.
        """
        ...

    def model(self, model_id: str) -> ModelClient:
        """Get a client configured for a specific model.

        Args:
            model_id: The model identifier (e.g., "claude-sonnet-4-20250514").

        Returns:
            A ModelClient instance for the specified model.
        """
        ...
