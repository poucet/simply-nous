"""LLM provider protocol - abstract interface for LLM providers.

The LLMProvider protocol defines how the engine interacts with LLM backends.
Each provider (Anthropic, OpenAI, etc.) implements this protocol, handling
format conversion and streaming internally.

Example:
    >>> from nous.llm import LLMProvider
    >>> from nous.types import Message, TextContent
    >>>
    >>> async def generate(provider: LLMProvider, prompt: str) -> str:
    ...     messages = [Message(role="user", content=[TextContent(text=prompt)])]
    ...     response = await provider.complete(messages)
    ...     return response.content[0].text
"""

from typing import Protocol, AsyncIterator, runtime_checkable

from nous.types import Message, Provider, ToolDefinition
from nous.llm.events import StreamEvent


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract LLM provider interface.

    Providers implement this protocol to integrate with the nous engine.
    Each provider handles its own message format conversion and API calls.
    """

    @property
    def provider(self) -> Provider:
        """The provider identifier for this instance."""
        ...

    async def list_models(self) -> list[str]:
        """Fetch available models from the provider.

        Returns:
            List of model IDs available from this provider.
            Results may be cached by the implementation.
        """
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
