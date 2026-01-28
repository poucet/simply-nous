"""Caching wrapper for LLM providers."""

from typing import AsyncIterator

from nous.llm.events import StreamEvent
from nous.llm.protocol import LLMProvider
from nous.types import Message, Provider, ToolDefinition


class CachingProvider:
    """Thin caching shim that wraps any LLMProvider.

    Caches list_models() results. Delegates all other methods.

    Example:
        >>> from nous.llm.caching import CachingProvider
        >>> from nous.llm.providers import AnthropicProvider
        >>>
        >>> provider = CachingProvider(AnthropicProvider())
        >>> models = await provider.list_models()  # fetches from API
        >>> models = await provider.list_models()  # returns cached
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self._models_cache: list[str] | None = None

    @property
    def provider(self) -> Provider:
        return self._inner.provider

    async def list_models(self) -> list[str]:
        """Fetch models, caching the result."""
        if self._models_cache is None:
            self._models_cache = await self._inner.list_models()
        return self._models_cache

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message:
        return await self._inner.complete(messages, system_prompt, tools, stream)

    def clear_cache(self) -> None:
        """Clear the models cache."""
        self._models_cache = None
