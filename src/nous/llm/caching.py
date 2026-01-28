"""Caching wrapper for LLM providers."""

from nous.llm.protocol import LLMProvider, ModelClient
from nous.types import Provider


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
        >>> client = provider.model("claude-sonnet-4-20250514")
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

    def model(self, model_id: str) -> ModelClient:
        """Get a client configured for a specific model."""
        return self._inner.model(model_id)

    def clear_cache(self) -> None:
        """Clear the models cache."""
        self._models_cache = None
