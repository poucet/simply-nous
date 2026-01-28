"""ProviderHub - Registry and factory for LLM providers.

The ProviderHub manages provider instances, allowing lazy instantiation
via factory functions and model-to-provider mapping.

Example:
    >>> from nous.llm import ProviderHub
    >>> from nous.llm.providers import AnthropicProvider
    >>> from nous.llm.caching import CachingProvider
    >>> from nous.types import Provider
    >>>
    >>> hub = ProviderHub()
    >>> hub.register(Provider.ANTHROPIC, lambda: CachingProvider(AnthropicProvider()))
    >>> provider = hub.get(Provider.ANTHROPIC)
    >>> client = provider.model("claude-sonnet-4-20250514")
    >>> # Or get client directly by model ID
    >>> client = await hub.client_for_model("claude-sonnet-4-20250514")
"""

import asyncio
from typing import Callable

from nous.llm.protocol import LLMProvider, ModelClient
from nous.types import Provider


class ProviderHub:
    """Registry of LLM providers.

    Manages provider registration and instantiation with lazy loading.
    Model-to-provider mapping is built by querying each provider's models.
    """

    def __init__(self) -> None:
        """Initialize an empty provider hub."""
        self._factories: dict[Provider, Callable[[], LLMProvider]] = {}
        self._instances: dict[Provider, LLMProvider] = {}
        self._model_cache: dict[str, Provider] = {}

    def register(
        self,
        provider: Provider,
        factory: Callable[[], LLMProvider],
    ) -> None:
        """Register a provider factory.

        Args:
            provider: The provider identifier.
            factory: A callable that returns an LLMProvider instance.
        """
        self._factories[provider] = factory
        self._instances.pop(provider, None)

    def get(self, provider: Provider) -> LLMProvider:
        """Get a provider instance.

        Args:
            provider: The provider identifier.

        Returns:
            The LLMProvider instance (lazily instantiated).

        Raises:
            KeyError: If the provider is not registered.
        """
        if provider not in self._instances:
            if provider not in self._factories:
                raise KeyError(f"Provider {provider.value} not registered")
            self._instances[provider] = self._factories[provider]()
        return self._instances[provider]

    async def get_for_model(self, model_id: str) -> LLMProvider:
        """Get a provider for a given model ID.

        Queries registered providers for their supported models in parallel.
        Results are cached for subsequent lookups.

        Args:
            model_id: The model identifier (e.g., "claude-sonnet-4-20250514").

        Returns:
            The LLMProvider instance for this model.

        Raises:
            KeyError: If no provider supports this model.
        """
        if model_id in self._model_cache:
            return self.get(self._model_cache[model_id])

        # Query all providers in parallel
        provider_ids = list(self._factories.keys())
        instances = [self.get(pid) for pid in provider_ids]
        model_lists = await asyncio.gather(*[inst.list_models() for inst in instances])

        # Build cache and find match
        for provider_id, models in zip(provider_ids, model_lists):
            for mid in models:
                self._model_cache[mid] = provider_id

        if model_id in self._model_cache:
            return self.get(self._model_cache[model_id])
        raise KeyError(f"No provider found for model: {model_id}")

    async def client_for_model(self, model_id: str) -> ModelClient:
        """Get a model client for a given model ID.

        Convenience method that finds the provider and creates the client.

        Args:
            model_id: The model identifier (e.g., "claude-sonnet-4-20250514").

        Returns:
            A ModelClient configured for the specified model.

        Raises:
            KeyError: If no provider supports this model.
        """
        provider = await self.get_for_model(model_id)
        return provider.model(model_id)

    def is_registered(self, provider: Provider) -> bool:
        """Check if a provider is registered."""
        return provider in self._factories

    @property
    def providers(self) -> list[Provider]:
        """List all registered providers."""
        return list(self._factories.keys())


def create_default_hub() -> ProviderHub:
    """Create a hub with common providers pre-registered.

    Providers are wrapped with CachingProvider for efficient model listing.
    Only providers with their SDKs installed are registered.

    Returns:
        A ProviderHub with available providers registered.
    """
    from nous.llm.caching import CachingProvider
    from nous.llm.providers import OllamaProvider

    hub = ProviderHub()

    # Ollama uses httpx (always available)
    hub.register(Provider.OLLAMA, lambda: CachingProvider(OllamaProvider()))

    # Optional SDK-based providers
    try:
        from nous.llm.providers import AnthropicProvider
        hub.register(Provider.ANTHROPIC, lambda: CachingProvider(AnthropicProvider()))
    except ImportError:
        pass

    try:
        from nous.llm.providers import GeminiProvider
        hub.register(Provider.GOOGLE, lambda: CachingProvider(GeminiProvider()))
    except ImportError:
        pass

    try:
        from nous.llm.providers import OpenAIProvider
        hub.register(Provider.OPENAI, lambda: CachingProvider(OpenAIProvider()))
    except ImportError:
        pass

    return hub
