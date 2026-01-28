"""Provider configuration management.

Handles API keys, model defaults, and provider settings.

Example:
    >>> from nous.llm.config import ProviderConfig, get_api_key
    >>> from nous.types import Provider
    >>>
    >>> # Get API key from env var
    >>> key = get_api_key(Provider.ANTHROPIC)
    >>>
    >>> # Configure provider defaults
    >>> config = ProviderConfig(
    ...     default_model="claude-sonnet-4-20250514",
    ...     max_tokens=4096,
    ... )
"""

import os
from dataclasses import dataclass, field
from typing import Any

from nous.types import Provider

# Environment variable names for each provider
API_KEY_ENV_VARS: dict[Provider, list[str]] = {
    Provider.ANTHROPIC: ["ANTHROPIC_API_KEY"],
    Provider.OPENAI: ["OPENAI_API_KEY"],
    Provider.GOOGLE: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    Provider.MISTRAL: ["MISTRAL_API_KEY"],
    Provider.OPENROUTER: ["OPENROUTER_API_KEY"],
    Provider.OLLAMA: [],  # No API key needed
}

# Default models for each provider
DEFAULT_MODELS: dict[Provider, str] = {
    Provider.ANTHROPIC: "claude-sonnet-4-20250514",
    Provider.OPENAI: "gpt-4o",
    Provider.GOOGLE: "gemini-2.0-flash",
    Provider.MISTRAL: "mistral-large-latest",
    Provider.OPENROUTER: "anthropic/claude-sonnet-4",
    Provider.OLLAMA: "llama3.2",
}


def get_api_key(provider: Provider) -> str | None:
    """Get API key for a provider from environment variables.

    Checks the standard environment variable(s) for the provider.

    Args:
        provider: The provider to get the key for.

    Returns:
        The API key if found, None otherwise.
    """
    env_vars = API_KEY_ENV_VARS.get(provider, [])
    for var in env_vars:
        key = os.environ.get(var)
        if key:
            return key
    return None


def get_default_model(provider: Provider) -> str:
    """Get the default model for a provider.

    Args:
        provider: The provider to get the default model for.

    Returns:
        The default model ID.
    """
    return DEFAULT_MODELS.get(provider, "")


@dataclass
class ProviderConfig:
    """Configuration for a single provider.

    Attributes:
        api_key: API key (if not using env var).
        base_url: Custom API base URL.
        default_model: Default model to use.
        max_tokens: Default max tokens for completions.
        timeout: Request timeout in seconds.
        extra: Additional provider-specific settings.
    """
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    max_tokens: int = 4096
    timeout: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HubConfig:
    """Configuration for the ProviderHub.

    Allows configuring multiple providers at once.

    Example:
        >>> config = HubConfig()
        >>> config.set(Provider.ANTHROPIC, ProviderConfig(max_tokens=8192))
        >>> config.set(Provider.OPENAI, ProviderConfig(default_model="gpt-4-turbo"))
    """
    _providers: dict[Provider, ProviderConfig] = field(default_factory=dict)

    def set(self, provider: Provider, config: ProviderConfig) -> None:
        """Set configuration for a provider."""
        self._providers[provider] = config

    def get(self, provider: Provider) -> ProviderConfig:
        """Get configuration for a provider (returns defaults if not set)."""
        return self._providers.get(provider, ProviderConfig())

    def has(self, provider: Provider) -> bool:
        """Check if a provider has custom configuration."""
        return provider in self._providers
