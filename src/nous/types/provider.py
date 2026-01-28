"""Provider types - LLM provider identification.

Example:
    >>> from nous.types import Provider
    >>> provider = Provider.ANTHROPIC
    >>> provider.value
    'anthropic'
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    MISTRAL = "mistral"
