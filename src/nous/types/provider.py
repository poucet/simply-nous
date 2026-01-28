"""Provider types - LLM provider abstractions."""

from typing import Protocol
from nous.types.conversation import Message


class IProvider(Protocol):
    """Interface for LLM providers."""

    async def generate(
        self,
        messages: list[Message],
        model: str | None = None,
        **kwargs,
    ) -> Message:
        """Generate a response from the LLM."""
        ...
