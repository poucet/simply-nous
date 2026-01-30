"""Storage interfaces - protocols for conversation and knowledge stores."""

from typing import Protocol
from nous.types.conversation import Message, Conversation


class IConversationStore(Protocol):
    """Interface for conversation storage."""

    async def save_message(self, thread_id: str, message: Message) -> str:
        """Save a message to a thread, return message ID."""
        ...

    async def get_history(self, thread_id: str, limit: int = 50) -> list[Message]:
        """Get message history for a thread."""
        ...

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        ...
