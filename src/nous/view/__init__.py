"""View package - bidirectional channel between engine and client."""

from nous.view.protocol import ConversationView
from nous.view.mock import MockConversationView

__all__ = ["ConversationView", "MockConversationView"]
