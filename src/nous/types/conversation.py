"""Conversation types - messages, threads, and conversations.

Example:
    >>> from nous.types import Message, TextContent
    >>> msg = Message(
    ...     role="user",
    ...     content=[TextContent(text="Hello!")],
    ... )
    >>> msg.role
    'user'
    >>> msg.id  # Auto-generated UUID
    '...'
"""

from datetime import datetime, UTC
from typing import Literal
from pydantic import BaseModel, Field
from uuid import uuid4

from nous.types.content import ContentBlock


class Message(BaseModel):
    """A single message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock]
    provider: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Thread(BaseModel):
    """A thread of messages, supports branching."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    parent_thread_id: str | None = None
    branch_point_message_id: str | None = None
    messages: list[Message] = Field(default_factory=list)


class Conversation(BaseModel):
    """A conversation containing one or more threads."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = ""
    user_id: str | None = None
    root_thread_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
