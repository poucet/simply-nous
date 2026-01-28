"""Tool types - calls and results."""

from typing import Any
from pydantic import BaseModel, Field
from uuid import uuid4

from nous.types.content import ContentBlock


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool."""
    tool_use_id: str
    content: list[ContentBlock]
    is_error: bool = False
