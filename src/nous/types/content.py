"""Content block types - the atomic units of conversation.

Example:
    >>> from nous.types import TextContent, ToolUseContent, Message
    >>> msg = Message(
    ...     role="assistant",
    ...     content=[
    ...         TextContent(text="Let me search for that."),
    ...         ToolUseContent(id="call_1", name="search", input={"q": "test"}),
    ...     ]
    ... )
    >>> msg.content[0].type
    'text'
"""

from typing import Any, Literal
from pydantic import BaseModel


class TextContent(BaseModel):
    """Plain text content."""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content with optional asset reference."""
    type: Literal["image"] = "image"
    mime_type: str
    asset_id: str | None = None
    data: str | None = None  # Base64, avoid storing in DB


class AudioContent(BaseModel):
    """Audio content."""
    type: Literal["audio"] = "audio"
    mime_type: str
    asset_id: str | None = None
    data: str | None = None


class ToolUseContent(BaseModel):
    """Tool use request from the AI model."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


# Content types that can appear in tool results
ToolContent = TextContent | ImageContent | AudioContent


class ToolResultContent(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: list[ToolContent] = []
    is_error: bool = False


# Union type for all content blocks
ContentBlock = TextContent | ImageContent | AudioContent | ToolUseContent | ToolResultContent
