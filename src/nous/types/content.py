"""Content block types - the atomic units of conversation."""

from typing import Literal
from pydantic import BaseModel


class TextContent(BaseModel):
    """Plain text content."""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content with optional attachment reference."""
    type: Literal["image"] = "image"
    mime_type: str
    attachment_id: str | None = None
    data: str | None = None  # Base64, avoid storing in DB


class AudioContent(BaseModel):
    """Audio content."""
    type: Literal["audio"] = "audio"
    mime_type: str
    attachment_id: str | None = None
    data: str | None = None


# Union type for all content blocks
ContentBlock = TextContent | ImageContent | AudioContent
