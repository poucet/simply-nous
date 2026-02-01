"""Knowledge types for RAG context.

Example:
    >>> from nous.types import KnowledgeChunk, TextContent, ImageContent
    >>> chunk = KnowledgeChunk(
    ...     content=[
    ...         TextContent(text="Q4 revenue increased 15%..."),
    ...         ImageContent(mime_type="image/png", data="..."),
    ...     ],
    ...     source="gdoc://abc123",
    ...     metadata={"tab": "Financials", "doc_title": "Q4 Report"}
    ... )
"""

from typing import Any

from pydantic import BaseModel

from nous.types.content import ContentBlock, TextContent


class KnowledgeChunk(BaseModel):
    """Retrieved chunk for RAG context.

    Supports multimodal content via ContentBlock list.
    Metadata preserves document structure (tabs, sections, etc.)
    without enforcing a specific schema.
    """

    content: list[ContentBlock]
    """Chunk content - text, images, audio, etc."""

    source: str | None = None
    """Source reference (URL, file path, doc ID, etc.)"""

    score: float | None = None
    """Relevance score from retrieval system."""

    metadata: dict[str, Any] = {}
    """Additional metadata (tab name, chunk index, title, etc.)"""

    def get_text(self) -> str:
        """Extract all text content from this chunk.

        Concatenates text from all TextContent blocks, separated by newlines.

        Raises:
            ValueError: If the chunk contains no TextContent block.
        """
        texts = [block.text for block in self.content if isinstance(block, TextContent)]
        if not texts:
            raise ValueError("KnowledgeChunk contains no TextContent block")
        return "\n".join(texts)
