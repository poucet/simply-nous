"""Context building for LLM requests.

ContextBuilder is a pluggable strategy for:
- Extracting RAG queries from conversation
- Formatting retrieved knowledge
- Injecting knowledge into the context

Example:
    >>> from nous.engine.context import ContextBuilder, DefaultContextBuilder
    >>> builder = DefaultContextBuilder()
    >>> query = builder.build_query(messages)
    >>> formatted = builder.format_knowledge(chunks)
"""

from typing import Protocol

from nous.types import Message, TextContent, KnowledgeChunk


class ContextBuilder(Protocol):
    """Strategy for building LLM context with knowledge injection."""

    def build_query(self, messages: list[Message]) -> str | None:
        """Extract a RAG query from conversation messages.

        Args:
            messages: Conversation history.

        Returns:
            Query string for retrieval, or None to skip RAG.
        """
        ...

    def format_knowledge(self, chunks: list[KnowledgeChunk]) -> str:
        """Format retrieved knowledge chunks into text.

        Args:
            chunks: Retrieved knowledge chunks.

        Returns:
            Formatted string to inject into context.
        """
        ...

    def inject_knowledge(
        self,
        system_prompt: str | None,
        messages: list[Message],
        knowledge: str,
    ) -> tuple[str | None, list[Message]]:
        """Inject formatted knowledge into the context.

        Args:
            system_prompt: Current system prompt.
            messages: Conversation messages.
            knowledge: Formatted knowledge string.

        Returns:
            Tuple of (new_system_prompt, new_messages).
        """
        ...


class DefaultContextBuilder:
    """Default context builder with sensible defaults.

    - Query: extracts text from last user message
    - Format: numbered list with sources
    - Inject: appends to system prompt
    """

    def build_query(self, messages: list[Message]) -> str | None:
        """Extract query from last user message."""
        for msg in reversed(messages):
            if msg.role == "user":
                parts = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        parts.append(block.text)
                if parts:
                    return " ".join(parts)
        return None

    def format_knowledge(self, chunks: list[KnowledgeChunk]) -> str:
        """Format as numbered list with sources."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text_parts = []
            for block in chunk.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
            if text_parts:
                source = f" (source: {chunk.source})" if chunk.source else ""
                parts.append(f"[{i}]{source}\n{chr(10).join(text_parts)}")
        return "\n\n".join(parts)

    def inject_knowledge(
        self,
        system_prompt: str | None,
        messages: list[Message],
        knowledge: str,
    ) -> tuple[str | None, list[Message]]:
        """Append knowledge to system prompt."""
        knowledge_section = f"## Relevant Knowledge\n\n{knowledge}"

        if system_prompt:
            new_prompt = f"{system_prompt}\n\n{knowledge_section}"
        else:
            new_prompt = knowledge_section

        return new_prompt, messages
