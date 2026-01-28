"""AgentRunner - ConversationView wrapper with dynamic RAG.

AgentRunner wraps any ConversationView and intercepts get_messages() to:
1. Extract a query from recent messages
2. Fetch relevant knowledge using a KnowledgeFetcher
3. Build a system prompt with base_prompt + knowledge
4. Prepend the system message to the conversation

This allows the system prompt to be dynamically recomputed on every turn,
incorporating fresh RAG results.

Example:
    >>> runner = AgentRunner(
    ...     view=my_view,
    ...     knowledge_fetcher=my_fetcher,
    ...     base_prompt="You are a helpful assistant.",
    ... )
    >>> # Runner implements ConversationView, pass it to engine
    >>> response = await engine.run_turn(client, runner)
"""

from typing import Protocol, Callable, Awaitable

from nous.engine.context import ContextBuilder, DefaultContextBuilder
from nous.types import Message, TextContent, KnowledgeChunk
from nous.types.content import ContentBlock
from nous.types.tool import ToolCall, ToolResult
from nous.view.protocol import ConversationView


class KnowledgeFetcher(Protocol):
    """Protocol for knowledge retrieval.

    Apps implement this to provide their RAG backend.
    """

    async def fetch(self, query: str) -> list[KnowledgeChunk]:
        """Fetch relevant knowledge chunks for a query.

        Args:
            query: The search query extracted from conversation.

        Returns:
            List of relevant knowledge chunks.
        """
        ...


class AgentRunner:
    """ConversationView wrapper with dynamic system prompt and RAG.

    Wraps an underlying ConversationView and intercepts get_messages()
    to dynamically build the system prompt with fresh knowledge.

    RAG is triggered only when there's a new user message, not on every
    tool call iteration. The system prompt is cached until the next user turn.

    All other ConversationView methods delegate to the wrapped view.
    """

    def __init__(
        self,
        view: ConversationView,
        knowledge_fetcher: KnowledgeFetcher | None = None,
        base_prompt: str | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """Initialize the AgentRunner.

        Args:
            view: The underlying ConversationView to wrap.
            knowledge_fetcher: Optional RAG backend. If None, no knowledge is fetched.
            base_prompt: Optional base system prompt.
            context_builder: Optional custom context builder. Defaults to DefaultContextBuilder.
        """
        self._view = view
        self._knowledge_fetcher = knowledge_fetcher
        self._base_prompt = base_prompt
        self._context_builder = context_builder or DefaultContextBuilder()
        self._cached_system_prompt: str | None = None
        self._needs_rag: bool = True

    async def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages with dynamically computed system prompt.

        RAG runs once at the start of each turn. The system prompt is cached
        until on_turn_complete() signals the turn is done.

        Args:
            limit: Maximum number of messages to return. None means all.

        Returns:
            List of messages with system message prepended.
        """
        messages = await self._view.get_messages(limit)

        if self._needs_rag:
            self._needs_rag = False
            self._cached_system_prompt = await self._build_system_prompt(messages)

        if self._cached_system_prompt:
            system_message = Message(
                role="system",
                content=[TextContent(text=self._cached_system_prompt)],
            )
            return [system_message] + messages

        return messages

    def invalidate_cache(self) -> None:
        """Force re-RAG on next get_messages() call."""
        self._needs_rag = True
        self._cached_system_prompt = None

    async def _build_system_prompt(self, messages: list[Message]) -> str | None:
        """Build the system prompt with optional knowledge.

        Args:
            messages: Current conversation messages.

        Returns:
            The complete system prompt, or None if no prompt.
        """
        if not self._knowledge_fetcher:
            return self._base_prompt

        query = self._context_builder.build_query(messages)
        if not query:
            return self._base_prompt

        chunks = await self._knowledge_fetcher.fetch(query)
        if not chunks:
            return self._base_prompt

        knowledge = self._context_builder.format_knowledge(chunks)
        new_prompt, _ = self._context_builder.inject_knowledge(
            self._base_prompt, messages, knowledge
        )
        return new_prompt

    # Delegate all other methods to wrapped view

    async def on_text_delta(self, text: str) -> None:
        """Delegate to wrapped view."""
        await self._view.on_text_delta(text)

    async def on_content_block(self, block: ContentBlock) -> None:
        """Delegate to wrapped view."""
        await self._view.on_content_block(block)

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Delegate to wrapped view."""
        return await self._view.call_tool(tool_call)

    async def add_message(self, message: Message) -> None:
        """Delegate to wrapped view."""
        await self._view.add_message(message)

    async def on_turn_complete(self) -> None:
        """Delegate to wrapped view and reset RAG flag for next turn."""
        await self._view.on_turn_complete()
        self._needs_rag = True
