"""Mock implementation of ConversationView for testing.

Example:
    >>> from nous.view import MemoryConversationView
    >>> view = MemoryConversationView(model_id="test", system_prompt="Be helpful")
    >>> view.add_user_message("Hello!")
    >>> view.model_id
    'test'
    >>> await view.on_text_delta("Hi there")
    >>> view.full_text
    'Hi there'
"""

from typing import Callable, Awaitable

from nous.types.content import ContentBlock, TextContent
from nous.types.conversation import Message
from nous.types.knowledge import KnowledgeChunk
from nous.types.tool import ToolCall, ToolResult

ToolHandler = Callable[[ToolCall], Awaitable[ToolResult]]
KnowledgeHandler = Callable[[str], Awaitable[list[KnowledgeChunk] | None]]


class MemoryConversationView:
    """In-memory ConversationView for testing and simple use cases.

    Auto-approves all tool calls by default. Stores all events for inspection.
    Optionally accepts custom tool and knowledge handlers.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        system_prompt: str | None = None,
        tool_handler: ToolHandler | None = None,
        knowledge_handler: KnowledgeHandler | None = None,
    ) -> None:
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._tool_handler = tool_handler
        self._knowledge_handler = knowledge_handler
        self._messages: list[Message] = []
        self.text_deltas: list[str] = []
        self.content_blocks: list[ContentBlock] = []
        self.tool_calls: list[ToolCall] = []
        self.added_messages: list[Message] = []
        self.turn_complete_count: int = 0
        self.knowledge_queries: list[str] = []

    # Read: Engine pulls state

    def get_messages(self, limit: int | None = None) -> list[Message]:
        if limit is None:
            return list(self._messages)
        return list(self._messages[-limit:])

    def get_system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def model_id(self) -> str:
        return self._model_id

    # Write: Engine pushes events

    async def on_text_delta(self, text: str) -> None:
        self.text_deltas.append(text)

    async def on_content_block(self, block: ContentBlock) -> None:
        self.content_blocks.append(block)

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        self.tool_calls.append(tool_call)
        if self._tool_handler:
            return await self._tool_handler(tool_call)
        return ToolResult(
            tool_use_id=tool_call.id,
            content=[TextContent(text=f"Mock result for {tool_call.name}")],
        )

    async def add_message(self, message: Message) -> None:
        """Persist message to conversation history."""
        self.added_messages.append(message)
        self._messages.append(message)

    async def on_turn_complete(self) -> None:
        """Signal: entire turn is complete."""
        self.turn_complete_count += 1

    async def fetch_knowledge(self, query: str) -> list[KnowledgeChunk] | None:
        self.knowledge_queries.append(query)
        if self._knowledge_handler:
            return await self._knowledge_handler(query)
        return None

    # Convenience methods

    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation."""
        self._messages.append(
            Message(role="user", content=[TextContent(text=text)])
        )

    def setup_message(self, message: Message) -> None:
        """Add a message to the conversation (for test setup, bypasses engine)."""
        self._messages.append(message)

    def clear_events(self) -> None:
        """Clear all recorded events."""
        self.text_deltas.clear()
        self.content_blocks.clear()
        self.tool_calls.clear()
        self.added_messages.clear()
        self.turn_complete_count = 0
        self.knowledge_queries.clear()

    @property
    def full_text(self) -> str:
        """Concatenate all text deltas received."""
        return "".join(self.text_deltas)

    @property
    def messages(self) -> list[Message]:
        """All messages in the conversation."""
        return list(self._messages)


# Alias for clarity in different contexts
MockConversationView = MemoryConversationView
