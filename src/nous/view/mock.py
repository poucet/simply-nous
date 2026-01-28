"""Mock implementation of ConversationView for testing."""

from nous.types.content import ContentBlock, TextContent
from nous.types.conversation import Message
from nous.types.tool import ToolCall, ToolResult


class MockConversationView:
    """In-memory ConversationView for testing.

    Auto-approves all tool calls and stores all events for inspection.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        system_prompt: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self.text_deltas: list[str] = []
        self.content_blocks: list[ContentBlock] = []
        self.tool_calls: list[ToolCall] = []
        self.completed_messages: list[Message] = []

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

    async def on_tool_call(self, tool_call: ToolCall) -> ToolResult:
        self.tool_calls.append(tool_call)
        return ToolResult(
            tool_use_id=tool_call.id,
            content=[TextContent(text=f"Mock result for {tool_call.name}")],
        )

    async def on_message_complete(self, message: Message) -> None:
        self.completed_messages.append(message)

    # Test helpers

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation (for test setup)."""
        self._messages.append(message)

    def clear_events(self) -> None:
        """Clear all recorded events."""
        self.text_deltas.clear()
        self.content_blocks.clear()
        self.tool_calls.clear()
        self.completed_messages.clear()

    @property
    def full_text(self) -> str:
        """Concatenate all text deltas received."""
        return "".join(self.text_deltas)
