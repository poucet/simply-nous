"""ConversationView protocol - bidirectional channel between engine and client.

The engine reads messages via get_messages().
The engine pushes events via on_text_delta(), on_content_block(), call_tool(),
add_message(), and on_turn_complete().

Note: The engine is agnostic to system prompts and RAG - those are handled
at a higher level before calling engine.run_turn().

Callback purposes:
- on_text_delta: Streaming text for real-time display
- on_content_block: Non-text content (tool calls) for display
- call_tool: Execute tool and return result
- add_message: Persist message to history (view handles UX internally)
- on_turn_complete: Entire agent turn done, ready for user input

Example implementation:
    class MyView:
        async def get_messages(self, limit=None): return self._messages

        async def on_text_delta(self, text): print(text, end="")
        async def on_content_block(self, block): pass
        async def call_tool(self, call): return ToolResult(...)
        async def add_message(self, msg):
            self._messages.append(msg)
            self._update_display(msg)  # View handles UX
        async def on_turn_complete(self): self.input_enabled = True
"""

from typing import Protocol

from nous.types.content import ContentBlock
from nous.types.conversation import Message
from nous.types.tool import ToolCall, ToolResult


class ConversationView(Protocol):
    """Bidirectional channel between engine and client.

    The engine uses this protocol to:
    - Read: Pull conversation messages
    - Write: Push events (text deltas, content blocks, tool calls)

    Implementations handle the transport layer (Discord, WebSocket, CLI, etc.)
    and can implement custom approval flows for tool calls.
    """

    # Read: Engine pulls state

    async def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages from the conversation.

        Args:
            limit: Maximum number of messages to return. None means all.

        Returns:
            List of messages, oldest first.
        """
        ...

    # Write: Engine pushes events

    async def on_text_delta(self, text: str) -> None:
        """Called when streaming text arrives.

        Args:
            text: The text chunk received from the model.
        """
        ...

    async def on_content_block(self, block: ContentBlock) -> None:
        """Called when a complete non-text content block is ready.

        Args:
            block: The completed content block (image, audio, etc.)
        """
        ...

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call requested by the model.

        The view is responsible for:
        - Displaying the tool call to the user
        - Handling approval (auto-approve, Discord DM, WebSocket UI, etc.)
        - Executing the tool or returning a rejection

        Args:
            tool_call: The tool call requested by the model.

        Returns:
            The result of executing (or rejecting) the tool call.
        """
        ...

    async def add_message(self, message: Message) -> None:
        """Persist a message to conversation history.

        Called for both assistant messages and tool result messages.
        The view should store these so get_messages() returns them.

        Args:
            message: The message to add (assistant or user with tool results).
        """
        ...

    async def on_turn_complete(self) -> None:
        """Called when the entire turn is complete.

        Signifies that all tool calls have been resolved and the final
        assistant message is ready. Useful for:
        - Re-enabling user input
        - Committing storage transactions
        - Analytics/logging
        """
        ...
