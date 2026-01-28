"""ConversationView protocol - bidirectional channel between engine and client."""

from typing import Protocol

from nous.types.content import ContentBlock
from nous.types.conversation import Message
from nous.types.tool import ToolCall, ToolResult


class ConversationView(Protocol):
    """Bidirectional channel between engine and client.

    The engine uses this protocol to:
    - Read: Pull conversation state (messages, system prompt, model)
    - Write: Push events (text deltas, content blocks, tool calls)

    Implementations handle the transport layer (Discord, WebSocket, CLI, etc.)
    and can implement custom approval flows for tool calls.
    """

    # Read: Engine pulls state

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages from the conversation.

        Args:
            limit: Maximum number of messages to return. None means all.

        Returns:
            List of messages, oldest first.
        """
        ...

    def get_system_prompt(self) -> str | None:
        """Get the system prompt for this conversation.

        Returns:
            System prompt string, or None if not set.
        """
        ...

    @property
    def model_id(self) -> str:
        """The model identifier to use for this conversation."""
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

    async def on_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Called when the model requests a tool call.

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

    async def on_message_complete(self, message: Message) -> None:
        """Called when an assistant message is complete.

        Args:
            message: The completed assistant message with all content blocks.
        """
        ...
