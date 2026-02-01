"""Tests for ConversationView protocol and MockConversationView."""

import pytest

from nous.types import TextContent, ImageContent, Message, ToolCall
from nous.view import ConversationView, MockConversationView


class TestMockConversationView:
    """Tests for MockConversationView implementation."""

    @pytest.mark.asyncio
    async def test_init_defaults(self):
        view = MockConversationView()
        assert await view.get_messages() == []

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        view = MockConversationView()
        msg1 = Message(role="user", content=[TextContent(text="Hello")])
        msg2 = Message(role="assistant", content=[TextContent(text="Hi!")])
        view.setup_message(msg1)
        view.setup_message(msg2)

        messages = await view.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self):
        view = MockConversationView()
        for i in range(5):
            view.setup_message(Message(role="user", content=[TextContent(text=f"msg{i}")]))

        messages = await view.get_messages(limit=2)
        assert len(messages) == 2
        assert messages[0].content[0].text == "msg3"
        assert messages[1].content[0].text == "msg4"

    @pytest.mark.asyncio
    async def test_on_text_delta(self):
        view = MockConversationView()
        await view.on_text_delta("Hello")
        await view.on_text_delta(" world")

        assert view.text_deltas == ["Hello", " world"]
        assert view.full_text == "Hello world"

    @pytest.mark.asyncio
    async def test_on_content_block(self):
        view = MockConversationView()
        block = ImageContent(mime_type="image/png", data="base64data")
        await view.on_content_block(block)

        assert len(view.content_blocks) == 1
        assert view.content_blocks[0].mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_call_tool_auto_approves(self):
        view = MockConversationView()
        tool_call = ToolCall(name="search", input={"query": "test"})
        result = await view.call_tool(tool_call)

        assert len(view.tool_calls) == 1
        assert view.tool_calls[0].name == "search"
        assert result.tool_call_id == tool_call.id
        assert not result.is_error
        assert "Mock result for search" in result.content[0].text

    @pytest.mark.asyncio
    async def test_add_message(self):
        view = MockConversationView()
        msg = Message(role="assistant", content=[TextContent(text="Done")])
        await view.add_message(msg)

        assert len(view.added_messages) == 1
        assert view.added_messages[0].role == "assistant"
        # Also persisted to get_messages
        assert len(await view.get_messages()) == 1

    @pytest.mark.asyncio
    async def test_on_turn_complete(self):
        view = MockConversationView()
        await view.on_turn_complete()

        assert view.turn_complete_count == 1

    @pytest.mark.asyncio
    async def test_clear_events(self):
        view = MockConversationView()
        await view.on_text_delta("text")
        await view.on_content_block(ImageContent(mime_type="image/png"))
        await view.call_tool(ToolCall(name="test", input={}))
        await view.add_message(Message(role="assistant", content=[]))
        await view.on_turn_complete()

        view.clear_events()

        assert view.text_deltas == []
        assert view.content_blocks == []
        assert view.tool_calls == []
        assert view.added_messages == []
        assert view.turn_complete_count == 0


class TestProtocolCompliance:
    """Verify MockConversationView satisfies ConversationView protocol."""

    def test_implements_protocol(self):
        view = MockConversationView()
        # Type checker verifies this at compile time, but runtime check too
        assert hasattr(view, "get_messages")
        assert hasattr(view, "on_text_delta")
        assert hasattr(view, "on_content_block")
        assert hasattr(view, "call_tool")
        assert hasattr(view, "add_message")
        assert hasattr(view, "on_turn_complete")

    @pytest.mark.asyncio
    async def test_protocol_method_signatures(self):
        view = MockConversationView()
        # Verify return types match protocol
        messages = await view.get_messages()
        assert isinstance(messages, list)
