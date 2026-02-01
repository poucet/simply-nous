"""Tests for nous types."""

from nous.types import (
    TextContent,
    Message,
    ToolCall,
    ToolResult,
    ToolUseContent,
    ToolResultContent,
)


def test_text_content():
    content = TextContent(text="Hello, world!")
    assert content.type == "text"
    assert content.text == "Hello, world!"


def test_message():
    msg = Message(
        role="user",
        content=[TextContent(text="Hello")]
    )
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert msg.id is not None


def test_message_with_provider_and_model():
    """Verify Message has provider and model fields."""
    msg = Message(
        role="assistant",
        content=[TextContent(text="Hi")],
        provider="anthropic",
        model="claude-3-opus"
    )
    assert msg.provider == "anthropic"
    assert msg.model == "claude-3-opus"


def test_tool_call():
    call = ToolCall(name="search", input={"query": "test"})
    assert call.name == "search"
    assert call.input["query"] == "test"


def test_tool_result():
    result = ToolResult(
        tool_call_id="123",
        content=[TextContent(text="Found 10 results")],
        is_error=False
    )
    assert result.tool_call_id == "123"
    assert not result.is_error


def test_tool_use_content():
    """ToolUseContent as a content block in messages."""
    content = ToolUseContent(
        id="tool_123",
        name="search",
        input={"query": "test"}
    )
    assert content.type == "tool_use"
    assert content.id == "tool_123"
    assert content.name == "search"
    assert content.input == {"query": "test"}


def test_tool_result_content():
    """ToolResultContent as a content block in messages."""
    content = ToolResultContent(
        tool_call_id="tool_123",
        content=[TextContent(text="Found 10 results")],
        is_error=False
    )
    assert content.type == "tool_result"
    assert content.tool_call_id == "tool_123"
    assert len(content.content) == 1
    assert not content.is_error


def test_message_with_tool_use_content():
    """Message can contain ToolUseContent blocks."""
    msg = Message(
        role="assistant",
        content=[
            TextContent(text="Let me search for that."),
            ToolUseContent(id="call_1", name="search", input={"q": "test"})
        ]
    )
    assert len(msg.content) == 2
    assert msg.content[0].type == "text"
    assert msg.content[1].type == "tool_use"


def test_message_with_tool_result_content():
    """Message can contain ToolResultContent blocks."""
    msg = Message(
        role="user",
        content=[
            ToolResultContent(
                tool_call_id="call_1",
                content=[TextContent(text="Result data")]
            )
        ]
    )
    assert len(msg.content) == 1
    assert msg.content[0].type == "tool_result"


def test_content_block_serialization():
    """ContentBlock types serialize with discriminator type field."""
    msg = Message(
        role="assistant",
        content=[
            TextContent(text="Hello"),
            ToolUseContent(id="t1", name="foo", input={}),
        ]
    )
    data = msg.model_dump()
    assert data["content"][0]["type"] == "text"
    assert data["content"][1]["type"] == "tool_use"
