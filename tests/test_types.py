"""Tests for nous types."""

from nous.types import TextContent, Message, ToolCall, ToolResult


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


def test_tool_call():
    call = ToolCall(name="search", input={"query": "test"})
    assert call.name == "search"
    assert call.input["query"] == "test"


def test_tool_result():
    result = ToolResult(
        tool_use_id="123",
        content=[TextContent(text="Found 10 results")],
        is_error=False
    )
    assert result.tool_use_id == "123"
    assert not result.is_error
