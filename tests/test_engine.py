"""Tests for the Engine - storage-agnostic conversation engine."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from nous.engine.engine import Engine
from nous.engine.context import DefaultContextBuilder
from nous.llm.events import (
    TextDeltaEvent,
    ToolCallEvent,
    MessageCompleteEvent,
    StreamEvent,
)
from nous.types import (
    Message,
    TextContent,
    ToolUseContent,
    ToolCall,
    ToolResult,
    ToolDefinition,
    KnowledgeChunk,
)
from nous.view.memory import MemoryConversationView


class MockModelClient:
    """Mock model client that yields predetermined events."""

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        """Initialize with a list of responses (one per call)."""
        self._responses = responses
        self._call_index = 0
        self.complete_calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Return events from the next predetermined response."""
        self.complete_calls.append({
            "messages": messages,
            "tools": tools,
            "stream": stream,
        })
        events = self._responses[self._call_index]
        self._call_index += 1

        async def generate() -> AsyncIterator[StreamEvent]:
            for event in events:
                yield event

        return generate()


def make_text_response(text: str) -> list[StreamEvent]:
    """Helper to create a simple text response."""
    return [
        TextDeltaEvent(text=text),
        MessageCompleteEvent(
            message=Message(role="assistant", content=[TextContent(text=text)])
        ),
    ]


def make_tool_response(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    text_before: str | None = None,
) -> list[StreamEvent]:
    """Helper to create a tool call response."""
    events: list[StreamEvent] = []
    content: list[TextContent | ToolUseContent] = []

    if text_before:
        events.append(TextDeltaEvent(text=text_before))
        content.append(TextContent(text=text_before))

    tool_call = ToolCall(id=tool_id, name=tool_name, input=tool_input)
    events.append(ToolCallEvent(tool_call=tool_call))
    content.append(ToolUseContent(id=tool_id, name=tool_name, input=tool_input))

    events.append(
        MessageCompleteEvent(message=Message(role="assistant", content=content))
    )
    return events


class TestEngineBasics:
    """Basic engine functionality tests."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Engine streams text and completes message."""
        client = MockModelClient([make_text_response("Hello, world!")])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("Hi")

        result = await engine.run_turn(client, view)

        assert result.role == "assistant"
        assert result.content[0].text == "Hello, world!"
        assert view.full_text == "Hello, world!"
        assert len(view.added_messages) == 1

    @pytest.mark.asyncio
    async def test_passes_messages(self):
        """Engine passes conversation messages to client."""
        client = MockModelClient([make_text_response("OK")])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("First message")
        view.setup_message(
            Message(role="assistant", content=[TextContent(text="First response")])
        )
        view.add_user_message("Second message")

        await engine.run_turn(client, view)

        passed_messages = client.complete_calls[0]["messages"]
        assert len(passed_messages) == 3
        assert passed_messages[0].role == "user"
        assert passed_messages[1].role == "assistant"
        assert passed_messages[2].role == "user"

    @pytest.mark.asyncio
    async def test_passes_tools(self):
        """Engine passes tool definitions to client."""
        client = MockModelClient([make_text_response("OK")])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("Search for something")

        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {}},
            )
        ]
        await engine.run_turn(client, view, tools=tools)

        assert client.complete_calls[0]["tools"] == tools


class TestStreamingCallbacks:
    """Tests for streaming callback flow."""

    @pytest.mark.asyncio
    async def test_text_deltas_streamed(self):
        """Text deltas are streamed to view."""
        events: list[StreamEvent] = [
            TextDeltaEvent(text="Hello"),
            TextDeltaEvent(text=" "),
            TextDeltaEvent(text="world"),
            MessageCompleteEvent(
                message=Message(
                    role="assistant", content=[TextContent(text="Hello world")]
                )
            ),
        ]
        client = MockModelClient([events])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("Hi")

        await engine.run_turn(client, view)

        assert view.text_deltas == ["Hello", " ", "world"]
        assert view.full_text == "Hello world"

    @pytest.mark.asyncio
    async def test_content_blocks_reported(self):
        """Non-text content blocks (tool calls) are reported to view."""
        tool_id = "call_123"
        events = make_tool_response(tool_id, "search", {"query": "test"})

        # Add a second response after tool result
        final_events = make_text_response("Found results")

        client = MockModelClient([events, final_events])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("Search for test")

        await engine.run_turn(client, view)

        # The ToolUseContent block should be reported via on_content_block
        assert len(view.content_blocks) == 1
        assert view.content_blocks[0].id == tool_id


class TestToolCallFlow:
    """Tests for tool call → result → continue flow."""

    @pytest.mark.asyncio
    async def test_tool_call_auto_approved(self):
        """Tool calls are auto-approved by MemoryConversationView."""
        tool_id = "call_abc"
        events = make_tool_response(tool_id, "get_weather", {"city": "Seattle"})
        final_events = make_text_response("It's sunny in Seattle!")

        client = MockModelClient([events, final_events])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("What's the weather in Seattle?")

        result = await engine.run_turn(client, view)

        assert len(view.tool_calls) == 1
        assert view.tool_calls[0].name == "get_weather"
        assert result.content[0].text == "It's sunny in Seattle!"

    @pytest.mark.asyncio
    async def test_tool_result_sent_back(self):
        """Tool result is sent back to the model."""
        tool_id = "call_xyz"
        events = make_tool_response(tool_id, "calculator", {"expr": "2+2"})
        final_events = make_text_response("The answer is 4")

        client = MockModelClient([events, final_events])
        engine = Engine()

        async def custom_handler(tc: ToolCall) -> ToolResult:
            return ToolResult(
                tool_use_id=tc.id, content=[TextContent(text="4")], is_error=False
            )

        view = MemoryConversationView(tool_handler=custom_handler)
        view.add_user_message("What is 2+2?")

        await engine.run_turn(client, view)

        # Second call should include tool result in messages
        second_call_messages = client.complete_calls[1]["messages"]
        last_msg = second_call_messages[-1]
        assert last_msg.role == "user"
        assert last_msg.content[0].type == "tool_result"
        assert last_msg.content[0].content[0].text == "4"

    @pytest.mark.asyncio
    async def test_tool_error_handled(self):
        """Tool errors are properly handled."""
        tool_id = "call_err"
        events = make_tool_response(tool_id, "failing_tool", {})
        final_events = make_text_response("Tool failed, let me try something else")

        client = MockModelClient([events, final_events])
        engine = Engine()

        async def error_handler(tc: ToolCall) -> ToolResult:
            return ToolResult(
                tool_use_id=tc.id,
                content=[TextContent(text="Error: Connection failed")],
                is_error=True,
            )

        view = MemoryConversationView(tool_handler=error_handler)
        view.add_user_message("Try the failing tool")

        await engine.run_turn(client, view)

        # Verify error was passed back to model
        second_call_messages = client.complete_calls[1]["messages"]
        tool_result = second_call_messages[-1].content[0]
        assert tool_result.is_error is True

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_response(self):
        """Multiple tool calls in one response are all executed."""
        # Response with two tool calls
        events: list[StreamEvent] = [
            ToolCallEvent(tool_call=ToolCall(id="call_1", name="tool_a", input={})),
            ToolCallEvent(tool_call=ToolCall(id="call_2", name="tool_b", input={})),
            MessageCompleteEvent(
                message=Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="call_1", name="tool_a", input={}),
                        ToolUseContent(id="call_2", name="tool_b", input={}),
                    ],
                )
            ),
        ]
        final_events = make_text_response("Both tools executed")

        client = MockModelClient([events, final_events])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("Use both tools")

        await engine.run_turn(client, view)

        assert len(view.tool_calls) == 2
        assert view.tool_calls[0].name == "tool_a"
        assert view.tool_calls[1].name == "tool_b"


class TestMultiTurnConversation:
    """Tests for multi-turn conversations."""

    @pytest.mark.asyncio
    async def test_multiple_turns_preserve_history(self):
        """Multiple turns preserve conversation history."""
        client = MockModelClient([
            make_text_response("I'm Claude"),
            make_text_response("Your name is Alice"),
        ])
        engine = Engine()

        view = MemoryConversationView()

        # First turn
        view.add_user_message("What's your name?")
        await engine.run_turn(client, view)

        # Second turn
        view.add_user_message("What's MY name?")
        await engine.run_turn(client, view)

        # Second call should have: user1, assistant1, user2
        second_messages = client.complete_calls[1]["messages"]
        assert len(second_messages) == 3
        assert second_messages[0].content[0].text == "What's your name?"
        assert second_messages[1].content[0].text == "I'm Claude"
        assert second_messages[2].content[0].text == "What's MY name?"

    @pytest.mark.asyncio
    async def test_tool_messages_in_history(self):
        """Tool call/result messages are preserved in history."""
        tool_id = "call_hist"
        events = make_tool_response(tool_id, "remember", {"fact": "test"})
        final_events = make_text_response("I'll remember that")
        second_turn = make_text_response("Yes, I remember")

        client = MockModelClient([events, final_events, second_turn])
        engine = Engine()

        view = MemoryConversationView()

        # First turn with tool call
        view.add_user_message("Remember 'test'")
        await engine.run_turn(client, view)

        # Second turn
        view.add_user_message("Do you remember?")
        await engine.run_turn(client, view)

        # Third call (second turn) should have full history:
        # user, assistant (tool), user (tool result), assistant (final), user
        third_messages = client.complete_calls[2]["messages"]
        assert len(third_messages) == 5
        assert third_messages[0].role == "user"
        assert third_messages[1].content[0].type == "tool_use"
        assert third_messages[2].content[0].type == "tool_result"
        assert third_messages[3].content[0].text == "I'll remember that"
        assert third_messages[4].content[0].text == "Do you remember?"


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_stream_without_complete_event_raises(self):
        """RuntimeError raised if stream ends without MessageCompleteEvent."""
        events: list[StreamEvent] = [TextDeltaEvent(text="partial")]

        client = MockModelClient([events])
        engine = Engine()

        view = MemoryConversationView()
        view.add_user_message("Test")

        with pytest.raises(RuntimeError, match="MessageCompleteEvent"):
            await engine.run_turn(client, view)


class TestContextBuilder:
    """Tests for the DefaultContextBuilder."""

    def test_build_query_from_last_user_message(self):
        """Query is extracted from last user message."""
        builder = DefaultContextBuilder()
        messages = [
            Message(role="user", content=[TextContent(text="First question")]),
            Message(role="assistant", content=[TextContent(text="Answer")]),
            Message(role="user", content=[TextContent(text="Follow-up question")]),
        ]

        query = builder.build_query(messages)

        assert query == "Follow-up question"

    def test_build_query_returns_none_for_empty(self):
        """Query returns None for empty messages."""
        builder = DefaultContextBuilder()

        query = builder.build_query([])

        assert query is None

    def test_format_knowledge_numbered(self):
        """Knowledge chunks are formatted as numbered list."""
        builder = DefaultContextBuilder()
        chunks = [
            KnowledgeChunk(content=[TextContent(text="Fact A")], source="a.md"),
            KnowledgeChunk(content=[TextContent(text="Fact B")], source="b.md"),
        ]

        formatted = builder.format_knowledge(chunks)

        assert "[1] (source: a.md)" in formatted
        assert "Fact A" in formatted
        assert "[2] (source: b.md)" in formatted
        assert "Fact B" in formatted

    def test_inject_knowledge_appends_to_system(self):
        """Knowledge is appended to system prompt."""
        builder = DefaultContextBuilder()
        messages = [Message(role="user", content=[TextContent(text="Q")])]

        new_system, new_messages = builder.inject_knowledge(
            "Original system", messages, "Knowledge content"
        )

        assert "Original system" in new_system
        assert "Relevant Knowledge" in new_system
        assert "Knowledge content" in new_system
        assert new_messages == messages

    def test_inject_knowledge_creates_system_if_none(self):
        """Knowledge section becomes system prompt if none exists."""
        builder = DefaultContextBuilder()
        messages = [Message(role="user", content=[TextContent(text="Q")])]

        new_system, _ = builder.inject_knowledge(None, messages, "Knowledge")

        assert "Relevant Knowledge" in new_system
        assert "Knowledge" in new_system
