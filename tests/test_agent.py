"""Tests for AgentRunner - ConversationView wrapper with RAG."""

import pytest

from nous.agent import AgentRunner, KnowledgeFetcher
from nous.types import Message, TextContent, KnowledgeChunk, ToolCall
from nous.view import MockConversationView


class MockKnowledgeFetcher:
    """Mock knowledge fetcher that returns predetermined chunks."""

    def __init__(self, chunks: list[KnowledgeChunk] | None = None):
        self.chunks = chunks or []
        self.fetch_calls: list[str] = []

    async def fetch(self, query: str) -> list[KnowledgeChunk]:
        self.fetch_calls.append(query)
        return self.chunks


class TestAgentRunnerBasics:
    """Basic AgentRunner functionality."""

    @pytest.mark.asyncio
    async def test_wraps_view_get_messages(self):
        """AgentRunner returns messages from wrapped view."""
        view = MockConversationView()
        view.add_user_message("Hello")
        runner = AgentRunner(view=view)

        messages = await runner.get_messages()

        assert len(messages) == 1
        assert messages[0].content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_prepends_system_message_with_base_prompt(self):
        """System message is prepended when base_prompt is set."""
        view = MockConversationView()
        view.add_user_message("Hello")
        runner = AgentRunner(view=view, base_prompt="You are helpful.")

        messages = await runner.get_messages()

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content[0].text == "You are helpful."
        assert messages[1].role == "user"

    @pytest.mark.asyncio
    async def test_no_system_message_without_base_prompt(self):
        """No system message when base_prompt is None and no knowledge."""
        view = MockConversationView()
        view.add_user_message("Hello")
        runner = AgentRunner(view=view)

        messages = await runner.get_messages()

        assert len(messages) == 1
        assert messages[0].role == "user"


class TestAgentRunnerRAG:
    """Tests for RAG integration."""

    @pytest.mark.asyncio
    async def test_fetches_knowledge_on_first_call(self):
        """Knowledge is fetched on first get_messages call."""
        view = MockConversationView()
        view.add_user_message("What is Python?")

        chunks = [KnowledgeChunk(content=[TextContent(text="Python is a language")])]
        fetcher = MockKnowledgeFetcher(chunks)

        runner = AgentRunner(
            view=view,
            knowledge_fetcher=fetcher,
            base_prompt="You are helpful.",
        )

        messages = await runner.get_messages()

        assert len(fetcher.fetch_calls) == 1
        assert fetcher.fetch_calls[0] == "What is Python?"
        assert "Python is a language" in messages[0].content[0].text

    @pytest.mark.asyncio
    async def test_caches_knowledge_within_turn(self):
        """Knowledge is NOT re-fetched on subsequent calls within same turn."""
        view = MockConversationView()
        view.add_user_message("Question")

        chunks = [KnowledgeChunk(content=[TextContent(text="Answer")])]
        fetcher = MockKnowledgeFetcher(chunks)

        runner = AgentRunner(view=view, knowledge_fetcher=fetcher)

        # Simulate multiple get_messages calls (like tool call loop)
        await runner.get_messages()
        await runner.get_messages()
        await runner.get_messages()

        # Should only fetch once
        assert len(fetcher.fetch_calls) == 1

    @pytest.mark.asyncio
    async def test_refetches_after_turn_complete(self):
        """Knowledge IS re-fetched after on_turn_complete signals new turn."""
        view = MockConversationView()
        view.add_user_message("First question")

        chunks = [KnowledgeChunk(content=[TextContent(text="Info")])]
        fetcher = MockKnowledgeFetcher(chunks)

        runner = AgentRunner(view=view, knowledge_fetcher=fetcher)

        # First turn
        await runner.get_messages()
        await runner.on_turn_complete()

        # Second turn
        view.add_user_message("Second question")
        await runner.get_messages()

        # Should fetch twice (once per turn)
        assert len(fetcher.fetch_calls) == 2
        assert fetcher.fetch_calls[0] == "First question"
        assert fetcher.fetch_calls[1] == "Second question"

    @pytest.mark.asyncio
    async def test_invalidate_cache_forces_refetch(self):
        """invalidate_cache() forces re-fetch on next get_messages."""
        view = MockConversationView()
        view.add_user_message("Question")

        chunks = [KnowledgeChunk(content=[TextContent(text="Info")])]
        fetcher = MockKnowledgeFetcher(chunks)

        runner = AgentRunner(view=view, knowledge_fetcher=fetcher)

        await runner.get_messages()
        runner.invalidate_cache()
        await runner.get_messages()

        assert len(fetcher.fetch_calls) == 2


class TestAgentRunnerDelegation:
    """Tests that AgentRunner delegates to wrapped view."""

    @pytest.mark.asyncio
    async def test_delegates_on_text_delta(self):
        view = MockConversationView()
        runner = AgentRunner(view=view)

        await runner.on_text_delta("Hello")

        assert view.text_deltas == ["Hello"]

    @pytest.mark.asyncio
    async def test_delegates_on_content_block(self):
        view = MockConversationView()
        runner = AgentRunner(view=view)

        block = TextContent(text="test")
        await runner.on_content_block(block)

        assert view.content_blocks == [block]

    @pytest.mark.asyncio
    async def test_delegates_call_tool(self):
        view = MockConversationView()
        runner = AgentRunner(view=view)

        tool_call = ToolCall(name="test", input={})
        result = await runner.call_tool(tool_call)

        assert len(view.tool_calls) == 1
        assert result.tool_call_id == tool_call.id

    @pytest.mark.asyncio
    async def test_delegates_add_message(self):
        view = MockConversationView()
        runner = AgentRunner(view=view)

        msg = Message(role="assistant", content=[TextContent(text="Hi")])
        await runner.add_message(msg)

        assert len(view.added_messages) == 1

    @pytest.mark.asyncio
    async def test_delegates_on_turn_complete(self):
        view = MockConversationView()
        runner = AgentRunner(view=view)

        await runner.on_turn_complete()

        assert view.turn_complete_count == 1
