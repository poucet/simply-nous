"""Tests for LLM provider layer."""

import pytest
from typing import AsyncIterator

from nous.llm import (
    LLMProvider,
    ModelClient,
    ProviderHub,
    CachingProvider,
    StreamEvent,
    TextDeltaEvent,
    ToolCallEvent,
    MessageCompleteEvent,
)
from nous.llm.providers import OllamaProvider
from nous.types import (
    Message,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    ToolDefinition,
    ToolCall,
    Provider,
)


# =============================================================================
# Mock Provider for Testing
# =============================================================================


class MockModelClient:
    """Mock model client for testing."""

    def __init__(self, provider_type: Provider, model_id: str):
        self._provider = provider_type
        self._model_id = model_id

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model_id(self) -> str:
        return self._model_id

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message:
        if stream:
            return self._stream(messages)
        return self._complete(messages)

    def _complete(self, messages: list[Message]) -> Message:
        last_msg = messages[-1].content[0].text if messages else "empty"
        return Message(
            role="assistant",
            content=[TextContent(text=f"Mock response to: {last_msg}")],
            provider=self._provider.value,
            model=self._model_id,
        )

    async def _stream(self, messages: list[Message]) -> AsyncIterator[StreamEvent]:
        last_msg = messages[-1].content[0].text if messages else "empty"
        response_text = f"Mock response to: {last_msg}"

        # Yield text in chunks
        for word in response_text.split():
            yield TextDeltaEvent(text=word + " ")

        # Yield final message
        yield MessageCompleteEvent(
            message=Message(
                role="assistant",
                content=[TextContent(text=response_text)],
                provider=self._provider.value,
                model=self._model_id,
            )
        )


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, provider_type: Provider = Provider.ANTHROPIC):
        self._provider = provider_type
        self._models = ["mock-model-1", "mock-model-2", "mock-model-3"]
        self.list_models_called = 0

    @property
    def provider(self) -> Provider:
        return self._provider

    async def list_models(self) -> list[str]:
        self.list_models_called += 1
        return self._models

    def model(self, model_id: str) -> MockModelClient:
        return MockModelClient(self._provider, model_id)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Verify providers satisfy protocol contracts."""

    def test_mock_provider_is_llm_provider(self):
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    def test_mock_client_is_model_client(self):
        provider = MockProvider()
        client = provider.model("mock-model-1")
        assert isinstance(client, ModelClient)

    def test_ollama_provider_is_llm_provider(self):
        provider = OllamaProvider()
        assert isinstance(provider, LLMProvider)

    def test_ollama_client_is_model_client(self):
        provider = OllamaProvider()
        client = provider.model("llama3.2")
        assert isinstance(client, ModelClient)

    def test_provider_has_required_attributes(self):
        provider = MockProvider()
        assert hasattr(provider, "provider")
        assert hasattr(provider, "list_models")
        assert hasattr(provider, "model")

    def test_client_has_required_attributes(self):
        provider = MockProvider()
        client = provider.model("test-model")
        assert hasattr(client, "provider")
        assert hasattr(client, "model_id")
        assert hasattr(client, "complete")


# =============================================================================
# ModelClient Tests
# =============================================================================


class TestModelClient:
    """Tests for ModelClient implementations."""

    @pytest.mark.asyncio
    async def test_complete_returns_message(self):
        provider = MockProvider()
        client = provider.model("mock-model-1")
        messages = [Message(role="user", content=[TextContent(text="Hello")])]

        response = await client.complete(messages)

        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert "Hello" in response.content[0].text

    @pytest.mark.asyncio
    async def test_complete_stream_yields_events(self):
        provider = MockProvider()
        client = provider.model("mock-model-1")
        messages = [Message(role="user", content=[TextContent(text="Hello")])]

        events = []
        async for event in await client.complete(messages, stream=True):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, TextDeltaEvent) for e in events)
        assert isinstance(events[-1], MessageCompleteEvent)

    def test_client_has_correct_model_id(self):
        provider = MockProvider()
        client = provider.model("specific-model")
        assert client.model_id == "specific-model"

    def test_client_has_correct_provider(self):
        provider = MockProvider(Provider.OLLAMA)
        client = provider.model("test")
        assert client.provider == Provider.OLLAMA


# =============================================================================
# ProviderHub Tests
# =============================================================================


class TestProviderHub:
    """Tests for ProviderHub registry."""

    def test_empty_hub(self):
        hub = ProviderHub()
        assert hub.providers == []

    def test_register_provider(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))

        assert hub.is_registered(Provider.ANTHROPIC)
        assert Provider.ANTHROPIC in hub.providers

    def test_get_unregistered_raises(self):
        hub = ProviderHub()
        with pytest.raises(KeyError):
            hub.get(Provider.ANTHROPIC)

    def test_get_returns_provider(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))

        provider = hub.get(Provider.ANTHROPIC)

        assert isinstance(provider, LLMProvider)
        assert provider.provider == Provider.ANTHROPIC

    def test_get_returns_same_instance(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))

        p1 = hub.get(Provider.ANTHROPIC)
        p2 = hub.get(Provider.ANTHROPIC)

        assert p1 is p2

    def test_register_replaces_existing(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))
        p1 = hub.get(Provider.ANTHROPIC)

        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))
        p2 = hub.get(Provider.ANTHROPIC)

        assert p1 is not p2

    @pytest.mark.asyncio
    async def test_get_for_model(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))

        provider = await hub.get_for_model("mock-model-1")

        assert provider.provider == Provider.ANTHROPIC

    @pytest.mark.asyncio
    async def test_get_for_model_unknown_raises(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))

        with pytest.raises(KeyError):
            await hub.get_for_model("unknown-model")

    @pytest.mark.asyncio
    async def test_get_for_model_caches_results(self):
        hub = ProviderHub()
        mock = MockProvider(Provider.ANTHROPIC)
        hub.register(Provider.ANTHROPIC, lambda: mock)

        await hub.get_for_model("mock-model-1")
        await hub.get_for_model("mock-model-1")

        # list_models should only be called once due to caching
        assert mock.list_models_called == 1

    @pytest.mark.asyncio
    async def test_client_for_model(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))

        client = await hub.client_for_model("mock-model-2")

        assert isinstance(client, ModelClient)
        assert client.model_id == "mock-model-2"

    def test_multiple_providers(self):
        hub = ProviderHub()
        hub.register(Provider.ANTHROPIC, lambda: MockProvider(Provider.ANTHROPIC))
        hub.register(Provider.OLLAMA, lambda: MockProvider(Provider.OLLAMA))

        assert len(hub.providers) == 2
        assert hub.get(Provider.ANTHROPIC).provider == Provider.ANTHROPIC
        assert hub.get(Provider.OLLAMA).provider == Provider.OLLAMA


# =============================================================================
# CachingProvider Tests
# =============================================================================


class TestCachingProvider:
    """Tests for CachingProvider wrapper."""

    def test_wraps_provider(self):
        inner = MockProvider(Provider.ANTHROPIC)
        cached = CachingProvider(inner)

        assert cached.provider == Provider.ANTHROPIC

    @pytest.mark.asyncio
    async def test_caches_list_models(self):
        inner = MockProvider(Provider.ANTHROPIC)
        cached = CachingProvider(inner)

        models1 = await cached.list_models()
        models2 = await cached.list_models()

        assert models1 == models2
        assert inner.list_models_called == 1

    def test_delegates_model(self):
        inner = MockProvider(Provider.ANTHROPIC)
        cached = CachingProvider(inner)

        client = cached.model("test-model")

        assert client.model_id == "test-model"

    def test_clear_cache(self):
        inner = MockProvider(Provider.ANTHROPIC)
        cached = CachingProvider(inner)

        # Populate cache
        import asyncio
        asyncio.get_event_loop().run_until_complete(cached.list_models())

        # Clear and refetch
        cached.clear_cache()
        asyncio.get_event_loop().run_until_complete(cached.list_models())

        assert inner.list_models_called == 2


# =============================================================================
# Message Conversion Tests
# =============================================================================


class TestMessageConversion:
    """Tests for message format handling."""

    def test_text_content_message(self):
        msg = Message(role="user", content=[TextContent(text="Hello")])
        assert msg.role == "user"
        assert msg.content[0].text == "Hello"

    def test_tool_use_content_message(self):
        msg = Message(
            role="assistant",
            content=[
                ToolUseContent(
                    id="call_123",
                    name="search",
                    input={"query": "test"},
                )
            ],
        )
        assert msg.content[0].name == "search"
        assert msg.content[0].input == {"query": "test"}

    def test_tool_result_content_message(self):
        msg = Message(
            role="user",
            content=[
                ToolResultContent(
                    tool_use_id="call_123",
                    content=[TextContent(text="Result: success")],
                    is_error=False,
                )
            ],
        )
        assert msg.content[0].tool_use_id == "call_123"
        assert not msg.content[0].is_error

    def test_mixed_content_message(self):
        msg = Message(
            role="assistant",
            content=[
                TextContent(text="I'll search for that."),
                ToolUseContent(
                    id="call_456",
                    name="search",
                    input={"query": "python"},
                ),
            ],
        )
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ToolUseContent)


# =============================================================================
# Stream Event Tests
# =============================================================================


class TestStreamEvents:
    """Tests for streaming event types."""

    def test_text_delta_event(self):
        event = TextDeltaEvent(text="Hello")
        assert event.text == "Hello"

    def test_tool_call_event(self):
        tool_call = ToolCall(id="call_1", name="search", input={"q": "test"})
        event = ToolCallEvent(tool_call=tool_call)
        assert event.tool_call.name == "search"

    def test_message_complete_event(self):
        msg = Message(role="assistant", content=[TextContent(text="Done")])
        event = MessageCompleteEvent(message=msg)
        assert event.message.role == "assistant"
