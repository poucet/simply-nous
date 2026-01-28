# Phase 2: LLM Layer - Tasks

**Status:** active
**Goal:** Extract ProviderHub from Episteme

## Copy from Episteme

| Source | Destination | Status |
|--------|-------------|--------|
| `backend/src/backend/llm/providers/` | `src/nous/llm/providers/` | todo |
| `backend/src/backend/llm/hub.py` | `src/nous/llm/hub.py` | todo |

## P0 - Must Have

### T2.1: Provider Protocol
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/llm/protocol.py`, `src/nous/llm/__init__.py`

Define the LLM provider interface:

```python
class LLMProvider(Protocol):
    """Abstract LLM provider interface."""

    @property
    def provider(self) -> Provider: ...

    async def complete(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[StreamEvent] | Message: ...
```

**Design notes:**
- Streaming via AsyncIterator of events (text delta, tool call, complete)
- Non-streaming returns complete Message
- Provider-specific formatting handled internally

### T2.2: Stream Events
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/llm/events.py`

Define streaming event types:

```python
@dataclass
class TextDeltaEvent:
    text: str

@dataclass
class ToolCallEvent:
    tool_call: ToolCall

@dataclass
class MessageCompleteEvent:
    message: Message

StreamEvent = TextDeltaEvent | ToolCallEvent | MessageCompleteEvent
```

### T2.3: Anthropic Provider
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/llm/providers/anthropic.py`

Implement AnthropicProvider:
- Uses `anthropic` SDK
- Converts nous Message → Anthropic format
- Converts Anthropic response → nous Message
- Streaming support via SDK's stream helper

### T2.4: ProviderHub
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/llm/hub.py`

Registry and factory for providers:

```python
class ProviderHub:
    """Registry of LLM providers."""

    def register(self, provider: Provider, factory: Callable[[], LLMProvider]) -> None: ...
    def get(self, provider: Provider) -> LLMProvider: ...
    def get_for_model(self, model_id: str) -> LLMProvider: ...
```

**Design notes:**
- Lazy instantiation via factory functions
- Model → Provider mapping for convenience
- Default hub with common providers pre-registered

## P1 - Should Have

### T2.5: OpenAI Provider
**Priority:** P1 | **Status:** todo
**Files:** `src/nous/llm/providers/openai.py`

Implement OpenAIProvider:
- Uses `openai` SDK
- Message format conversion
- Streaming support

### T2.6: Provider Tests
**Priority:** P1 | **Status:** todo
**Files:** `tests/test_llm.py`

- Protocol compliance tests
- Message conversion tests (mock SDK responses)
- ProviderHub registry tests

## P2 - Nice to Have

### T2.7: Provider Configuration
**Priority:** P2 | **Status:** todo
**Files:** `src/nous/llm/config.py`

- API key management (env vars, config file)
- Model defaults per provider
- Rate limiting configuration

---

## Test

```bash
uv run pytest tests/test_llm.py
# Register providers, convert messages, verify streaming events
```

---

## Dependencies

Phase 1 types used:
- `Message`, `ContentBlock` for request/response
- `ToolCall`, `ToolResult`, `ToolDefinition` for tool use
- `Provider` enum for provider identification
