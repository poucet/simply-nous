# Phase 2: LLM Layer - Tasks

**Status:** active
**Goal:** Extract ProviderHub from Episteme

## Summary

| Pri | ✓ | Task | Title | Link |
|-----|---|------|-------|------|
| P0 | ✅ | T2.1 | Provider Protocol | [#t21](#t21-provider-protocol) |
| P0 | ✅ | T2.2 | Stream Events | [#t22](#t22-stream-events) |
| P0 | ✅ | T2.3 | Model Capabilities | [#t23](#t23-model-capabilities) |
| P0 | ✅ | T2.4 | Anthropic Provider | [#t24](#t24-anthropic-provider) |
| P0 | ✅ | T2.5 | ProviderHub | [#t25](#t25-providerhub) |
| P1 | ✅ | T2.6 | Ollama Provider | [#t26](#t26-ollama-provider) |
| P1 | ✅ | T2.7 | Gemini Provider | [#t27](#t27-gemini-provider) |
| P1 | ✅ | T2.8 | Provider Tests | [#t28](#t28-provider-tests) |
| P2 | ✅ | T2.9 | OpenAI Provider | [#t29](#t29-openai-provider) |
| P2 | ⬜ | T2.10 | Mistral Provider | [#t210](#t210-mistral-provider) |
| P2 | ⬜ | T2.11 | Provider Configuration | [#t211](#t211-provider-configuration) |

## Copy from Episteme

| Source | Destination | Status |
|--------|-------------|--------|
| `backend/src/backend/llm/providers/` | `src/nous/llm/providers/` | todo |
| `backend/src/backend/llm/hub.py` | `src/nous/llm/hub.py` | todo |

## P0 - Must Have

### T2.1: Provider Protocol
**Priority:** P0 | **Status:** done
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
**Priority:** P0 | **Status:** done
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

### T2.3: Model Capabilities
**Priority:** P0 | **Status:** done
**Files:** `src/nous/llm/capabilities.py`

Define model capability metadata:

```python
@dataclass
class ModelCapabilities:
    """What a model can do."""
    vision: bool = False           # Can process images
    audio_input: bool = False      # Can process audio
    audio_output: bool = False     # Can generate audio
    image_output: bool = False     # Can generate images
    tools: bool = False            # Supports tool use
    streaming: bool = True         # Supports streaming
    max_tokens: int | None = None  # Context window size

class ModelRegistry:
    """Registry of known models and their capabilities."""
    def get(self, model_id: str) -> ModelCapabilities: ...
    def supports_vision(self, model_id: str) -> bool: ...
    def filter(self, **requirements) -> list[str]: ...
```

**Design notes:**
- Query capabilities before routing: `if registry.supports_vision(model_id)`
- Filter models: `registry.filter(vision=True, tools=True)`
- Preprocessing hook: convert images to text descriptions for non-vision models
- Output multimodal: handle image generation responses (DALL-E, Gemini)

### T2.4: Anthropic Provider
**Priority:** P0 | **Status:** done
**Files:** `src/nous/llm/providers/anthropic.py`

Implement AnthropicProvider:
- Uses `anthropic` SDK
- Converts nous Message → Anthropic format
- Converts Anthropic response → nous Message
- Streaming support via SDK's stream helper

### T2.5: ProviderHub
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

### T2.6: Ollama Provider
**Priority:** P1 | **Status:** done
**Files:** `src/nous/llm/providers/ollama.py`

Implement OllamaProvider:
- Uses Ollama HTTP API (no SDK needed)
- Local model support (llama, mistral, etc.)
- Message format conversion
- Streaming support

### T2.7: Gemini Provider
**Priority:** P1 | **Status:** done
**Files:** `src/nous/llm/providers/gemini.py`

Implement GeminiProvider:
- Uses `google-genai` SDK (new package)
- Message format conversion
- Streaming support

### T2.8: Provider Tests
**Priority:** P1 | **Status:** done
**Files:** `tests/test_llm.py`

- Protocol compliance tests (6 tests)
- ModelClient tests (4 tests)
- ProviderHub registry tests (11 tests)
- CachingProvider tests (4 tests)
- Message conversion tests (4 tests)
- Stream event tests (3 tests)

## P2 - Nice to Have

### T2.9: OpenAI Provider
**Priority:** P2 | **Status:** done
**Files:** `src/nous/llm/providers/openai.py`

Implement OpenAIProvider:
- Uses `openai` SDK
- Message format conversion
- Streaming support

### T2.10: Mistral Provider
**Priority:** P2 | **Status:** todo
**Files:** `src/nous/llm/providers/mistral.py`

Implement MistralProvider:
- Uses `mistralai` SDK
- Message format conversion
- Streaming support

### T2.11: Provider Configuration
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
