# Phase 3: Engine - Tasks

**Status:** active
**Goal:** Storage-agnostic engine with ConversationView callbacks

## Summary

| Pri | ✓ | Task | Title | Link |
|-----|---|------|-------|------|
| P0 | ⬜ | T3.1 | Engine Core | [#t31](#t31-engine-core) |
| P0 | ⬜ | T3.2 | Context Builder | [#t32](#t32-context-builder) |
| P0 | ⬜ | T3.3 | Callback Integration | [#t33](#t33-callback-integration) |
| P1 | ⬜ | T3.4 | Content Processor | [#t34](#t34-content-processor) |
| P1 | ⬜ | T3.5 | Memory View | [#t35](#t35-memory-view) |
| P1 | ⬜ | T3.6 | Engine Tests | [#t36](#t36-engine-tests) |
| P2 | ⬜ | T3.7 | Demo CLI | [#t37](#t37-demo-cli) |

## Source Files (Episteme → Nous)

| Source | Destination | Status |
|--------|-------------|--------|
| `backend/llm/engine.py` | `src/nous/engine/engine.py` | todo |
| `backend/llm/context_builder.py` | `src/nous/engine/context.py` | todo |
| `backend/llm/content_processor.py` | `src/nous/engine/content.py` | todo |

## P0 - Must Have

### T3.1: Engine Core
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/engine/engine.py`, `src/nous/engine/__init__.py`

Extract Engine class from Episteme, removing storage dependencies:

```python
class Engine:
    """Storage-agnostic conversation engine."""

    def __init__(self, hub: ProviderHub):
        self.hub = hub

    async def run_turn(self, view: ConversationView) -> Message:
        """Run one conversation turn.

        1. Build context from view.get_messages()
        2. Call LLM via hub
        3. Stream events through view callbacks
        4. Return final assistant message
        """
        ...
```

**Design notes:**
- Engine reads from ConversationView, not IConversationStore
- No direct database access - all state comes from view
- Provider selection via `hub.get_for_model(view.model_id)`

### T3.2: Context Builder
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/engine/context.py`

Build LLM context from ConversationView:

```python
class ContextBuilder:
    """Builds LLM context from conversation view."""

    def build(self, view: ConversationView) -> tuple[str | None, list[Message]]:
        """Return (system_prompt, messages) ready for LLM."""
        system = view.get_system_prompt()
        messages = view.get_messages()
        # Apply context window limits
        # Handle multi-modal content
        return system, messages
```

**Design notes:**
- Respects model context window limits
- Handles message truncation if needed
- May inject system context (model capabilities, etc.)

### T3.3: Callback Integration
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/engine/engine.py`

Wire streaming events to ConversationView callbacks:

```python
async def _stream_completion(self, view: ConversationView, ...):
    async for event in client.complete(...):
        match event:
            case TextDeltaEvent(text=text):
                await view.on_token(text)
            case ToolCallEvent(tool_call=tc):
                result = await view.on_tool_call(tc)
                # Continue with tool result
            case MessageCompleteEvent(message=msg):
                await view.on_message(msg)
                return msg
```

**Design notes:**
- All events flow through view callbacks
- Tool execution is view's responsibility (approval, execution)
- Engine waits for tool result before continuing

## P1 - Should Have

### T3.4: Content Processor
**Priority:** P1 | **Status:** todo
**Files:** `src/nous/engine/content.py`

Process multi-modal content for different providers:

```python
class ContentProcessor:
    """Process content blocks for provider compatibility."""

    def prepare_for_model(
        self,
        messages: list[Message],
        capabilities: ModelCapabilities
    ) -> list[Message]:
        """Adapt content to model capabilities.

        - Convert images to descriptions for non-vision models
        - Handle audio content
        - Truncate long content
        """
        ...
```

**Design notes:**
- Uses ModelCapabilities to determine processing
- Graceful degradation for unsupported content types
- Preserves original messages (returns copies)

### T3.5: Memory View
**Priority:** P1 | **Status:** todo
**Files:** `src/nous/view/memory.py`

In-memory ConversationView for testing and simple use cases:

```python
class MemoryConversationView:
    """In-memory implementation of ConversationView."""

    def __init__(
        self,
        model_id: str,
        system_prompt: str | None = None,
    ):
        self.messages: list[Message] = []
        self._model_id = model_id
        self._system_prompt = system_prompt
        self.tokens: list[str] = []  # Captured for testing

    def add_user_message(self, content: str) -> None: ...
    def get_messages(self) -> list[Message]: ...
    async def on_token(self, token: str) -> None: ...
    # ... etc
```

**Design notes:**
- Simple list-based storage
- Captures all callbacks for test assertions
- Useful for CLI demos and testing

### T3.6: Engine Tests
**Priority:** P1 | **Status:** todo
**Files:** `tests/test_engine.py`

Test engine with mocked providers:

- Context building from view
- Streaming callback flow
- Tool call → result → continue flow
- Multi-turn conversation
- Error handling (provider errors, tool failures)

## P2 - Nice to Have

### T3.7: Demo CLI
**Priority:** P2 | **Status:** todo
**Files:** `src/nous/demo.py`

Update demo to use Engine + MemoryConversationView:

```bash
uv run python -m nous.demo
# Interactive chat using Engine with MemoryView
```

---

## Test

```bash
uv run pytest tests/test_engine.py
# Engine runs turn, MemoryConversationView captures callbacks
```

---

## Dependencies

Phase 1 types used:
- `Message`, `ContentBlock` for conversation
- `ToolCall`, `ToolResult` for tool flow
- `ConversationView` protocol

Phase 2 LLM layer used:
- `ProviderHub` for provider access
- `ModelClient` for completion
- `StreamEvent` for streaming
- `ModelCapabilities` for content processing
