# Phase 1: Foundation - Tasks

**Status:** active
**Goal:** Core types and ConversationView protocol

## Completed

- [x] **Project setup** - pyproject.toml, README.md, .claude/CLAUDE.md
- [x] **Types package** - content.py, conversation.py, tool.py, provider.py
- [x] **Storage protocols** - IConversationStore, IKnowledgeStore
- [x] **Test suite** - test_types.py with 4 passing tests

## P0 - Must Have

### T1.1: ConversationView Protocol
**Priority:** P0
**Files:** `src/nous/view/protocol.py`, `src/nous/view/__init__.py`

Define the bidirectional channel between engine and client:

```python
class ConversationView(Protocol):
    # Read: Engine pulls
    def get_messages(self, limit: int | None = None) -> list[Message]: ...
    def get_system_prompt(self) -> str | None: ...
    @property
    def model_id(self) -> str: ...
    @property
    def is_private(self) -> bool: ...

    # Write: Engine pushes
    async def on_token(self, token: str) -> None: ...
    async def on_tool_call(self, tool_call: ToolCall) -> ToolResult: ...
    async def on_message(self, message: Message) -> None: ...
    async def on_status(self, status: str) -> None: ...
```

### T1.2: MemoryConversationView
**Priority:** P0
**Files:** `src/nous/view/memory.py`

Reference implementation for testing:
- Stores messages in list
- Collects tokens, tool calls
- No persistence

### T1.3: View Tests
**Priority:** P0
**Files:** `tests/test_view.py`

- Protocol compliance tests
- MemoryConversationView tests

## P1 - Should Have

### T1.4: Type Refinements
**Priority:** P1
**Files:** `src/nous/types/content.py`, `src/nous/types/conversation.py`

Align types with Noema UCM:
- Add `is_private` to Message
- Add origin metadata (user/assistant/system/import)

### T1.5: Documentation
**Priority:** P1
**Files:** All `src/nous/**/*.py`

- Docstrings for public APIs
- Example usage in module docstrings

---

## Moved to Phase 2

The following tasks moved to Phase 2 (LLM Layer):

- ProviderHub extraction
- Claude/OpenAI/Gemini/Ollama providers
- Provider tests

This keeps Phase 1 focused on the core ConversationView abstraction.
