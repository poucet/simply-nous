# Phase 1: Foundation - Tasks

**Status:** active
**Goal:** Core types and ConversationView protocol

## Copy from Episteme

| Source | Destination | Status |
|--------|-------------|--------|
| `api/src/noema_schemas/content.py` | `src/nous/types/content.py` | done |
| `api/src/noema_schemas/message.py` | `src/nous/types/conversation.py` | done |
| `api/src/noema_schemas/tool.py` | `src/nous/types/tool.py` | done |

## Completed

- [x] **Project setup** - pyproject.toml, README.md, .claude/CLAUDE.md
- [x] **Types package** - content.py, conversation.py, tool.py, provider.py
- [x] **Storage protocols** - IConversationStore, IKnowledgeStore
- [x] **Test suite** - test_types.py with 4 passing tests

## P0 - Must Have

### T1.1: ConversationView Protocol ✓
**Priority:** P0 | **Status:** done
**Files:** `src/nous/view/protocol.py`, `src/nous/view/__init__.py`

Define the bidirectional channel between engine and client:

```python
class ConversationView(Protocol):
    """Bidirectional channel between engine and client."""

    # Read: Engine pulls state
    def get_messages(self, limit: int | None = None) -> list[Message]: ...
    def get_system_prompt(self) -> str | None: ...
    @property
    def model_id(self) -> str: ...

    # Write: Engine pushes events
    async def on_text_delta(self, text: str) -> None: ...
    async def on_content_block(self, block: ContentBlock) -> None: ...
    async def on_tool_call(self, tool_call: ToolCall) -> ToolResult: ...
    async def on_message_complete(self, message: Message) -> None: ...
```

**Design notes:**
- `on_tool_call` returns `ToolResult` — view handles approval (Discord DM, WebSocket UI, auto-approve)
- `on_text_delta` for streaming text as it arrives
- `on_content_block` for complete non-text blocks (images, audio)
- `on_message_complete` when assistant turn finishes

### T1.2: MockConversationView ✓
**Priority:** P0 | **Status:** done
**Files:** `src/nous/view/mock.py`

Reference implementation for testing:
- Stores messages in list
- Collects text deltas, content blocks
- Auto-approves tool calls (returns mock result)
- No persistence

### T1.3: View Tests ✓
**Priority:** P0 | **Status:** done
**Files:** `tests/test_view.py`

- Protocol compliance tests
- MockConversationView tests
- Tool call → result flow

## P1 - Should Have

### T1.4: Type Alignment with Episteme
**Priority:** P1
**Files:** `src/nous/types/content.py`, `src/nous/types/conversation.py`

Ensure types match Episteme's `noema_schemas`:
- `ContentBlock` union includes `ToolUseContent`, `ToolResultContent`
- `Message` has `provider`, `model` fields
- Serialization matches API contract

### T1.5: Documentation
**Priority:** P1
**Files:** All `src/nous/**/*.py`

- Docstrings for public APIs
- Example usage in module docstrings

---

## Test

```bash
uv run pytest tests/
# Import types, create Message with ContentBlocks, serialize to JSON
# MockConversationView receives callbacks correctly
```

---

## Compatibility Reference

See [COMPATIBILITY.md](../../COMPATIBILITY.md) for alignment status with Episteme and Lumina.
