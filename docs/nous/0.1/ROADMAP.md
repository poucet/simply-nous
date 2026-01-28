# Nous 0.1 Roadmap

> The capacity to think - shared AI core for the Simply ecosystem.

## The Trilogy

| Project | Greek | Role |
|---------|-------|------|
| **Noema** | νόημα (thought-object) | Data structures, content model (Rust) |
| **Nous** | νοῦς (intellect) | Thinking capacity, shared core (Python) |
| **Episteme** | ἐπιστήμη (knowledge) | Platform, resulting knowledge (Python) |
| **Lumina** | (light) | Discord interface (Python) |

---

## Phase Overview

| Phase | Name | Status | Goal |
|-------|------|--------|------|
| 1 | Foundation | **Active** | Core types and ConversationView protocol |
| 2 | LLM Layer | Pending | Extract ProviderHub from Episteme |
| 3 | Engine | Pending | Storage-agnostic engine with callbacks |
| 4 | MCP | Pending | Tool execution with approval workflows |
| 5 | Integration | Pending | Episteme + Lumina adopt nous |

---

## Phase 1: Foundation (Active)

**Version:** 0.1.0
**Goal:** Core types and ConversationView protocol

### Tasks

- [x] Project structure and pyproject.toml
- [x] Core types: ContentBlock, Message, Conversation
- [x] Tool types: ToolCall, ToolResult
- [x] Storage protocols: IConversationStore, IKnowledgeStore
- [ ] **ConversationView protocol** - bidirectional channel
- [ ] Unit tests for all types

### ConversationView Protocol

The key abstraction enabling storage-agnostic engines:

```python
class ConversationView(Protocol):
    """Bidirectional channel between engine and client."""

    # Read: Engine pulls history
    def get_messages(self, limit: int | None = None) -> list[Message]: ...
    def get_system_prompt(self) -> str | None: ...
    @property
    def model_id(self) -> str: ...
    @property
    def is_private(self) -> bool: ...

    # Write: Engine pushes events back
    async def on_token(self, token: str) -> None: ...
    async def on_tool_call(self, tool_call: ToolCall) -> ToolResult: ...
    async def on_message(self, message: Message) -> None: ...
    async def on_status(self, status: str) -> None: ...
```

---

## Phase 2: LLM Layer

**Version:** 0.2.0
**Goal:** Extract ProviderHub from Episteme

### Source Files (Episteme → Nous)

| Episteme Source | Nous Destination |
|-----------------|------------------|
| `backend/llm/providers/hub.py` | `src/nous/llm/hub.py` |
| `backend/llm/providers/base.py` | `src/nous/llm/providers/base.py` |
| `backend/llm/providers/claude.py` | `src/nous/llm/providers/claude.py` |
| `backend/llm/providers/openai.py` | `src/nous/llm/providers/openai.py` |
| `backend/llm/providers/gemini.py` | `src/nous/llm/providers/gemini.py` |
| `backend/llm/providers/ollama.py` | `src/nous/llm/providers/ollama.py` |

### Tasks

- [ ] Extract ProviderHub factory
- [ ] Extract base provider interface
- [ ] Extract Claude provider (streaming)
- [ ] Extract OpenAI provider (streaming)
- [ ] Extract Gemini provider (streaming)
- [ ] Extract Ollama provider (streaming)
- [ ] Provider tests with mocked APIs
- [ ] **Demo**: CLI that calls Claude through nous

---

## Phase 3: Engine

**Version:** 0.3.0
**Goal:** Storage-agnostic engine with ConversationView callbacks

### Source Files (Episteme → Nous)

| Episteme Source | Nous Destination |
|-----------------|------------------|
| `backend/llm/engine.py` | `src/nous/engine/engine.py` |
| `backend/llm/context_builder.py` | `src/nous/engine/context.py` |
| `backend/llm/content_processor.py` | `src/nous/engine/content.py` |

### Key Changes

1. **Remove storage dependencies** - Engine reads from ConversationView, not IConversationStore
2. **Add callbacks** - Engine pushes tokens, tool calls, messages through view
3. **Streaming support** - All providers stream through `on_token()`

### Tasks

- [ ] Extract Engine class (remove storage deps)
- [ ] Extract ContextBuilder
- [ ] Extract ContentProcessor
- [ ] Implement callback-based streaming
- [ ] MemoryConversationView for testing
- [ ] Engine tests with mocked providers
- [ ] **Demo**: Engine with MemoryView

---

## Phase 4: MCP

**Version:** 0.4.0
**Goal:** Tool execution with approval workflows

### Source Files (Episteme → Nous)

| Episteme Source | Nous Destination |
|-----------------|------------------|
| `backend/mcp/client.py` | `src/nous/mcp/client.py` |
| `backend/mcp/executor.py` | `src/nous/mcp/executor.py` |
| `backend/mcp/discovery.py` | `src/nous/mcp/discovery.py` |

### Key Design

Tool calls flow through ConversationView:
1. Engine calls `view.on_tool_call(tool_call)`
2. View handles approval (UI, auto-approve, etc.)
3. View executes tool and returns `ToolResult`
4. Engine continues with result

### Tasks

- [ ] Extract MCPClient
- [ ] Extract ToolExecutor
- [ ] Integrate with ConversationView.on_tool_call()
- [ ] Approval workflow in view (client responsibility)
- [ ] MCP tests with mock server
- [ ] **Demo**: Engine with tool calling

---

## Phase 5: Integration

**Version:** 0.5.0
**Goal:** Episteme and Lumina adopt nous

### Episteme Integration

| Step | Description |
|------|-------------|
| 5.1 | Add `nous` dependency to pyproject.toml |
| 5.2 | Replace `noema_schemas` imports with `nous.types` |
| 5.3 | Create `StoredConversationView` backed by PostgreSQL |
| 5.4 | Replace `backend/llm/` with `nous.llm` + `nous.engine` |
| 5.5 | Update WebSocket handler to use engine callbacks |
| 5.6 | Delete migrated code from Episteme |

### Lumina Integration

| Step | Description |
|------|-------------|
| 5.1 | Add `nous` dependency |
| 5.2 | Replace LangChain message types with `nous.types` |
| 5.3 | Create `DiscordConversationView` backed by Discord API |
| 5.4 | Replace `agent/models.py` with `nous.llm` |
| 5.5 | Update Agent to use nous Engine |

### Tasks

- [ ] Episteme: Create StoredConversationView
- [ ] Episteme: Migrate to nous imports
- [ ] Episteme: Delete backend/llm/ after migration
- [ ] Lumina: Create DiscordConversationView
- [ ] Lumina: Migrate to nous imports
- [ ] Lumina: Drop LangChain dependency
- [ ] **Demo**: Both apps running on shared core

---

## Module Structure (Target)

```
nous/
├── types/
│   ├── __init__.py
│   ├── content.py      # TextContent, ImageContent, AudioContent
│   ├── conversation.py # Message, Conversation
│   ├── tool.py         # ToolCall, ToolResult
│   └── provider.py     # ProviderConfig, ModelInfo
│
├── view/
│   ├── __init__.py
│   └── protocol.py     # ConversationView protocol
│
├── engine/
│   ├── __init__.py
│   ├── engine.py       # Main Engine class
│   ├── context.py      # ContextBuilder
│   └── content.py      # ContentProcessor
│
├── llm/
│   ├── __init__.py
│   ├── hub.py          # ProviderHub
│   └── providers/
│       ├── __init__.py
│       ├── base.py     # BaseLLMProvider
│       ├── claude.py
│       ├── openai.py
│       ├── gemini.py
│       └── ollama.py
│
├── mcp/
│   ├── __init__.py
│   ├── client.py       # MCPClient
│   ├── executor.py     # ToolExecutor
│   └── discovery.py    # Tool discovery
│
└── storage/
    ├── __init__.py
    └── protocols.py    # IConversationStore, IKnowledgeStore (for reference)
```

---

## Key Design Principles

1. **Storage-agnostic** - Engine reads ConversationView, doesn't know about storage
2. **Callback-based** - Events (tokens, tool calls, messages) flow through view
3. **Client responsibility** - Persistence, approval UI, display are client concerns
4. **Protocol-based** - Use `typing.Protocol` for interfaces
5. **Async everywhere** - All I/O operations use async/await

---

## References

- [ECOSYSTEM.md](../design/ECOSYSTEM.md) - Architecture overview
- [Noema STORAGE.md](../../noema/docs/STORAGE.md) - Reference UCM implementation
- [Lumina ARCHITECTURE.md](../../lumina/docs/ARCHITECTURE.md) - Current Lumina design
