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

| Phase | ✓ | Name | Goal |
|-------|---|------|------|
| 1 | ✅ | Foundation | Core types and ConversationView protocol |
| 2 | ✅ | LLM Layer | Extract ProviderHub from Episteme |
| 3 | ✅ | Engine | Storage-agnostic engine with callbacks |
| 4 | ⬜ | MCP | Tool execution with approval workflows |
| 5 | ⬜ | Episteme Integration | Episteme imports nous, deletes duplicated code |
| 6 | ⬜ | Lumina Integration | Lumina drops LangChain, adopts nous |

---

## Phase 1: Foundation (Complete)

**Version:** 0.1.0
**Goal:** Core types and ConversationView protocol

### Tasks

- [x] Project structure and pyproject.toml
- [x] Core types: ContentBlock, Message, Conversation
- [x] Tool types: ToolCall, ToolResult, ToolDefinition
- [x] ConversationView protocol
- [x] MockConversationView for testing
- [x] Unit tests for all types (21 tests)

### Test

```bash
uv run pytest tests/test_types.py
# Import types, create Message with ContentBlocks, serialize to JSON
```

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

## Phase 2: LLM Layer (Complete)

**Version:** 0.2.0
**Goal:** Extract ProviderHub from Episteme

### Architecture

Model-agnostic provider pattern:
- `LLMProvider` - connection/auth handling, `provider.model(id)` returns client
- `ModelClient` - model-specific client with `complete()` method
- `ProviderHub` - registry with model→provider mapping
- `CachingProvider` - shim for caching `list_models()` results

### Tasks

- [x] LLMProvider/ModelClient protocols
- [x] Stream events (TextDelta, ToolCall, MessageComplete)
- [x] ModelCapabilities registry
- [x] AnthropicProvider (streaming, tools)
- [x] ProviderHub with parallel model lookup
- [x] OllamaProvider (HTTP API, no SDK)
- [x] GeminiProvider (google-genai SDK)
- [x] Provider tests (32 tests)
- [x] OpenAI provider
- [x] Mistral provider + OpenRouter provider
- [x] Provider configuration

### Test

```bash
uv run pytest tests/test_llm.py
# 32 tests: protocol compliance, hub registry, caching, message conversion
```

---

## Phase 3: Engine (Complete)

**Version:** 0.3.0
**Goal:** Storage-agnostic engine with ConversationView callbacks

### Source Files (Episteme → Nous)

| Episteme Source | Nous Destination |
|-----------------|------------------|
| `backend/llm/engine.py` | `src/nous/engine/engine.py` |
| `backend/llm/context_builder.py` | `src/nous/engine/context.py` |
| `backend/llm/content_processor.py` | `src/nous/engine/content.py` |

### Delivered

- [x] Engine class with tool call loop
- [x] ContextBuilder protocol (pluggable)
- [x] ContentProcessor for model capabilities
- [x] Callback-based streaming
- [x] MemoryConversationView for testing
- [x] RAG via on_knowledge_needed() callback
- [x] Engine tests (mocked providers)
- [x] Demo CLI with Ollama

### Test

```bash
uv run pytest tests/test_engine.py
# MockConversationView receives callbacks, engine completes turn without storage
```

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

### Test

```bash
uv run pytest tests/test_mcp.py
# Engine calls tool, MockConversationView.on_tool_call returns result, engine continues
```

---

## Phase 5: Episteme Integration

**Version:** 0.5.0
**Goal:** Episteme imports nous, deletes duplicated code

### Delete from Episteme

| Path | Reason |
|------|--------|
| `api/src/noema_schemas/` | Use `nous.types` |
| `backend/src/backend/llm/providers/` | Use `nous.llm` |
| `backend/src/backend/llm/engine.py` | Use `nous.engine` |

### Create in Episteme

- `WebConversationView` implementing `nous.ConversationView`
  - `get_messages()` → SQLite query
  - `on_text_delta()` → WebSocket event
  - `on_tool_call()` → approval UI → return result

### Tasks

- [ ] Add `nous` dependency to pyproject.toml
- [ ] Create `WebConversationView`
- [ ] Replace `noema_schemas` imports with `nous.types`
- [ ] Replace `backend/llm/` with `nous.llm` + `nous.engine`
- [ ] Update WebSocket handler to use engine callbacks
- [ ] Delete migrated code

### Test

```bash
# Episteme web app works end-to-end with nous backend
cd ~/projects/simply/episteme && uv run pytest
```

---

## Phase 6: Lumina Integration

**Version:** 0.6.0
**Goal:** Lumina drops LangChain, adopts nous

### Delete from Lumina

| Path | Reason |
|------|--------|
| `agent/models.py` | LangChain model creation → use `nous.llm` |
| `agent/cortices/` | LangChain agent executor → use `nous.engine` |
| `services/logger/logger_chat_model.py` | LangChain wrapper → use callback |

### Create in Lumina

- `DiscordConversationView` implementing `nous.ConversationView`
  - `get_messages()` → Discord channel history API
  - `on_text_delta()` → buffer + channel.send()
  - `on_tool_call()` → `require_owner_confirmation()` pattern → return result
- Discord logging via `on_text_delta` callback

### Keep in Lumina

| Path | Reason |
|------|--------|
| `mcp_protocol/` | MCP servers, decorators - wire to nous executor |
| `require_owner_confirmation()` | Used in `on_tool_call` implementation |

### Tasks

- [ ] Add `nous` dependency
- [ ] Create `DiscordConversationView`
- [ ] Replace LangChain message types with `nous.types`
- [ ] Replace `agent/models.py` with `nous.llm`
- [ ] Update Agent to use nous Engine
- [ ] Drop LangChain dependency

### Test

```bash
# Lumina responds in Discord channel using nous engine
cd ~/projects/simply/lumina && uv run pytest
```

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
