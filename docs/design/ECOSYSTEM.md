# Simply Ecosystem - Architecture Overview

> The shared foundation for Noema, Episteme, and Lumina

## Overview

The Simply ecosystem consists of four interconnected projects with distinct roles:

| Project | Greek | Language | Role |
|---------|-------|----------|------|
| **Noema** | νόημα (thought-object) | Rust | Desktop client with UCM storage |
| **Nous** | νοῦς (intellect) | Python | Shared AI core library |
| **Episteme** | ἐπιστήμη (knowledge) | Python | Knowledge platform (web) |
| **Lumina** | (light) | Python | Discord bot with RAG |

---

## Target Architecture

```
                         ┌─────────────────────────────────┐
                         │            nous                 │
                         │  Types | Engine | MCP           │
                         │  ConversationView protocol      │
                         └─────────────────────────────────┘
                                      ▲
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
              ┌─────┴─────┐    ┌──────┴──────┐   ┌──────┴──────┐
              │  Lumina   │    │  Episteme   │   │   Noema     │
              │  Discord  │    │   FastAPI   │   │   Tauri     │
              │           │    │  PostgreSQL │   │   SQLite    │
              │  Reader   │    │   Store     │   │   Store     │
              │ (Discord) │    │  (Full)     │   │  (UCM)      │
              └───────────┘    └─────────────┘   └─────────────┘
```

---

## Core Engine Architecture

### Key Insight: Storage-Agnostic Engine

**The LLM engine should NOT know about storage.**

The engine operates on a **ConversationView** abstraction - a bidirectional channel between engine and client. Whether that view is backed by:
- Discord API (read-only, ephemeral)
- SQLite/PostgreSQL (persistent, owned)
- Memory (testing, throwaway)

...is **transparent to the engine**. Storage is the client's concern.

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│   Episteme (FastAPI) | Lumina (Discord) | Noema (Tauri)     │
│                                                             │
│   Responsibility: Create ConversationView, inject into      │
│   engine, handle persistence after engine returns           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ injects ConversationView
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Engine Layer                          │
│                         (nous)                               │
│                                                             │
│   ConversationView ─────► Engine ─────► Events              │
│   (read history)         (LLM call)    (via callbacks)      │
│                                                             │
│   Engine reads from view, pushes events back through it.    │
│   Does NOT write to storage. Does NOT know storage exists.  │
└─────────────────────────────────────────────────────────────┘
```

### ConversationView Interface

The view is **bidirectional** - the engine reads history AND pushes events back during agentic execution.

```python
class ConversationView(Protocol):
    """Bidirectional channel between engine and client."""

    # ===== Read: Engine pulls history =====

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get message history (most recent last)."""
        ...

    def get_system_prompt(self) -> str | None:
        """Get system prompt if any."""
        ...

    @property
    def model_id(self) -> str:
        """Current model for this conversation."""
        ...

    @property
    def is_private(self) -> bool:
        """Whether content should stay local-only."""
        ...

    # ===== Write: Engine pushes events back =====

    async def on_token(self, token: str) -> None:
        """Streaming token from LLM."""
        ...

    async def on_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Tool needs execution - client handles approval + execution."""
        ...

    async def on_message(self, message: Message) -> None:
        """Complete message (assistant response, tool result, etc.)."""
        ...

    async def on_status(self, status: str) -> None:
        """Status update (thinking, searching, etc.)."""
        ...
```

### Agentic Flow Example

```
Engine                              ConversationView                    Client
  │                                       │                               │
  │── get_messages() ────────────────────►│                               │
  │◄─────────────────── [history] ────────│                               │
  │                                       │                               │
  │── on_token("Let") ───────────────────►│── stream to UI ──────────────►│
  │── on_token(" me") ───────────────────►│── stream to UI ──────────────►│
  │── on_token(" check") ────────────────►│── stream to UI ──────────────►│
  │                                       │                               │
  │── on_tool_call(search_web) ──────────►│── show approval UI ──────────►│
  │                                       │◄─────── user approves ────────│
  │                                       │── execute tool ──────────────►│
  │◄─────────────────── ToolResult ───────│◄─────── result ───────────────│
  │                                       │                               │
  │── on_message(assistant_response) ────►│── persist + display ─────────►│
  │                                       │                               │
```

---

## Storage Comparison

### Summary Matrix

| Aspect | Noema | Episteme | Lumina |
|--------|-------|----------|--------|
| **Language** | Rust | Python | Python |
| **Database** | SQLite | PostgreSQL | SQLite |
| **Content Model** | UCM (Turn/Span/View) | SpanSet/Span/Thread | N/A (Discord) |
| **Storage Pattern** | Trait-based | Service layer | Database classes |
| **Branching** | View/Fork | Thread-based | N/A |

### Noema (Reference Implementation)

Most advanced storage - already has UCM:

```
Conversation
  └── View (main_view_id)
        └── ViewSelection (turn → span)
              └── Turn (role)
                    └── Span (model_id, multiple per turn)
                          └── Message (sequence, role)
                                └── StoredContent
```

### Episteme (Migration Target)

Current model needs migration to UCM:
- SpanSet → Turn
- Thread → View
- Thread.selected_span_id → ViewSelection table

### Lumina (Read-Only)

No conversation storage - Discord owns the data:
- Implements `ConversationView` backed by Discord API
- Only reads history, doesn't persist

---

## Client Implementations

### Episteme (full storage)

```python
class StoredConversationView(ConversationView):
    def __init__(self, store: IConversationStore, conversation_id: str, ws: WebSocket):
        self.store = store
        self.conversation_id = conversation_id
        self.ws = ws

    def get_messages(self, limit=None):
        return self.store.get_history(self.conversation_id, limit)

    async def on_token(self, token: str):
        await self.ws.send_json({"type": "token", "data": token})

    async def on_message(self, message: Message):
        await self.store.save_message(self.conversation_id, message)
        await self.ws.send_json({"type": "message", "data": message.dict()})
```

### Lumina (Discord owns data)

```python
class DiscordConversationView(ConversationView):
    def __init__(self, channel: discord.TextChannel):
        self.channel = channel

    async def get_messages(self, limit=None):
        history = await self.channel.history(limit=limit).flatten()
        return [to_nous_message(m) for m in history]

    async def on_message(self, message: Message):
        await self.channel.send(message.text)  # Discord persists
```

### Testing (memory only)

```python
class MemoryConversationView(ConversationView):
    def __init__(self):
        self.messages = []
        self.tokens = []

    def get_messages(self, limit=None):
        return self.messages[-limit:] if limit else self.messages

    async def on_token(self, token: str):
        self.tokens.append(token)

    async def on_message(self, message: Message):
        self.messages.append(message)
```

---

## What Goes Where

### nous (shared core)

```
nous/
├── types/
│   ├── content.py      # ContentBlock (Text, Image, Audio)
│   ├── conversation.py # Message, Conversation
│   ├── tool.py         # ToolCall, ToolResult
│   └── provider.py     # ProviderConfig, ModelInfo
├── engine/
│   ├── engine.py       # Main Engine class
│   ├── context.py      # ContextBuilder
│   └── content.py      # ContentProcessor
├── llm/
│   ├── hub.py          # ProviderHub (factory + registry)
│   └── providers/      # Claude, OpenAI, Gemini, Ollama
├── mcp/
│   ├── client.py       # MCPClient
│   └── executor.py     # ToolExecutor
└── view/
    └── protocol.py     # ConversationView protocol
```

### Episteme (client)

```
episteme/
├── backend/
│   ├── api/            # FastAPI endpoints
│   ├── ws/             # WebSocket handlers
│   └── storage/        # PostgreSQL implementation
│       └── view.py     # StoredConversationView
└── frontend/           # React UI
```

### Lumina (client)

```
lumina/
├── agent/
│   └── view.py         # DiscordConversationView
├── services/
│   └── discord/        # Discord-specific services
└── cogs/               # Slash commands
```

---

## Open Questions

1. **Content addressing**: Should nous define content-addressed storage, or leave to implementations?
2. **View/branching**: Should Turn/Span/View be defined in nous or left to storage implementations?
3. **Async consistency**: Should `get_messages()` be async to support Discord's async API?
