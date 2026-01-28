# Nous 0.1 Roadmap

> The capacity to think - shared AI core for the Simply ecosystem.

## The Trilogy

| Project | Greek | Role |
|---------|-------|------|
| **Noema** | νόημα (thought-object) | Data structures, content model |
| **Nous** | νοῦς (intellect) | Thinking capacity, shared core |
| **Episteme** | ἐπιστήμη (knowledge) | Platform, resulting knowledge |

## Phases

### Phase 1: Foundation ✅ (active)
**Version:** 0.1.0
**Goal:** Extract types and LLM layer from Episteme

- [x] Project structure and pyproject.toml
- [x] Core types: ContentBlock, Message, Thread, Conversation
- [x] Tool types: ToolCall, ToolResult
- [x] Storage protocols: IConversationStore, IKnowledgeStore
- [ ] LLM ProviderHub abstraction
- [ ] Provider implementations (Claude, OpenAI, Gemini)

### Phase 2: MCP Integration (pending)
**Version:** 0.2.0
**Goal:** Extract MCP client and tool executor

- [ ] MCP client from Episteme
- [ ] Tool executor with approval workflows
- [ ] Tool schema definitions

### Phase 3: Storage Interfaces (pending)
**Version:** 0.3.0
**Goal:** Define conversation and knowledge store protocols

- [ ] Conversation store implementations
- [ ] Knowledge/RAG store implementations
- [ ] Migration utilities

## Key Patterns

1. **Protocol-based interfaces** - Use `typing.Protocol` for storage interfaces
2. **Pydantic models** - All types are Pydantic BaseModel subclasses
3. **Async everywhere** - All I/O operations use async/await
4. **Provider-agnostic** - LLM layer abstracts provider differences

## Source Extraction Map

| Episteme Source | Nous Destination |
|-----------------|------------------|
| `api/src/noema_schemas/content.py` | `src/nous/types/content.py` |
| `api/src/noema_schemas/conversation.py` | `src/nous/types/conversation.py` |
| `api/src/noema_schemas/tool.py` | `src/nous/types/tool.py` |
| `backend/src/backend/llm/hub.py` | `src/nous/llm/hub.py` |
| `backend/src/backend/llm/providers/` | `src/nous/llm/providers/` |
