# Phase 3: Engine - Handoff

Context and continuity for Phase 4 (MCP).

---

## System State

Phase 3 delivered a complete storage-agnostic engine:

- **Engine** (`src/nous/engine/engine.py`) - Runs conversation turns with tool call loop
- **ContextBuilder** (`src/nous/engine/context.py`) - Pluggable protocol for building LLM context
- **ContentProcessor** (`src/nous/engine/content.py`) - Adapts content to model capabilities
- **MemoryView** (`src/nous/view/memory.py`) - In-memory ConversationView for testing/demos

All streaming flows through ConversationView callbacks:
- `on_token()` - text deltas
- `on_tool_call()` - tool execution (returns ToolResult)
- `on_message()` - complete messages
- `on_knowledge_needed()` - RAG retrieval

Demo CLI works with Ollama for local testing.

## Architectural Notes

**Key Pattern: ContextBuilder as Protocol**
- ContextBuilder is pluggable, not hardcoded in Engine
- Allows custom query extraction, mid-stream RAG, custom formatting
- DefaultContextBuilder provides sensible defaults

**RAG via Callback**
- `on_knowledge_needed(query)` callback in ConversationView
- Engine orchestrates when to retrieve, view provides how
- `KnowledgeChunk` type supports multimodal content (text, images, audio)

**Tool Call Flow**
- Engine calls `view.on_tool_call(tool_call)`
- View handles approval and execution
- Engine continues with returned `ToolResult`

## Open Questions / Risks

- Tool approval workflows are client responsibility (view implementation)
- MCP integration will need to wire into on_tool_call() pattern
- May need batched tool calls for parallel execution

## Next Steps

Phase 4 should:
1. Extract MCPClient from Episteme
2. Create ToolExecutor that runs MCP tools
3. Wire MCP tools into ConversationView.on_tool_call()
4. Consider approval workflow patterns (auto-approve, confirm, deny)
