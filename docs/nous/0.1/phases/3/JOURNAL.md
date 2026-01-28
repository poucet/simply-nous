# Phase 3: Engine - Journal

Chronological stream of thoughts, changes, and observations.

---

## 2026-01-28 - T3.1 Engine Core + RAG Design

**Context**: Starting Phase 3 with Engine implementation

**Notes**:
- Discussed RAG placement: decided on Option C - `on_knowledge_needed` callback in ConversationView
- Engine orchestrates when to retrieve, view provides how
- Created `KnowledgeChunk` type (not `Document`) for multimodal RAG context
- Supports `list[ContentBlock]` for text, images, audio in retrieved chunks
- Metadata dict for flexible structure (tabs, sections, chunk index)

**Key Design Decision**:
- ContextBuilder is a pluggable protocol, not hardcoded in engine
- Allows mid-stream RAG injection, custom query extraction, custom formatting
- DefaultContextBuilder provides sensible defaults

**Changes**:
- `src/nous/types/knowledge.py` - KnowledgeChunk type
- `src/nous/view/protocol.py` - on_knowledge_needed callback
- `src/nous/engine/context.py` - ContextBuilder protocol + DefaultContextBuilder
- `src/nous/engine/engine.py` - Engine class with tool call loop

---
