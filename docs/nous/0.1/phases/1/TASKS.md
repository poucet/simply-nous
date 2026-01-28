# Phase 1: Foundation - Tasks

**Status:** active
**Goal:** Extract types and LLM layer from Episteme

## Completed

- [x] **Project setup** - pyproject.toml, README.md, .claude/CLAUDE.md
- [x] **Types package** - content.py, conversation.py, tool.py, provider.py
- [x] **Storage protocols** - IConversationStore, IKnowledgeStore
- [x] **Test suite** - test_types.py with 4 passing tests

## Remaining

### T1.1: LLM ProviderHub
**Priority:** P0
**Files:** `src/nous/llm/hub.py`

Create the ProviderHub class that manages multiple LLM providers:
- Provider registration and lookup
- Unified generate() interface
- Model routing

### T1.2: Claude Provider
**Priority:** P0
**Files:** `src/nous/llm/providers/claude.py`

Extract and adapt Claude provider from Episteme:
- Anthropic SDK integration
- Message format conversion
- Streaming support

### T1.3: OpenAI Provider
**Priority:** P1
**Files:** `src/nous/llm/providers/openai.py`

Extract and adapt OpenAI provider:
- OpenAI SDK integration
- Message format conversion
- Streaming support

### T1.4: Gemini Provider
**Priority:** P2
**Files:** `src/nous/llm/providers/gemini.py`

Extract and adapt Gemini provider:
- Google AI SDK integration
- Message format conversion
- Streaming support

### T1.5: Provider Tests
**Priority:** P0
**Files:** `tests/test_llm.py`

Add tests for LLM layer:
- ProviderHub tests
- Mock provider tests
- Integration tests (optional, require API keys)
