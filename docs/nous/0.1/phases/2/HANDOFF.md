# Phase 2: LLM Layer - Handoff

Context and continuity for Phase 3.

---

## System State

**What's working:**
- LLMProvider/ModelClient protocol pattern established
- 6 providers implemented: Anthropic, OpenAI, Gemini, Ollama, Mistral, OpenRouter
- ProviderHub with parallel model lookup across providers
- CachingProvider shim for `list_models()` results
- Stream events: TextDelta, ToolCall, MessageComplete
- ModelCapabilities registry for filtering by features
- Provider configuration via env vars and HubConfig
- 32 tests passing

**Module structure:**
```
nous/llm/
├── __init__.py          # Public API
├── hub.py               # ProviderHub registry
├── protocol.py          # LLMProvider, ModelClient protocols
├── events.py            # StreamEvent types
├── capabilities.py      # ModelCapabilities, ModelRegistry
├── config.py            # ProviderConfig, HubConfig
└── providers/
    ├── anthropic.py
    ├── openai.py
    ├── gemini.py
    ├── ollama.py
    ├── mistral.py
    └── openrouter.py
```

## Architectural Notes

**Provider pattern:**
- `LLMProvider` handles connection/auth, returns `ModelClient` via `.model(id)`
- `ModelClient` has single responsibility: `complete()` method with streaming
- Hub manages model→provider mapping for `hub.get_for_model("gpt-4")`

**Streaming design:**
- All providers yield `StreamEvent` (TextDelta | ToolCall | MessageComplete)
- Non-streaming mode still returns final `MessageComplete` event
- Callers iterate async: `async for event in client.complete(...)`

**Key decisions:**
1. No base classes - protocols only (duck typing)
2. Providers are stateless factories for model clients
3. Caching is a transparent shim, not baked into providers
4. OpenRouter uses OpenAI SDK with custom base_url (no separate SDK)

## Open Questions / Risks

1. **RAG placement** - Where does retrieval-augmented generation live?
   - Option A: In nous engine (context builder adds retrieved docs)
   - Option B: In client (ConversationView provides augmented context)
   - Decision needed before Phase 3 engine work

2. **Tool definition source** - Phase 4 MCP will need tool schemas
   - Currently ToolDefinition exists in types but not wired to providers

3. **Multi-modal content** - Image/audio content blocks defined but not tested end-to-end

## Next Steps

For Phase 3 (Engine):

1. Start with `ConversationView` → engine data flow
2. Extract ContextBuilder from Episteme (builds prompt from view)
3. Extract ContentProcessor (handles multi-modal content)
4. Wire engine to ProviderHub for provider-agnostic completion
5. Implement callbacks: `on_token()`, `on_tool_call()`, `on_message()`
6. Create `MemoryConversationView` for testing (simple in-memory impl)
