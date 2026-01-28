# Nous Compatibility Checkpoints

Track alignment between nous and its consumers.

## Targets

| Name | Path |
|------|------|
| lumina | ~/projects/simply/lumina |
| episteme | ~/projects/simply/episteme |

## Checkpoints

### CP1: Message Types

**Expects:** `Message` with `list[ContentBlock]` where ContentBlock = TextContent | ImageContent | AudioContent

**Check prompt:**
```
Find message/content type definitions.
Look for: Message class/struct, ContentBlock, TextContent, ImageContent, content arrays
Report: file path, type structure, field names
```

**lumina status:** misaligned — Uses LangChain's `BaseMessage`/`HumanMessage`/`AIMessage`, not nous types
**episteme status:** aligned — `MessageResponse.content: List[ContentBlock]` with `ContentBlock = Union[TextContent, ImageContent, AudioContent, ...]`

---

### CP2: Streaming Callbacks

**Expects:** `on_text_delta(text: str)` for streaming text, `on_content_block(block: ContentBlock)` for complete blocks

**Check prompt:**
```
Find streaming/callback patterns for LLM responses.
Look for: on_token, on_delta, on_chunk, on_stream, onChunk, streaming handlers
Report: callback signature, file path, how text vs complete content is handled
```

**lumina status:** misaligned — Uses LangChain iterator (`ChatGenerationChunk`) + realtime audio callbacks, not `on_text_delta` pattern
**episteme status:** misaligned — Uses async iterator with `StreamChunk` types (`TextDeltaChunk`), events to UI, not callbacks

---

### CP3: Tool Call Flow

**Expects:** `on_tool_call(tool_call: ToolCall) -> ToolResult` — engine calls view, view returns result

**Check prompt:**
```
Find tool/function call handling patterns.
Look for: ToolCall, ToolResult, tool execution, function calling, MCP
Report: how tool calls flow between engine and UI, approval patterns if any
```

**lumina status:** partial — Has `require_owner_confirmation()` decorator (Discord DM + emoji reactions), but only on identity ops; LangChain agent path has no approval
**episteme status:** aligned — Has approval flow: `ToolCallRequestedEvent` → `set_approval_decision()` → execute

---

### CP4: Conversation View Pattern

**Expects:** Protocol with read methods (get_messages, get_system_prompt) and write callbacks (on_*)

**Check prompt:**
```
Find conversation/chat view abstractions.
Look for: ConversationView, ChatView, delegate patterns, protocols, view models
Report: how conversation state is accessed, how updates are pushed to UI
```

**lumina status:** not found — Discord-first architecture; uses Discord channel history API, no ConversationView protocol
**episteme status:** misaligned — React hooks + WebSocket events, not a protocol with read methods + write callbacks

---

### CP5: Model Selection

**Expects:** `model_id: str` property on view

**Check prompt:**
```
Find how model selection is handled.
Look for: model_id, model selection, provider config, model picker
Report: where model choice lives, how it's passed to LLM calls
```

**lumina status:** misaligned — `AgentState.selected_model` on Agent class, not on a view
**episteme status:** misaligned — Model passed as parameters through call chain, stored in conversation/message

---

## History

| Date | Checkpoints | Result | Notes |
|------|-------------|--------|-------|
| 2026-01-28 | CP1-CP5 | 2 aligned, 1 partial, 6 misaligned, 1 not found | Initial check; lumina has Discord-native approval via `require_owner_confirmation()` |
