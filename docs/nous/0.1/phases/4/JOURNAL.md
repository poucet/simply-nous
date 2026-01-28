# Phase 4: MCP - Journal

Chronological stream of thoughts, changes, and observations.

---

## 2026-01-28

### T4.1: MCP Client - Complete

Implemented `MCPClient` in `src/nous/mcp/client.py`:

- `MCPServerConfig` - Pydantic model with validation for stdio/SSE transport
- Connection management with `AsyncExitStack` for clean resource handling
- Multi-server support with tool-to-server mapping
- Async context manager for convenience

Added `mcp>=1.0` as optional dependency. All 82 tests pass.

### T4.2: Tool Executor - Complete

Implemented `ToolExecutor` in `src/nous/mcp/executor.py`:

- `execute(tool_call, timeout)` - executes via MCPClient, returns `ToolResult`
- Timeout support with configurable default (30s)
- Error handling for timeout, tool not found, and execution failures
- Content conversion from MCP types (text, image) to nous types

### T4.3: View Integration - Complete

Wired ToolExecutor into MemoryConversationView:

- Added docstring example showing `tool_handler=executor.execute` pattern
- Kept design simple: `tool_handler` callback is the single extension point
- Avoided adding redundant `executor` param (user just passes `executor.execute`)

All P0 tasks complete. Phase 4 now has approval workflows (P1) and tests remaining.

### T4.4: Approval Workflows - Complete

Created `src/nous/mcp/approval.py` with:

- `ApprovalResult` enum: APPROVED, DENIED, PROMPT
- `ApprovalPolicy` protocol for custom policies
- `AutoApprovePolicy` / `AutoDenyPolicy` - blanket policies
- `AllowlistPolicy` / `DenylistPolicy` - name-based filtering

View handles UI for PROMPT results. Policies are composable.

### T4.5: MCP Tests - Complete

Added `tests/test_mcp.py` with 23 tests covering:

- Approval policies: AutoApprove, AutoDeny, Allowlist, Denylist
- ToolExecutor: success, errors, timeouts, content conversion
- MCPClient basics: config validation, context manager

Total test count now 105.

### T4.6: Demo with Tools - Complete

Updated `src/nous/demo.py` with `--mcp-server` flag:

- Accepts HTTP URL for MCP server connection
- Wires MCPClient + ToolExecutor into DemoView
- Prints tool calls and results during conversation
- Clean disconnect on exit

**Phase 4 complete.** All tasks done: MCPClient, ToolExecutor, View integration, approval policies, tests, and demo.

---

## 2026-01-29

### Post-release: ModelInfo enhancement

Enhanced `list_models()` across all providers to return `ModelInfo` instead of plain strings:

- Added `ModelInfo` dataclass with `id`, `name`, `provider`, and `capabilities`
- Added `context_window` field to `ModelCapabilities`
- Each provider now infers capabilities from model metadata (vision, audio, tools, context)
- Updated `LLMProvider` protocol, `CachingProvider`, `ProviderHub`
- Demo CLI updated to extract `.id` from ModelInfo

This enables better model selection based on capabilities.
