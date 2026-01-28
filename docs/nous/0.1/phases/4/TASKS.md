# Phase 4: MCP - Tasks

**Status:** active
**Goal:** Tool execution with approval workflows

## Summary

| Pri | ✓ | Task | Title | Link |
|-----|---|------|-------|------|
| P0 | ⬜ | T4.1 | MCP Client | [#t41](#t41-mcp-client) |
| P0 | ⬜ | T4.2 | Tool Executor | [#t42](#t42-tool-executor) |
| P0 | ⬜ | T4.3 | View Integration | [#t43](#t43-view-integration) |
| P1 | ⬜ | T4.4 | Approval Workflows | [#t44](#t44-approval-workflows) |
| P1 | ⬜ | T4.5 | MCP Tests | [#t45](#t45-mcp-tests) |
| P2 | ⬜ | T4.6 | Demo with Tools | [#t46](#t46-demo-with-tools) |

## Source Files (Episteme → Nous)

| Source | Destination | Status |
|--------|-------------|--------|
| `backend/mcp/client.py` | `src/nous/mcp/client.py` | todo |
| `backend/mcp/executor.py` | `src/nous/mcp/executor.py` | todo |
| `backend/mcp/discovery.py` | `src/nous/mcp/discovery.py` | todo |

## P0 - Must Have

### T4.1: MCP Client
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/mcp/client.py`, `src/nous/mcp/__init__.py`

Extract MCPClient from Episteme:

```python
class MCPClient:
    """Client for MCP server communication."""

    async def connect(self, server_config: MCPServerConfig) -> None:
        """Connect to an MCP server."""
        ...

    async def list_tools(self) -> list[ToolDefinition]:
        """Get available tools from connected servers."""
        ...

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Execute a tool and return result."""
        ...
```

**Design notes:**
- Handle multiple MCP server connections
- Support stdio and SSE transports
- Graceful connection/disconnection

### T4.2: Tool Executor
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/mcp/executor.py`

Execute tools with result formatting:

```python
class ToolExecutor:
    """Executes tool calls via MCP."""

    def __init__(self, client: MCPClient):
        self.client = client

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return formatted result."""
        result = await self.client.call_tool(
            tool_call.name,
            tool_call.arguments
        )
        return ToolResult(
            tool_call_id=tool_call.id,
            content=[TextContent(text=str(result))]
        )
```

**Design notes:**
- Convert MCP responses to ToolResult
- Handle tool errors gracefully
- Support timeout/cancellation

### T4.3: View Integration
**Priority:** P0 | **Status:** todo
**Files:** `src/nous/view/protocol.py` (if needed)

Wire tool execution into ConversationView.on_tool_call():

```python
# In a concrete view implementation:
async def on_tool_call(self, tool_call: ToolCall) -> ToolResult:
    # 1. Check approval (auto, prompt, deny)
    approved = await self.check_approval(tool_call)
    if not approved:
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content=[TextContent(text="Tool call denied")]
        )

    # 2. Execute via ToolExecutor
    return await self.executor.execute(tool_call)
```

**Design notes:**
- Approval is view's responsibility
- View owns the ToolExecutor instance
- Engine just waits for ToolResult

## P1 - Should Have

### T4.4: Approval Workflows
**Priority:** P1 | **Status:** todo
**Files:** `src/nous/mcp/approval.py`

Approval patterns for tool execution:

```python
class ApprovalPolicy(Protocol):
    """Policy for tool call approval."""

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        """Check if tool call should be approved."""
        ...

class AutoApprovePolicy:
    """Approve all tool calls."""
    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        return ApprovalResult.APPROVED

class AllowlistPolicy:
    """Approve tools on allowlist."""
    def __init__(self, allowed: set[str]):
        self.allowed = allowed

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        if tool_call.name in self.allowed:
            return ApprovalResult.APPROVED
        return ApprovalResult.PROMPT
```

**Design notes:**
- Policies are composable
- PROMPT result means ask user (view handles UI)
- Episteme/Lumina implement their own UI flows

### T4.5: MCP Tests
**Priority:** P1 | **Status:** todo
**Files:** `tests/test_mcp.py`

Test MCP with mock server:

- Client connection/disconnection
- Tool listing
- Tool execution
- Error handling
- Approval policies

### T4.6: Demo with Tools
**Priority:** P2 | **Status:** todo
**Files:** `src/nous/demo.py`

Update demo to support tool calling:

```bash
uv run python -m nous.demo --mcp-server path/to/server.py
# Interactive chat with tool access
```

---

## Test

```bash
uv run pytest tests/test_mcp.py
# Engine calls tool, view executes via MCP, returns result
```

---

## Dependencies

Phase 3 components used:
- `Engine` for conversation turns
- `ConversationView.on_tool_call()` callback
- `MemoryView` for testing

Types used:
- `ToolCall`, `ToolResult` for tool flow
- `ToolDefinition` for tool schemas
