"""Tests for MCP client, executor, and approval policies."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nous.mcp import (
    ApprovalResult,
    AutoApprovePolicy,
    AutoDenyPolicy,
    AllowlistPolicy,
    DenylistPolicy,
    ToolExecutor,
)
from nous.types import ToolCall, TextContent


# =============================================================================
# Approval Policy Tests
# =============================================================================


class TestApprovalResult:
    def test_enum_values(self):
        assert ApprovalResult.APPROVED.value == "approved"
        assert ApprovalResult.DENIED.value == "denied"
        assert ApprovalResult.PROMPT.value == "prompt"


class TestAutoApprovePolicy:
    @pytest.fixture
    def policy(self):
        return AutoApprovePolicy()

    @pytest.fixture
    def tool_call(self):
        return ToolCall(name="read_file", input={"path": "/tmp/test"})

    async def test_always_approves(self, policy, tool_call):
        result = await policy.check(tool_call)
        assert result == ApprovalResult.APPROVED

    async def test_approves_any_tool(self, policy):
        for name in ["dangerous_tool", "delete_all", "format_disk"]:
            call = ToolCall(name=name, input={})
            result = await policy.check(call)
            assert result == ApprovalResult.APPROVED


class TestAutoDenyPolicy:
    @pytest.fixture
    def policy(self):
        return AutoDenyPolicy()

    async def test_always_denies(self, policy):
        call = ToolCall(name="read_file", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.DENIED


class TestAllowlistPolicy:
    @pytest.fixture
    def policy(self):
        return AllowlistPolicy(allowed={"read_file", "list_files"})

    async def test_approves_allowed_tools(self, policy):
        call = ToolCall(name="read_file", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.APPROVED

    async def test_prompts_for_unknown_tools(self, policy):
        call = ToolCall(name="delete_file", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.PROMPT

    async def test_empty_allowlist_prompts_all(self):
        policy = AllowlistPolicy(allowed=set())
        call = ToolCall(name="any_tool", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.PROMPT


class TestDenylistPolicy:
    @pytest.fixture
    def policy(self):
        return DenylistPolicy(denied={"delete_file", "format_disk"})

    async def test_denies_blocked_tools(self, policy):
        call = ToolCall(name="delete_file", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.DENIED

    async def test_approves_unblocked_tools(self, policy):
        call = ToolCall(name="read_file", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.APPROVED

    async def test_empty_denylist_approves_all(self):
        policy = DenylistPolicy(denied=set())
        call = ToolCall(name="any_tool", input={})
        result = await policy.check(call)
        assert result == ApprovalResult.APPROVED


# =============================================================================
# ToolExecutor Tests
# =============================================================================


class TestToolExecutor:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.call_tool = AsyncMock()
        return client

    @pytest.fixture
    def executor(self, mock_client):
        return ToolExecutor(mock_client, timeout=5.0)

    @pytest.fixture
    def tool_call(self):
        return ToolCall(name="read_file", input={"path": "/tmp/test"})

    async def test_execute_returns_tool_result(self, executor, mock_client, tool_call):
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "file contents"
        mock_client.call_tool.return_value = [mock_content]

        result = await executor.execute(tool_call)

        assert result.tool_call_id == tool_call.id
        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0].text == "file contents"

    async def test_execute_handles_tool_not_found(self, executor, mock_client, tool_call):
        mock_client.call_tool.side_effect = ValueError("Tool 'read_file' not found")

        result = await executor.execute(tool_call)

        assert result.is_error is True
        assert "not found" in result.content[0].text

    async def test_execute_handles_timeout(self, executor, mock_client, tool_call):
        import asyncio
        mock_client.call_tool.side_effect = asyncio.TimeoutError()

        result = await executor.execute(tool_call, timeout=0.1)

        assert result.is_error is True
        assert "timed out" in result.content[0].text

    async def test_execute_handles_generic_error(self, executor, mock_client, tool_call):
        mock_client.call_tool.side_effect = RuntimeError("Connection lost")

        result = await executor.execute(tool_call)

        assert result.is_error is True
        assert "failed" in result.content[0].text

    async def test_execute_with_image_content(self, executor, mock_client, tool_call):
        mock_content = MagicMock()
        mock_content.type = "image"
        mock_content.mimeType = "image/png"
        mock_content.data = "base64data"
        mock_client.call_tool.return_value = [mock_content]

        result = await executor.execute(tool_call)

        assert result.is_error is False
        assert result.content[0].mime_type == "image/png"

    async def test_execute_with_empty_result(self, executor, mock_client, tool_call):
        mock_client.call_tool.return_value = []

        result = await executor.execute(tool_call)

        assert result.is_error is False
        assert len(result.content) == 1
        assert "successfully" in result.content[0].text

    async def test_uses_default_timeout(self, mock_client):
        executor = ToolExecutor(mock_client, timeout=30.0)
        assert executor.timeout == 30.0

    async def test_timeout_override(self, executor, mock_client, tool_call):
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "ok"
        mock_client.call_tool.return_value = [mock_content]

        # Should use override timeout, not default
        result = await executor.execute(tool_call, timeout=60.0)

        assert result.is_error is False


# =============================================================================
# MCPClient Tests (unit tests with mocks)
# =============================================================================


class TestMCPClientBasics:
    def test_initial_state(self):
        from nous.mcp import MCPClient
        client = MCPClient()
        assert client.connected_servers == []

    async def test_connect_requires_mcp_package(self):
        from nous.mcp import MCPClient, MCPServerConfig

        client = MCPClient()
        config = MCPServerConfig(name="test", url="http://localhost:8080")

        # This will fail with ImportError if mcp isn't installed,
        # or try to connect to a non-existent server
        # We just verify the interface exists
        assert hasattr(client, "connect")
        assert hasattr(client, "disconnect")
        assert hasattr(client, "list_tools")
        assert hasattr(client, "call_tool")

    async def test_context_manager_protocol(self):
        from nous.mcp import MCPClient

        async with MCPClient() as client:
            assert client.connected_servers == []
        # Should not raise


class TestMCPServerConfig:
    def test_creates_config(self):
        from nous.mcp import MCPServerConfig

        config = MCPServerConfig(name="tools", url="http://localhost:8080/mcp")
        assert config.name == "tools"
        assert config.url == "http://localhost:8080/mcp"

    def test_validates_required_fields(self):
        from nous.mcp import MCPServerConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MCPServerConfig(name="test")  # missing url
