"""Approval policies for tool execution.

Policies determine whether tool calls should be approved automatically,
denied, or require user confirmation.

Example:
    >>> from nous.mcp import AllowlistPolicy, ApprovalResult
    >>> policy = AllowlistPolicy(allowed={"read_file", "list_files"})
    >>> result = await policy.check(tool_call)
    >>> if result == ApprovalResult.APPROVED:
    ...     # Execute tool
    >>> elif result == ApprovalResult.PROMPT:
    ...     # Ask user for confirmation
"""

from enum import Enum
from typing import Protocol

from nous.types import ToolCall


class ApprovalResult(Enum):
    """Result of an approval check."""

    APPROVED = "approved"
    DENIED = "denied"
    PROMPT = "prompt"


class ApprovalPolicy(Protocol):
    """Protocol for tool call approval policies."""

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        """Check if a tool call should be approved.

        Args:
            tool_call: The tool call to check.

        Returns:
            ApprovalResult indicating whether to approve, deny, or prompt.
        """
        ...


class AutoApprovePolicy:
    """Approve all tool calls automatically."""

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        """Always returns APPROVED."""
        return ApprovalResult.APPROVED


class AutoDenyPolicy:
    """Deny all tool calls."""

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        """Always returns DENIED."""
        return ApprovalResult.DENIED


class AllowlistPolicy:
    """Approve tools on allowlist, prompt for others."""

    def __init__(self, allowed: set[str]) -> None:
        """Initialize with set of allowed tool names.

        Args:
            allowed: Set of tool names to auto-approve.
        """
        self.allowed = allowed

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        """Approve if tool is in allowlist, otherwise prompt."""
        if tool_call.name in self.allowed:
            return ApprovalResult.APPROVED
        return ApprovalResult.PROMPT


class DenylistPolicy:
    """Deny tools on denylist, approve others."""

    def __init__(self, denied: set[str]) -> None:
        """Initialize with set of denied tool names.

        Args:
            denied: Set of tool names to auto-deny.
        """
        self.denied = denied

    async def check(self, tool_call: ToolCall) -> ApprovalResult:
        """Deny if tool is in denylist, otherwise approve."""
        if tool_call.name in self.denied:
            return ApprovalResult.DENIED
        return ApprovalResult.APPROVED
