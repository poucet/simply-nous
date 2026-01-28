"""Streaming event types for LLM responses."""

from dataclasses import dataclass

from nous.types import Message, ToolCall


@dataclass
class TextDeltaEvent:
    """Incremental text chunk from a streaming response."""

    text: str


@dataclass
class ToolCallEvent:
    """A tool call parsed from the response stream."""

    tool_call: ToolCall


@dataclass
class MessageCompleteEvent:
    """Final complete message after streaming finishes."""

    message: Message


StreamEvent = TextDeltaEvent | ToolCallEvent | MessageCompleteEvent
