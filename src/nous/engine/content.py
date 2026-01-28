"""Content processing for provider compatibility.

ContentProcessor adapts message content to model capabilities using
pluggable strategies. Async to support external service calls.

Example:
    >>> from nous.engine.content import ContentProcessor, PlaceholderAdapter
    >>>
    >>> processor = ContentProcessor(image_adapter=PlaceholderAdapter())
    >>> adapted = await processor.prepare_for_model(messages, caps)

Custom adapter example (vision API):
    >>> class VisionAdapter:
    ...     async def adapt(self, content: ImageContent) -> ContentBlock:
    ...         description = await vision_api.describe(content.data)
    ...         return TextContent(text=description)
"""

from typing import Protocol

from nous.llm.capabilities import ModelCapabilities
from nous.types import (
    Message,
    TextContent,
    ImageContent,
    AudioContent,
    ToolResultContent,
    ContentBlock,
)


class ImageAdapter(Protocol):
    """Strategy for adapting images when model lacks vision."""

    async def adapt(self, content: ImageContent) -> ContentBlock:
        """Convert image to alternative content.

        Args:
            content: Image content with data loaded.

        Returns:
            Adapted content (TextContent, or preserved ImageContent).
        """
        ...


class AudioAdapter(Protocol):
    """Strategy for adapting audio when model lacks audio input."""

    async def adapt(self, content: AudioContent) -> ContentBlock:
        """Convert audio to alternative content.

        Args:
            content: Audio content with data loaded.

        Returns:
            Adapted content (TextContent, or preserved AudioContent).
        """
        ...


class PlaceholderImageAdapter:
    """Simple adapter that replaces images with placeholder text."""

    def __init__(self, placeholder: str = "[Image attached]") -> None:
        self.placeholder = placeholder

    async def adapt(self, _content: ImageContent) -> ContentBlock:
        return TextContent(text=self.placeholder)


class PlaceholderAudioAdapter:
    """Simple adapter that replaces audio with placeholder text."""

    def __init__(self, placeholder: str = "[Audio attached]") -> None:
        self.placeholder = placeholder

    async def adapt(self, _content: AudioContent) -> ContentBlock:
        return TextContent(text=self.placeholder)


class ContentProcessor:
    """Adapts message content to model capabilities.

    Storage-agnostic: expects all content data to be pre-loaded.
    Returns copies of messages - never mutates originals.

    Args:
        image_adapter: Strategy for images when model lacks vision.
        audio_adapter: Strategy for audio when model lacks audio_input.
    """

    def __init__(
        self,
        image_adapter: ImageAdapter | None = None,
        audio_adapter: AudioAdapter | None = None,
    ) -> None:
        self.image_adapter = image_adapter or PlaceholderImageAdapter()
        self.audio_adapter = audio_adapter or PlaceholderAudioAdapter()

    async def prepare_for_model(
        self,
        messages: list[Message],
        capabilities: ModelCapabilities,
    ) -> list[Message]:
        """Adapt messages to model capabilities.

        Converts unsupported content types using configured adapters.

        Args:
            messages: Conversation messages with loaded content.
            capabilities: Target model's capabilities.

        Returns:
            Adapted messages (copies, originals unchanged).
        """
        adapted = []
        for msg in messages:
            adapted.append(await self._adapt_message(msg, capabilities))
        return adapted

    async def _adapt_message(
        self,
        message: Message,
        capabilities: ModelCapabilities,
    ) -> Message:
        """Adapt a single message's content blocks."""
        adapted_content = []
        for block in message.content:
            adapted_content.append(await self._adapt_block(block, capabilities))
        return Message(role=message.role, content=adapted_content)

    async def _adapt_block(
        self,
        block: ContentBlock,
        capabilities: ModelCapabilities,
    ) -> ContentBlock:
        """Adapt a single content block based on capabilities."""
        match block:
            case ImageContent() if not capabilities.vision:
                return await self.image_adapter.adapt(block)

            case AudioContent() if not capabilities.audio_input:
                return await self.audio_adapter.adapt(block)

            case ToolResultContent() as tool_result:
                adapted_content = []
                for c in tool_result.content:
                    adapted_content.append(
                        await self._adapt_tool_content(c, capabilities)
                    )
                return ToolResultContent(
                    tool_use_id=tool_result.tool_use_id,
                    content=adapted_content,
                    is_error=tool_result.is_error,
                )

            case _:
                return block

    async def _adapt_tool_content(
        self,
        content: TextContent | ImageContent | AudioContent,
        capabilities: ModelCapabilities,
    ) -> TextContent | ImageContent | AudioContent:
        """Adapt content within tool results."""
        match content:
            case ImageContent() if not capabilities.vision:
                return await self.image_adapter.adapt(content)
            case AudioContent() if not capabilities.audio_input:
                return await self.audio_adapter.adapt(content)
            case _:
                return content

    def extract_text(self, messages: list[Message]) -> str:
        """Extract all text content from messages.

        Useful for embedding generation or summarization.

        Args:
            messages: Messages to extract text from.

        Returns:
            Concatenated text content, newline-separated.
        """
        texts = []
        for message in messages:
            for block in message.content:
                if isinstance(block, TextContent):
                    texts.append(block.text)
        return "\n".join(texts)
