# Nous (νοῦς)

> The capacity to think - shared AI core for the Simply ecosystem.

**Nous** is the shared foundation for [Episteme](../episteme) and [Lumina](../lumina), providing:

- **Types**: ContentBlock, Message, Thread, Conversation, ToolCall
- **LLM**: ProviderHub with Claude, OpenAI, Gemini, Ollama support
- **MCP**: Tool execution with approval workflows
- **Storage**: Abstract interfaces for conversation and knowledge stores

## The Trilogy

| Project | Greek | Role |
|---------|-------|------|
| **Noema** | νόημα (thought-object) | Data structures, content model |
| **Nous** | νοῦς (intellect) | Thinking capacity, shared core |
| **Episteme** | ἐπιστήμη (knowledge) | Platform, resulting knowledge |

## Installation

```bash
# Core only
uv add simply-nous

# With providers
uv add "simply-nous[claude]"
uv add "simply-nous[all]"
```

## Usage

```python
from nous.types import ContentBlock, Message
from nous.llm import ProviderHub

hub = ProviderHub()
response = await hub.generate("claude", messages=[
    Message(role="user", content=[ContentBlock(type="text", text="Hello")])
])
```

## Development

```bash
uv sync --all-extras
uv run pytest
```
