# Nous (νοῦς)

> The capacity to think — shared AI core for the Simply ecosystem.

**Nous** is the shared foundation for [Episteme](../episteme) and [Lumina](../lumina), providing:

- **Types** — ContentBlock, Message, ToolCall, ToolResult
- **LLM** — Provider abstraction (Claude, OpenAI, Gemini, Ollama, Mistral, OpenRouter)
- **Engine** — Storage-agnostic conversation engine with streaming
- **View** — ConversationView protocol for bidirectional client communication
- **MCP** — Tool execution with approval workflows

## Installation

```bash
# Core only
uv add simply-nous

# With specific providers
uv add "simply-nous[claude]"
uv add "simply-nous[openai]"
uv add "simply-nous[mcp]"

# Everything
uv add "simply-nous[all]"
```

## Quick Start

```python
from nous.engine import Engine
from nous.llm.providers import OllamaProvider
from nous.view.memory import MemoryConversationView

# Create provider and view
provider = OllamaProvider()
client = provider.model("llama3.2")
view = MemoryConversationView()

# Add user message and run
view.add_user_message("Hello!")
engine = Engine()
response = await engine.run_turn(client, view)
```

## Architecture

```
nous/
├── types/      # Pydantic models (ContentBlock, Message, ToolCall, etc.)
├── llm/        # Provider abstraction and ProviderHub
│   └── providers/  # Claude, OpenAI, Gemini, Ollama, Mistral, OpenRouter
├── engine/     # Conversation engine with streaming callbacks
├── view/       # ConversationView protocol + MemoryConversationView
├── mcp/        # MCP client, executor, and approval policies
└── storage/    # Abstract storage interfaces
```

### Key Patterns

1. **ConversationView protocol** — Engine reads state from view, pushes events via callbacks
2. **Provider abstraction** — Unified interface across LLM providers
3. **Streaming** — All completions stream through view callbacks
4. **Tool execution** — View handles approval, MCP executor runs tools

## Demo CLI

Interactive chat using Ollama:

```bash
# Basic usage (requires Ollama running locally)
uv run python -m nous.demo

# Specify model
uv run python -m nous.demo --model llama3.2

# With MCP tools
uv run python -m nous.demo --mcp-server http://localhost:8080/mcp
```

## Development

```bash
uv sync --all-extras
uv run pytest
```

## The Trilogy

| Project | Greek | Role |
|---------|-------|------|
| **Noema** | νόημα (thought-object) | Data structures, content model |
| **Nous** | νοῦς (intellect) | Thinking capacity, shared core |
| **Episteme** | ἐπιστήμη (knowledge) | Platform, resulting knowledge |

## Status

**v0.1.4** — Core complete, ready for integration.

- Phase 1: Foundation (types, view protocol) ✓
- Phase 2: LLM Layer (providers, hub) ✓
- Phase 3: Engine (conversation, streaming) ✓
- Phase 4: MCP (tools, approval) ✓
