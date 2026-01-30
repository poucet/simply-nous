# CLAUDE.md

This file provides guidance to Claude Code when working with the nous codebase.

## Project Overview

Nous is the shared AI core extracted from Episteme. It provides types, LLM abstraction, and MCP tools used by both Episteme and Lumina.

## Development Commands

- `uv sync` - Install dependencies
- `uv run pytest` - Run tests
- `uv run python -m nous.demo` - Run demo CLI

## Architecture

```
nous/
├── types/      # Pydantic models (ContentBlock, Message, etc.)
├── llm/        # LLM provider abstraction (ProviderHub)
└── mcp/        # MCP client and tool execution
```

## Key Patterns

1. **Protocol-based interfaces** - Use `typing.Protocol` for abstractions
2. **Pydantic models** - All types are Pydantic BaseModel subclasses
3. **Async everywhere** - All I/O operations use async/await
4. **Provider-agnostic** - LLM layer abstracts provider differences

## Source Files

Types are extracted from Episteme's `api/src/noema_schemas/`.
LLM layer is extracted from Episteme's `backend/src/backend/llm/`.

## Git Commit Conventions

Always prefix commit messages with an emoji:
- ✨ New features
- 🐛 Bug fixes
- ♻️ Refactoring
- 📝 Documentation
- 🧪 Tests

## Simply Workflow

- After completing a task, always run `/simply:commit` to create atomic commits
- Don't ask for permission to continue - proceed to the next task automatically
- Use `source .venv/bin/activate && python` instead of `uv run python`
