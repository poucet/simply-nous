"""Interactive demo CLI using Engine + MemoryConversationView.

Usage:
    uv run python -m nous.demo
    uv run python -m nous.demo --model llama3.2
    uv run python -m nous.demo --mcp-server http://localhost:8080/mcp
"""

import argparse
import asyncio
import sys

from nous.engine import Engine
from nous.llm.providers import OllamaProvider
from nous.types import ToolCall, ToolResult, ToolDefinition
from nous.view.memory import MemoryConversationView


class DemoView(MemoryConversationView):
    """View that prints streaming text and tool calls in real-time."""

    async def on_text_delta(self, text: str) -> None:
        """Print text as it streams."""
        print(text, end="", flush=True)
        await super().on_text_delta(text)

    async def on_turn_complete(self) -> None:
        """Print newline after turn completes."""
        print()
        await super().on_turn_complete()

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Print tool call before executing."""
        print(f"\n[Tool: {tool_call.name}({tool_call.input})]", flush=True)
        result = await super().call_tool(tool_call)
        # Show result preview
        if result.content:
            preview = str(result.content[0])[:100]
            print(f"[Result: {preview}...]" if len(str(result.content[0])) > 100 else f"[Result: {preview}]")
        return result


async def main(model_id: str | None = None, mcp_server: str | None = None) -> None:
    """Run interactive chat loop."""
    provider = OllamaProvider()

    try:
        models = await provider.list_models()
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}", file=sys.stderr)
        print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)

    if not models:
        print("No models available. Pull a model first:", file=sys.stderr)
        print("  ollama pull llama3.2", file=sys.stderr)
        sys.exit(1)

    if model_id is None:
        model_id = models[0]
        print(f"Available models: {', '.join(models)}")

    if model_id not in models:
        print(f"Model '{model_id}' not found.", file=sys.stderr)
        print(f"Available: {', '.join(models)}", file=sys.stderr)
        sys.exit(1)

    print(f"Using: {model_id}")

    # Set up MCP if requested
    tools: list[ToolDefinition] = []
    tool_handler = None

    if mcp_server:
        try:
            from nous.mcp import MCPClient, MCPServerConfig, ToolExecutor
        except ImportError:
            print("MCP support requires: pip install simply-nous[mcp]", file=sys.stderr)
            sys.exit(1)

        print(f"Connecting to MCP server: {mcp_server}")
        mcp_client = MCPClient()
        try:
            await mcp_client.connect(MCPServerConfig(name="tools", url=mcp_server))
            tools = await mcp_client.list_tools()
            executor = ToolExecutor(mcp_client)
            tool_handler = executor.execute
            print(f"Tools available: {', '.join(t.name for t in tools)}")
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}", file=sys.stderr)
            sys.exit(1)

    print("Type 'quit' or Ctrl+C to exit.\n")

    client = provider.model(model_id)
    engine = Engine()
    view = DemoView(tool_handler=tool_handler)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            view.add_user_message(user_input)
            print("Assistant: ", end="", flush=True)
            try:
                await engine.run_turn(client, view, tools=tools if tools else None)
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
    finally:
        if mcp_server and 'mcp_client' in dir():
            await mcp_client.disconnect_all()


def cli() -> None:
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(description="Interactive chat demo (Ollama)")
    parser.add_argument(
        "--model", "-m",
        help="Model ID (e.g., llama3.2, mistral, codellama)",
    )
    parser.add_argument(
        "--mcp-server",
        help="MCP server URL (e.g., http://localhost:8080/mcp)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model, args.mcp_server))


if __name__ == "__main__":
    cli()
