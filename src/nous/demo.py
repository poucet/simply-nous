"""Interactive demo CLI using Engine + MemoryConversationView.

Usage:
    uv run python -m nous.demo
    uv run python -m nous.demo --model gpt-4o
    uv run python -m nous.demo --model claude-sonnet-4-20250514
"""

import argparse
import asyncio
import sys

from nous.engine import Engine
from nous.llm.hub import create_default_hub
from nous.view.memory import MemoryConversationView


class DemoView(MemoryConversationView):
    """View that prints streaming text in real-time."""

    async def on_text_delta(self, text: str) -> None:
        """Print text as it streams."""
        print(text, end="", flush=True)
        await super().on_text_delta(text)

    async def on_turn_complete(self) -> None:
        """Print newline after turn completes."""
        print()
        await super().on_turn_complete()


async def main(model_id: str | None = None) -> None:
    """Run interactive chat loop."""
    hub = create_default_hub()

    if not hub.providers:
        print("No LLM providers available. Install an SDK:", file=sys.stderr)
        print("  pip install anthropic  # For Claude", file=sys.stderr)
        print("  pip install openai     # For OpenAI", file=sys.stderr)
        sys.exit(1)

    # Default to first available provider's common model
    if model_id is None:
        from nous.types import Provider

        if hub.is_registered(Provider.ANTHROPIC):
            model_id = "claude-sonnet-4-20250514"
        elif hub.is_registered(Provider.OPENAI):
            model_id = "gpt-4o"
        elif hub.is_registered(Provider.OLLAMA):
            model_id = "llama3.2"
        else:
            provider = hub.providers[0]
            print(f"Using first registered provider: {provider.value}")
            model_id = "default"

    print(f"Model: {model_id}")
    print("Type 'quit' or Ctrl+C to exit.\n")

    try:
        client = await hub.client_for_model(model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    engine = Engine()
    view = DemoView()

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
        view.clear_events()

        print("Assistant: ", end="", flush=True)
        try:
            await engine.run_turn(client, view)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


def cli() -> None:
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(description="Interactive chat demo")
    parser.add_argument(
        "--model", "-m",
        help="Model ID (e.g., claude-sonnet-4-20250514, gpt-4o, llama3.2)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model))


if __name__ == "__main__":
    cli()
