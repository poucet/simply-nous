"""Interactive demo CLI using Engine + MemoryConversationView.

Usage:
    uv run python -m nous.demo
    uv run python -m nous.demo --model llama3.2
"""

import argparse
import asyncio
import sys

from nous.engine import Engine
from nous.llm.providers import OllamaProvider
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
    print("Type 'quit' or Ctrl+C to exit.\n")

    client = provider.model(model_id)
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
    parser = argparse.ArgumentParser(description="Interactive chat demo (Ollama)")
    parser.add_argument(
        "--model", "-m",
        help="Model ID (e.g., llama3.2, mistral, codellama)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model))


if __name__ == "__main__":
    cli()
