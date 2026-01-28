"""Model capabilities and registry for LLM providers."""

from dataclasses import dataclass, field


@dataclass
class ModelCapabilities:
    """What a model can do."""

    vision: bool = False
    audio_input: bool = False
    audio_output: bool = False
    image_output: bool = False
    tools: bool = False
    streaming: bool = True
    max_tokens: int | None = None
    context_window: int | None = None


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


@dataclass
class ModelRegistry:
    """Registry of known models and their capabilities."""

    _models: dict[str, ModelCapabilities] = field(default_factory=dict)

    def register(self, model_id: str, capabilities: ModelCapabilities) -> None:
        """Register a model with its capabilities."""
        self._models[model_id] = capabilities

    def get(self, model_id: str) -> ModelCapabilities:
        """Get capabilities for a model. Returns defaults if unknown."""
        return self._models.get(model_id, ModelCapabilities())

    def supports_vision(self, model_id: str) -> bool:
        """Check if a model supports vision/image input."""
        return self.get(model_id).vision

    def supports_tools(self, model_id: str) -> bool:
        """Check if a model supports tool use."""
        return self.get(model_id).tools

    def filter(self, **requirements: bool) -> list[str]:
        """Filter models by capability requirements.

        Args:
            **requirements: Capability name/value pairs to match.
                Example: filter(vision=True, tools=True)

        Returns:
            List of model IDs matching all requirements.
        """
        result = []
        for model_id, caps in self._models.items():
            if all(getattr(caps, k, None) == v for k, v in requirements.items()):
                result.append(model_id)
        return result
