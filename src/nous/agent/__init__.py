"""Agent module - agentic layer with RAG integration.

The agent layer handles dynamic system prompt construction with knowledge
retrieval. It wraps ConversationView to intercept get_messages() and
inject relevant context.

Example:
    >>> from nous.agent import AgentRunner, KnowledgeFetcher
    >>> from nous.engine import Engine, ContextBuilder
    >>>
    >>> class MyKnowledgeFetcher:
    ...     async def fetch(self, query: str) -> list[KnowledgeChunk]:
    ...         # Your RAG implementation
    ...         return await my_vector_db.search(query)
    >>>
    >>> runner = AgentRunner(
    ...     view=my_view,
    ...     knowledge_fetcher=MyKnowledgeFetcher(),
    ...     base_prompt="You are a helpful assistant.",
    ... )
    >>> # Pass runner to engine - it implements ConversationView
    >>> response = await engine.run_turn(client, runner)
"""

from nous.agent.runner import AgentRunner, KnowledgeFetcher

__all__ = [
    "AgentRunner",
    "KnowledgeFetcher",
]
