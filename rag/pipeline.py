"""RAG pipeline orchestrating retrieval and Ollama generation."""

from __future__ import annotations

from config.settings import get_settings
from llm.ollama_client import OllamaClient
from rag.retriever import retrieve_context
from utils.logger import setup_logger


def _build_prompt(query: str, contexts: list[dict[str, str]]) -> str:
    """Build a simple context-injected prompt for the LLM."""
    if contexts:
        context_lines = [
            f"- [{item.get('type', 'unknown')}] {item.get('source', 'unknown')}: {item.get('content', '')}"
            for item in contexts
        ]
        context_block = "\n".join(context_lines)
    else:
        context_block = "No relevant context found."

    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n"
    )


def run_rag(query: str) -> str:
    """Run retrieval + generation and return the final LLM response."""
    settings = get_settings()
    logger = setup_logger(level=settings.log_level)

    if not query.strip():
        return "Please provide a non-empty query."

    contexts = retrieve_context(query=query, k=settings.top_k)
    prompt = _build_prompt(query=query, contexts=contexts)

    try:
        llm_client = OllamaClient(settings=settings)
        return llm_client.generate_response(prompt=prompt)
    except Exception as exc:
        logger.warning("RAG pipeline generation failed: %s", exc)
        return "Unable to generate a response right now. Please verify Ollama is running."
