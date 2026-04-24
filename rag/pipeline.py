"""RAG pipeline orchestrating retrieval and Ollama generation."""

from __future__ import annotations

from config.settings import get_settings
from llm.ollama_client import OllamaClient
from prompts.devmentor_prompt import build_user_message, get_system_prompt
from rag.retriever import retrieve_context
from utils.logger import setup_logger


def run_rag(query: str) -> str:
    """Run retrieval + generation and return the final LLM response.

    Flow:
      1. Retrieve top-k relevant chunks from ChromaDB.
      2. Build a structured user message (RAG context + code query).
      3. Send system prompt + user message to Ollama via /api/chat.

    Args:
        query: The code snippet or error message submitted by the user.

    Returns:
        Structured DevMentor response with Bug ID, Root Cause, and Fix.
    """
    settings = get_settings()
    logger = setup_logger(level=settings.log_level)

    if not query.strip():
        return "Please provide a non-empty query."

    contexts = retrieve_context(query=query, k=settings.top_k)

    system = get_system_prompt()
    user = build_user_message(query=query, contexts=contexts)

    try:
        llm_client = OllamaClient(settings=settings)
        return llm_client.generate_chat_response(system=system, user=user)
    except Exception as exc:
        logger.warning("RAG pipeline generation failed: %s", exc)
        return "Unable to generate a response right now. Please verify Ollama is running."
