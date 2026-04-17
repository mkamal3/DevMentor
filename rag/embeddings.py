"""Shared embedding model helpers for RAG modules."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for the process."""
    return SentenceTransformer(model_name)
