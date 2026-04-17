"""Context retrieval helpers for ChromaDB-backed RAG."""

from __future__ import annotations

import chromadb

from config.settings import get_settings
from rag.embeddings import get_embedding_model
from utils.logger import setup_logger


def retrieve_context(query: str, k: int = 3) -> list[dict[str, str]]:
    """Returns top-k relevant chunks with content and metadata."""
    settings = get_settings()
    logger = setup_logger(level=settings.log_level)

    if not query.strip():
        logger.warning("Empty retrieval query provided.")
        return []

    safe_k = max(1, k)

    try:
        embedder = get_embedding_model(settings.embedding_model_name)
        query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

        client = chromadb.PersistentClient(path=settings.chroma_path)
        collection = client.get_or_create_collection(name=settings.chroma_collection_name)
        result = collection.query(query_embeddings=query_embedding, n_results=safe_k)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return []

    raw_documents = result.get("documents", [[]])
    raw_metadatas = result.get("metadatas", [[]])
    documents = raw_documents[0] if raw_documents else []
    metadatas = raw_metadatas[0] if raw_metadatas else []

    output: list[dict[str, str]] = []
    for idx, content in enumerate(documents):
        if not isinstance(content, str):
            continue
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        output.append(
            {
                "content": content,
                "source": str(metadata.get("source", "unknown")),
                "type": str(metadata.get("type", "unknown")),
            }
        )

    logger.info("RAG retrieval returned %s results.", len(output))
    return output
