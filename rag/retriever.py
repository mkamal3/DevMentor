"""Vector retrieval utilities for DevMentor RAG."""

from typing import List

import chromadb

from config.settings import Settings
from rag.embeddings import get_embedding_model


def retrieve_context(query: str, settings: Settings, k: int = 3) -> List[str]:
    """Retrieve top-k relevant context snippets from ChromaDB.

    Args:
        query: User query text.
        settings: Runtime settings object.
        k: Number of top results to return.

    Returns:
        List of retrieved context strings.
    """
    embedder = get_embedding_model(settings.embedding_model_name)
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(name=settings.chroma_collection_name)
    result = collection.query(query_embeddings=query_embedding, n_results=k)

    documents = result.get("documents", [])
    if not documents:
        return []
    return [doc for doc in documents[0] if isinstance(doc, str)]
