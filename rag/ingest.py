"""Document ingestion utilities for ChromaDB."""

from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb

from config.settings import Settings
from rag.embeddings import get_embedding_model


def _collect_documents(paths: Iterable[Path]) -> List[Tuple[str, str]]:
    """Collect text documents from given directories.

    Returns:
        List of tuples as ``(source_path, content)``.
    """
    docs: List[Tuple[str, str]] = []
    allowed_suffixes = {".txt", ".md", ".py", ".json", ".log"}

    for directory in paths:
        if not directory.exists():
            continue
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in allowed_suffixes:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    docs.append((str(file_path), content))
    return docs


def _get_or_create_collection(settings: Settings):
    """Return persistent Chroma collection."""
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(name=settings.chroma_collection_name)


def ingest_documents(settings: Settings) -> int:
    """Ingest docs/errors directories into the Chroma collection.

    Args:
        settings: Runtime settings object.

    Returns:
        Number of documents ingested.
    """
    docs_dirs = [
        Path(settings.data_docs_path),
        Path(settings.data_errors_path),
    ]
    documents = _collect_documents(docs_dirs)
    if not documents:
        return 0

    embedder = get_embedding_model(settings.embedding_model_name)
    collection = _get_or_create_collection(settings)

    ids = [f"doc_{idx}" for idx, _ in enumerate(documents)]
    texts = [content for _, content in documents]
    metadatas = [{"source": source} for source, _ in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(documents)
