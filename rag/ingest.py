"""Document ingestion pipeline for ChromaDB-backed RAG."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import chromadb

from config.settings import get_settings
from rag.embeddings import get_embedding_model
from utils.logger import setup_logger

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
ALLOWED_SUFFIXES = {".txt", ".md"}


def _iter_source_files(base_path: Path) -> Iterable[Path]:
    """Yield text source files recursively under a base path."""
    if not base_path.exists() or not base_path.is_dir():
        return

    for file_path in base_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_SUFFIXES:
            yield file_path


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using fixed character windows."""
    if not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        chunk = text[cursor : cursor + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        cursor += step

    return chunks


def _build_chunk_id(source: str, chunk_index: int, chunk_text: str) -> str:
    """Create deterministic chunk ids to support safe re-ingestion."""
    digest = hashlib.sha1(f"{source}|{chunk_index}|{chunk_text}".encode("utf-8")).hexdigest()
    return f"chunk_{digest}"


def ingest_documents() -> None:
    """Loads, splits, embeds, and stores documents in ChromaDB."""
    settings = get_settings()
    logger = setup_logger(level=settings.log_level)

    docs_dir = Path(settings.data_docs_path)
    errors_dir = Path(settings.data_errors_path)
    source_roots: list[tuple[str, Path]] = [("docs", docs_dir), ("errors", errors_dir)]

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict[str, str]] = []
    document_count = 0

    for source_type, root in source_roots:
        if not root.exists():
            logger.warning("RAG source folder missing: %s", root)
            continue

        for file_path in _iter_source_files(root):
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw_text:
                continue

            document_count += 1
            chunks = _split_text(raw_text)
            for index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": file_path.name, "type": source_type})
                all_ids.append(_build_chunk_id(str(file_path), index, chunk))

    logger.info("RAG ingestion discovered %s documents.", document_count)
    logger.info("RAG ingestion created %s chunks.", len(all_chunks))

    if not all_chunks:
        logger.warning("No chunks available for ingestion; skipping Chroma upsert.")
        return

    embedder = get_embedding_model(settings.embedding_model_name)
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True).tolist()

    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(name=settings.chroma_collection_name)
    collection.upsert(ids=all_ids, documents=all_chunks, metadatas=all_metadatas, embeddings=embeddings)

    logger.info("RAG ingestion completed successfully.")
