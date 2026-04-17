"""DevMentor entrypoint for local CLI-style usage."""

from pathlib import Path

from config.settings import get_settings
from llm.ollama_client import OllamaClient
from rag.embeddings import get_embedding_model
from rag.ingest import ingest_documents
from rag.pipeline import RAGPipeline
from utils.logger import setup_logger


def _embedding_cache_path(model_name: str) -> Path:
    """Build expected Hugging Face cache directory for a model."""
    canonical_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    safe_name = canonical_name.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe_name}"


def _is_embedding_cached(model_name: str) -> bool:
    """Check whether an embedding model already exists in local HF cache."""
    model_dir = _embedding_cache_path(model_name=model_name)
    snapshots_dir = model_dir / "snapshots"
    return model_dir.exists() and snapshots_dir.exists() and any(snapshots_dir.iterdir())


def _warm_embedding_model(model_name: str) -> None:
    """Warm embedding model once so first retrieval call is faster."""
    get_embedding_model(model_name)


def main() -> None:
    """Initialize services and run a demo query."""
    settings = get_settings()
    logger = setup_logger(level=settings.log_level)
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)

    try:
        cached = _is_embedding_cached(model_name=settings.embedding_model_name)
        if cached:
            logger.info("Embedding model cache detected for '%s'.", settings.embedding_model_name)
        else:
            logger.info("Embedding model cache missing for '%s'. Warming now...", settings.embedding_model_name)
        _warm_embedding_model(model_name=settings.embedding_model_name)
        logger.info("Embedding model warmup complete.")
    except Exception as exc:  # pragma: no cover - optional warmup
        logger.warning("Embedding model warmup skipped due to error: %s", exc)

    try:
        ingested = ingest_documents(settings=settings)
        logger.info("Document ingestion completed. Documents processed: %s", ingested)
    except Exception as exc:  # pragma: no cover - startup resilience
        logger.warning("Ingestion skipped due to error: %s", exc)

    try:
        llm_client = OllamaClient(settings=settings)
        rag_pipeline = RAGPipeline(settings=settings, llm_client=llm_client)
        demo_query = "How can I debug a Python AttributeError quickly?"
        response = rag_pipeline.run(query=demo_query)
        logger.info("Sample response: %s", response[:400] if response else "(empty response)")
    except Exception as exc:  # pragma: no cover - Ollama may not be running
        logger.warning("LLM request skipped or failed: %s", exc)
        logger.info("Tip: Start Ollama and pull model '%s' to enable responses.", settings.ollama_model)


if __name__ == "__main__":
    main()
