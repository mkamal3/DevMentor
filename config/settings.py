"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration for DevMentor."""

    app_name: str = "DevMentor"
    environment: str = "development"
    log_level: str = "INFO"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    chroma_collection_name: str = "devmentor_docs"
    chroma_path: str = "chroma_db"

    data_docs_path: str = "data/docs"
    data_errors_path: str = "data/errors"
    top_k: int = Field(default=3, ge=1, le=20)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        """Return project root from current working directory."""
        return Path.cwd()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
