"""Ollama client wrapper for generating responses."""

from typing import Any, Dict

import requests

from config.settings import Settings


class OllamaClient:
    """Simple reusable wrapper around Ollama's generate endpoint."""

    def __init__(self, settings: Settings, timeout_seconds: int = 300) -> None:
        """Initialize client with app settings."""
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout_seconds = timeout_seconds

    def generate_response(self, prompt: str) -> str:
        """Generate a text response using the configured Ollama model.

        Args:
            prompt: Input prompt string.

        Returns:
            Model-generated response text.
        """
        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(
            url=f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()
