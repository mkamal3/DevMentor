"""Ollama client wrapper for generating responses."""

import re
from typing import Any, Dict, List

import requests

from config.settings import Settings


class OllamaClient:
    """Reusable wrapper around Ollama's generate and chat endpoints."""

    def __init__(self, settings: Settings, timeout_seconds: int = 300) -> None:
        """Initialize client with app settings."""
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def _clean_response_text(text: str) -> str:
        """Remove leaked chat control tokens from model output."""
        cleaned = text
        noise_tokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "[response]",
        ]
        for token in noise_tokens:
            cleaned = cleaned.replace(token, "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def generate_response(self, prompt: str) -> str:
        """Generate a response via the /api/generate endpoint (single prompt string).

        Args:
            prompt: Full prompt string including any context and instructions.

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
        return self._clean_response_text(str(data.get("response", "")))

    def generate_chat_response(self, system: str, user: str) -> str:
        """Generate a response via the /api/chat endpoint with role separation.

        Keeps the system prompt (persona + format rules) cleanly separated from
        the user message (code to review + RAG context), which produces more
        consistent structured output than a single flat prompt string.

        Args:
            system: The system prompt defining DevMentor's role and output format.
            user: The user-turn message containing the code/error and RAG context.

        Returns:
            Model-generated response text.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        response = requests.post(
            url=f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = str(data.get("message", {}).get("content", ""))
        return self._clean_response_text(content)
