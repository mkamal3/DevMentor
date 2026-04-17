"""High-level RAG pipeline orchestrating retrieval and generation."""

from llm.ollama_client import OllamaClient
from config.settings import Settings
from rag.retriever import retrieve_context


class RAGPipeline:
    """Combine retrieval with LLM generation for local assistant responses."""

    def __init__(self, settings: Settings, llm_client: OllamaClient) -> None:
        """Initialize pipeline dependencies."""
        self._settings = settings
        self._llm_client = llm_client

    def run(self, query: str) -> str:
        """Run retrieval-augmented generation for a query."""
        contexts = retrieve_context(query=query, settings=self._settings, k=self._settings.top_k)
        context_block = "\n\n".join(contexts) if contexts else "No relevant context found."

        prompt = (
            "You are DevMentor, a local code assistant.\n"
            "Use the context to answer the user's question clearly.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
        return self._llm_client.generate_response(prompt=prompt)
