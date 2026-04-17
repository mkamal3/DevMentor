# DevMentor

DevMentor is a privacy-first, locally running AI assistant scaffold for code review and debugging, built with Ollama + RAG.

## Features in this scaffold

- Config system via environment variables (`config/settings.py`)
- Local Ollama client wrapper (`llm/ollama_client.py`)
- RAG components:
  - ingestion (`rag/ingest.py`)
  - retrieval (`rag/retriever.py`)
  - pipeline orchestration (`rag/pipeline.py`)
- Logging utility (`utils/logger.py`)
- FastAPI starter (`api/main.py`)
- Prompt/UI placeholders for separate team implementation

## Project setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment file:

```bash
copy .env.example .env
```

## Run Ollama locally

1. Install Ollama from [https://ollama.com](https://ollama.com)
2. Pull the model:

```bash
ollama pull qwen2.5-coder:7b
```

3. Ensure Ollama is running at `http://localhost:11434`.

## Run the app

```bash
python app.py
```

The app will:
- attempt to ingest local documents from `data/docs` and `data/errors`
- attempt a sample RAG query through Ollama
- log warnings instead of crashing if Ollama is unavailable

## Run FastAPI scaffold (optional)

```bash
uvicorn api.main:app --reload
```

Health endpoint:
- `GET /health`

## Notes for contributors

- `prompts/devmentor_prompt.py` is placeholder-only
- `ui/gradio_app.py` is scaffold-only and intentionally raises `NotImplementedError`
- `models/` is reserved for future LoRA artifacts
