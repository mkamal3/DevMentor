# DevMentor

DevMentor is a privacy-first, locally running AI assistant for code review and debugging, built with Ollama + RAG.

## Features

- Config system via environment variables (`config/settings.py`)
- Ollama client wrapper with both generate/chat methods (`llm/ollama_client.py`)
- RAG pipeline:
  - document ingestion + chunking (`rag/ingest.py`)
  - similarity retrieval (`rag/retriever.py`)
  - retrieval + generation orchestration (`rag/pipeline.py`)
- Structured prompt layer (`prompts/devmentor_prompt.py`)
- Gradio app with three operation modes (`ui/gradio_app.py`)
- FastAPI backend scaffold (`api/main.py`)
- LoRA fine-tuning scripts (`finetune/`)
- Benchmark utility (`utils/benchmark.py`)

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

## Run the CLI app

```bash
python app.py
```

The CLI app will:
- warm the embedding model cache
- ingest docs from `data/docs` and `data/errors`
- run a sample RAG query through Ollama
- log warnings instead of crashing if Ollama is unavailable

## Run the Gradio UI

```bash
python ui/gradio_app.py
```

Optional flags:

```bash
python ui/gradio_app.py --port 7860
python ui/gradio_app.py --share
```

## Run FastAPI backend (optional)

```bash
uvicorn api.main:app --reload
```

Health endpoint:
- `GET /health`

## Team workflow

| Member | Machine | Setup notes |
|--------|---------|-------------|
| Kamal  | RTX 5070 Ti (12 GB VRAM) | Default `.env` — runs `qwen2.5-coder:7b` with full GPU inference |
| Javed  | No dedicated GPU | See **Dev without a GPU** below |
| Bhanu  | Colab / GPU machine | Run `pip install -r requirements-finetune.txt`; use `finetune/` scripts |
| Ketu   | Any | Build Gradio UI in `ui/gradio_app.py` |

### Dev without a GPU (Javed)

**Option A — Use the smaller model locally (recommended for prompt iteration):**

In your local `.env` (never committed), set:
```
OLLAMA_MODEL=qwen2.5-coder:1.5b
```
Pull it once:
```bash
ollama pull qwen2.5-coder:1.5b
```
This runs on CPU in a few seconds — sufficient for developing and testing prompt templates.

**Option B — Point to Kamal's machine over the network:**

On Kamal's machine, start Ollama with network access:
```powershell
$env:OLLAMA_HOST = "0.0.0.0"; ollama serve
```
In your local `.env`, replace the base URL:
```
OLLAMA_BASE_URL=http://<kamals-local-ip>:11434
OLLAMA_MODEL=qwen2.5-coder:7b
```
No code changes required — just a config switch.

## Fine-tuning workflow (optional)

Install additional dependencies:

```bash
pip install -r requirements-finetune.txt
```

Then use:
- `finetune/prepare_dataset.py`
- `finetune/train_lora.py`

with config from `finetune/configs/lora_config.yaml`.

## Notes for contributors

- `models/` is reserved for LoRA adapter artifacts
- Never commit `.env` — use `.env.example` as the template
