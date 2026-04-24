"""DevMentor Gradio UI — wired to the local Ollama + RAG pipeline.

Runs entirely offline. No HuggingFace Transformers required — inference
goes through Ollama (qwen2.5-coder:7b by default) and retrieval goes through
the ChromaDB pipeline Kamal built.

Three modes match the project proposal ablation study:
  Mode A — Prompt Engineering only  (no RAG context injected)
  Mode B — Base + RAG               (ChromaDB context injected)
  Mode C — Base + RAG + LoRA        (LoRA model via Ollama, same RAG)

Usage
-----
    python ui/gradio_app.py
    python ui/gradio_app.py --share      # public Gradio link (72 h)
    python ui/gradio_app.py --port 7861
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from config.settings import get_settings  # noqa: E402
from llm.ollama_client import OllamaClient  # noqa: E402
from prompts.devmentor_prompt import build_user_message, get_system_prompt  # noqa: E402
from rag.retriever import retrieve_context  # noqa: E402

# ---------------------------------------------------------------------------
# CSS — dark theme matching Ketu's Colab design
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'Syne', sans-serif;
}
#dm-title { text-align: center; padding: 1.5rem 0 0.5rem; }
#dm-title h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
#dm-title p { color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin: 4px 0 0; }
.label-text { font-family: 'Syne', sans-serif !important; color: #94a3b8 !important; font-weight: 600 !important; }
textarea, input {
    background: #0a0c10 !important;
    border: 1px solid #1e2435 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}
.output-box {
    background: #141720 !important;
    border: 1px solid #1e2435 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
button.primary {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
"""

# ---------------------------------------------------------------------------
# Example bug cases (shown at bottom of UI for quick testing)
# ---------------------------------------------------------------------------
_EXAMPLES = [
    ["def divide(a, b):\n    return a / b", "Mode A — Prompt Engineering Only"],
    ["def get_first(lst):\n    return lst[0]", "Mode B — Base + RAG"],
    ["def factorial(n):\n    return n * factorial(n-1)", "Mode C — Base + RAG + LoRA"],
    ["data = None\nprint(data['key'])", "Mode B — Base + RAG"],
    ["for i in range(len(items)+1):\n    print(items[i])", "Mode A — Prompt Engineering Only"],
    [
        "import threading\ncounter = 0\ndef inc():\n    global counter\n"
        "    for _ in range(100000): counter += 1\n"
        "t1=threading.Thread(target=inc); t2=threading.Thread(target=inc)\n"
        "t1.start(); t2.start(); t1.join(); t2.join()\nprint(counter)",
        "Mode B — Base + RAG",
    ],
]

# ---------------------------------------------------------------------------
# Core inference handler
# ---------------------------------------------------------------------------

def analyze(code: str, mode: str) -> tuple[str, str]:
    """Run DevMentor analysis and return (response, metrics) strings."""
    if not code.strip():
        return "Please paste some code first.", ""

    settings = get_settings()

    # Determine model: Mode C uses the LoRA adapter if configured
    import os
    if mode == "Mode C — Base + RAG + LoRA":
        model_name = os.environ.get("LORA_MODEL", settings.ollama_model)
    else:
        model_name = settings.ollama_model

    # Retrieve RAG context for Modes B and C
    contexts: list[dict] = []
    if mode != "Mode A — Prompt Engineering Only":
        try:
            contexts = retrieve_context(query=code, k=settings.top_k)
        except Exception:
            contexts = []

    system = get_system_prompt()
    user = build_user_message(query=code, contexts=contexts)

    # Swap model on the client for Mode C
    from config.settings import Settings
    patched = Settings(
        ollama_model=model_name,
        ollama_base_url=settings.ollama_base_url,
        embedding_model_name=settings.embedding_model_name,
        chroma_collection_name=settings.chroma_collection_name,
        chroma_path=settings.chroma_path,
        data_docs_path=settings.data_docs_path,
        data_errors_path=settings.data_errors_path,
        top_k=settings.top_k,
    )

    t0 = time.perf_counter()
    try:
        client = OllamaClient(settings=patched)
        response = client.generate_chat_response(system=system, user=user)
    except Exception as exc:
        return f"Error: {exc}\n\nMake sure Ollama is running: ollama serve", ""
    latency = time.perf_counter() - t0

    rag_note = f"{len(contexts)} chunks" if contexts else "none"
    metrics = (
        f"Latency: {latency:.2f}s  |  "
        f"Model: {model_name}  |  "
        f"RAG chunks: {rag_note}  |  "
        f"Words: {len(response.split())}"
    )
    return response, metrics


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def launch_ui(share: bool = False, port: int = 7860) -> None:
    """Build and launch the DevMentor Gradio interface."""
    with gr.Blocks(css=_CSS, title="DevMentor") as demo:

        gr.HTML("""
        <div id="dm-title">
            <h1>DevMentor</h1>
            <p>// privacy-first local code review &amp; debugging assistant · CSIT595 · Montclair State</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=[
                        "Mode A — Prompt Engineering Only",
                        "Mode B — Base + RAG",
                        "Mode C — Base + RAG + LoRA",
                    ],
                    value="Mode A — Prompt Engineering Only",
                    label="Select Mode",
                )
                code_input = gr.Textbox(
                    lines=18,
                    placeholder=(
                        "# Paste your Python / JS / Java code here...\n\n"
                        "def get_item(lst, i):\n"
                        "    return lst[i]  # try: get_item([], 0)"
                    ),
                    label="Code Input",
                )
                analyze_btn = gr.Button("Analyze with DevMentor", variant="primary")

            with gr.Column(scale=1):
                metrics_out = gr.Textbox(
                    label="Metrics",
                    interactive=False,
                    lines=1,
                    elem_classes=["output-box"],
                )
                response_out = gr.Textbox(
                    lines=22,
                    label="DevMentor Analysis",
                    interactive=False,
                    elem_classes=["output-box"],
                )

        analyze_btn.click(
            fn=analyze,
            inputs=[code_input, mode],
            outputs=[response_out, metrics_out],
        )

        gr.Examples(
            examples=_EXAMPLES,
            inputs=[code_input, mode],
            label="Example Bug Cases (click to load)",
        )

        gr.Markdown(
            "**Running locally via Ollama** — no code leaves your machine. "
            "Set `LORA_MODEL=<model-name>` env var to enable Mode C with Bhanu's LoRA adapter.",
            elem_classes=["output-box"],
        )

    demo.launch(share=share, server_port=port, debug=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DevMentor Gradio UI")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Local port (default 7860)")
    args = parser.parse_args()
    launch_ui(share=args.share, port=args.port)
