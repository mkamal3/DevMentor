"""DevMentor Gradio UI — wired to the local Ollama + RAG pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from config.settings import Settings, get_settings  # noqa: E402
from llm.ollama_client import OllamaClient  # noqa: E402
from prompts.devmentor_prompt import build_user_message, get_system_prompt  # noqa: E402
from rag.retriever import retrieve_context  # noqa: E402

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'Syne', sans-serif;
}

/* ── Header ────────────────────────────────────────────────────────────── */
#dm-title { text-align: center; padding: 1.2rem 0 0.8rem; }
#dm-title h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
#dm-title p {
    color: #64748b; font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; margin: 4px 0 0;
}

/* ── Tile colours ──────────────────────────────────────────────────────── */
/* Mode selector — cyan accent */
#mode-tile {
    border-left: 4px solid #38bdf8 !important;
    background: #090f1a !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
/* Analyze button tile — purple accent */
#btn-tile {
    border-left: 4px solid #818cf8 !important;
    background: #0e0c1e !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    display: flex; align-items: center;
}
/* Code input — green accent */
#code-tile {
    border-left: 4px solid #34d399 !important;
    background: #080f0f !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
/* Metrics — amber accent */
#metrics-tile {
    border-left: 4px solid #f59e0b !important;
    background: #100d00 !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
}
/* Analysis — pink accent */
#analysis-tile {
    border-left: 4px solid #f472b6 !important;
    background: #130a14 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}

/* ── Horizontal radio buttons ──────────────────────────────────────────── */
#mode-tile .wrap { display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 18px !important; }
#mode-tile label { white-space: nowrap !important; color: #e2e8f0 !important; }
#mode-tile input[type="radio"] { accent-color: #38bdf8; }

/* ── Text inputs / textarea ─────────────────────────────────────────────── */
textarea,
input:not([type="radio"]):not([type="checkbox"]):not([type="button"]):not([type="submit"]) {
    background: #0a0c10 !important;
    border: 1px solid #1e2435 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}

/* ── Analyze button ─────────────────────────────────────────────────────── */
#analyze-btn button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 14px !important;
}

/* ── Output boxes ────────────────────────────────────────────────────────── */
#metrics-tile textarea, #analysis-tile textarea {
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Examples table ─────────────────────────────────────────────────────── */
.examples-holder table { width: 100% !important; }
.examples-holder td { white-space: nowrap !important; }
"""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
_EXAMPLES = [
    ["def divide(a, b):\n    return a / b",                        "Mode A — Prompt Engineering Only"],
    ["def get_first(lst):\n    return lst[0]",                     "Mode B — Base + RAG"],
    ["def factorial(n):\n    return n * factorial(n-1)",           "Mode C — Base + RAG + LoRA"],
    ["data = None\nprint(data['key'])",                            "Mode B — Base + RAG"],
    ["for i in range(len(items)+1):\n    print(items[i])",         "Mode A — Prompt Engineering Only"],
    [
        "counter = 0\ndef inc():\n    global counter\n"
        "    for _ in range(100000): counter += 1\n"
        "import threading\nt1=threading.Thread(target=inc)\n"
        "t2=threading.Thread(target=inc)\n"
        "t1.start();t2.start();t1.join();t2.join()\nprint(counter)",
        "Mode B — Base + RAG",
    ],
]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def analyze(code: str, mode: str) -> tuple[str, str]:
    if not code.strip():
        return "Please paste some code first.", ""

    settings = get_settings()
    model_name = (
        os.environ.get("LORA_MODEL", settings.ollama_model)
        if mode == "Mode C — Base + RAG + LoRA"
        else settings.ollama_model
    )

    contexts: list[dict] = []
    if mode != "Mode A — Prompt Engineering Only":
        try:
            contexts = retrieve_context(query=code, k=settings.top_k)
        except Exception:
            contexts = []

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
        response = OllamaClient(settings=patched).generate_chat_response(
            system=get_system_prompt(),
            user=build_user_message(query=code, contexts=contexts),
        )
    except Exception as exc:
        return f"Error: {exc}\n\nMake sure Ollama is running: ollama serve", ""
    latency = time.perf_counter() - t0

    rag_note = f"{len(contexts)} chunks" if contexts else "none"
    metrics = (
        f"Latency: {latency:.2f}s  |  Model: {model_name}  |  "
        f"RAG chunks: {rag_note}  |  Words: {len(response.split())}"
    )
    return response, metrics


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def launch_ui(share: bool = False, port: int = 7860) -> None:
    with gr.Blocks(title="DevMentor") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="dm-title">
            <h1>DevMentor</h1>
            <p>privacy-first local code review &amp; debugging assistant</p>
        </div>
        """)

        # ── Row 1: horizontal mode selector | analyze button ─────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=4, elem_id="mode-tile"):
                mode = gr.Radio(
                    choices=[
                        "Mode A — Prompt Engineering Only",
                        "Mode B — Base + RAG",
                        "Mode C — Base + RAG + LoRA",
                    ],
                    value="Mode A — Prompt Engineering Only",
                    label="Select Mode",
                    container=False,
                )
            with gr.Column(scale=1, elem_id="btn-tile"):
                analyze_btn = gr.Button(
                    "Analyze with DevMentor",
                    variant="primary",
                    elem_id="analyze-btn",
                )

        # ── Row 2: code input | metrics + analysis ────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_id="code-tile"):
                code_input = gr.Textbox(
                    lines=22,
                    placeholder=(
                        "# Paste your Python / JS / Java code here...\n\n"
                        "def get_item(lst, i):\n"
                        "    return lst[i]  # try: get_item([], 0)"
                    ),
                    label="Code Input",
                    container=False,
                )
            with gr.Column(scale=1):
                with gr.Group(elem_id="metrics-tile"):
                    metrics_out = gr.Textbox(
                        label="Metrics",
                        interactive=False,
                        lines=2,
                        container=False,
                    )
                with gr.Group(elem_id="analysis-tile"):
                    response_out = gr.Textbox(
                        lines=19,
                        label="DevMentor Analysis",
                        interactive=False,
                        container=False,
                    )

        # ── Wire ────────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[code_input, mode],
            outputs=[response_out, metrics_out],
        )

        # ── Examples — single horizontal row ─────────────────────────────────
        gr.Examples(
            examples=_EXAMPLES,
            inputs=[code_input, mode],
            label="Example Bug Cases",
        )

    demo.launch(share=share, server_port=port, debug=False, css=_CSS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DevMentor Gradio UI")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_ui(share=args.share, port=args.port)
