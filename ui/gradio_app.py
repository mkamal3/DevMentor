"""DevMentor Gradio UI — wired to the local Ollama + RAG pipeline."""

from __future__ import annotations

import argparse
import html
import os
import re
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
# Section heading colours (applied inside the HTML analysis output)
# ---------------------------------------------------------------------------
_SECTION_COLORS = {
    "1": ("#f87171", "#2a0f0f"),   # coral-red  — Bug Identification
    "2": ("#fbbf24", "#1f1700"),   # amber      — Root Cause Explanation
    "3": ("#34d399", "#091f16"),   # green      — Suggested Fix
    "4": ("#38bdf8", "#071624"),   # sky-blue   — Commentary
}

# ---------------------------------------------------------------------------
# Markdown → HTML converter
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    """Convert a DevMentor markdown response to styled HTML."""
    if not text:
        return ""

    # Split out fenced code blocks so we don't escape their content twice
    parts = re.split(r"(```[^\n]*\n[\s\S]*?```)", text)
    rendered: list[str] = []

    for part in parts:
        if part.startswith("```"):
            lines = part.split("\n")
            code_body = "\n".join(lines[1:]).rstrip("`").rstrip()
            rendered.append(
                '<pre style="background:#0a0c10;border:1px solid #1e2435;'
                'border-radius:8px;padding:12px;overflow-x:auto;margin:10px 0;">'
                '<code style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:0.88em;color:#e2e8f0;">{html.escape(code_body)}</code></pre>'
            )
        else:
            escaped = html.escape(part)
            # Colour each ## N. heading
            for num, (fg, bg) in _SECTION_COLORS.items():
                escaped = re.sub(
                    rf"## {num}\. ([^\n]+)",
                    (
                        f'<h3 style="color:{fg};background:{bg};font-family:Syne,sans-serif;'
                        f'font-size:1.05rem;font-weight:700;margin:18px 0 6px;'
                        f'border-left:4px solid {fg};padding:6px 10px;border-radius:4px;">'
                        f"## {num}. \\1</h3>"
                    ),
                    escaped,
                )
            # Soft line breaks
            escaped = escaped.replace("\n", "<br>")
            rendered.append(escaped)

    return (
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.88rem;'
        'color:#e2e8f0;padding:6px 2px;line-height:1.6;">'
        + "".join(rendered)
        + "</div>"
    )

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'Syne', sans-serif;
}

/* Header */
#dm-title { text-align: center; padding: 1.2rem 0 0.8rem; }
#dm-title h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
#dm-title p { color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin: 4px 0 0; }

/* Tile: Mode selector — cyan */
#mode-tile {
    border-left: 4px solid #38bdf8 !important;
    background: #090f1a !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
/* Tile: Analyze button — purple */
#btn-tile {
    border-left: 4px solid #818cf8 !important;
    background: #0e0c1e !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    display: flex; align-items: center;
}
/* Tile: Code input — green */
#code-tile {
    border-left: 4px solid #34d399 !important;
    background: #080f0f !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
/* Tile: Metrics — amber */
#metrics-tile {
    border-left: 4px solid #f59e0b !important;
    background: #100d00 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    margin-bottom: 8px;
}
/* Tile: Analysis — pink */
#analysis-tile {
    border-left: 4px solid #f472b6 !important;
    background: #130a14 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}

/* Tile labels */
#mode-tile label.block,
#code-tile label.block,
#metrics-tile label.block {
    color: #94a3b8 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.tile-label {
    color: #94a3b8;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 6px;
}

/* Horizontal radio */
#mode-tile .wrap { display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 20px !important; }
#mode-tile label { white-space: nowrap !important; color: #e2e8f0 !important; }
#mode-tile input[type="radio"] { accent-color: #38bdf8; }

/* Text inputs — exclude radio/checkbox */
textarea,
input:not([type="radio"]):not([type="checkbox"]):not([type="button"]):not([type="submit"]) {
    background: #0a0c10 !important;
    border: 1px solid #1e2435 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}
#metrics-tile textarea {
    background: transparent !important;
    border: none !important;
    color: #d4a017 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Analyze button */
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
"""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
_EXAMPLES = [
    ["def divide(a, b):\n    return a / b",                "Mode A — Prompt Engineering Only"],
    ["def get_first(lst):\n    return lst[0]",             "Mode B — Base + RAG"],
    ["def factorial(n):\n    return n * factorial(n-1)",   "Mode C — Base + RAG + LoRA"],
    ["data = None\nprint(data['key'])",                    "Mode B — Base + RAG"],
    ["for i in range(len(items)+1):\n    print(items[i])", "Mode A — Prompt Engineering Only"],
]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def analyze(code: str, mode: str) -> tuple[str, str]:
    if not code.strip():
        return _md_to_html("Please paste some code first."), ""

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
        client = OllamaClient(settings=patched)
        response = client.generate_chat_response(
            system=get_system_prompt(),
            user=build_user_message(query=code, contexts=contexts),
        )
        # Some LoRA checkpoints can echo training templates instead of producing
        # final analysis. If detected, retry once with a stricter instruction.
        lower = response.lower()
        looks_like_template_echo = (
            "please provide a response in this format" in lower
            or ("bug:" in lower and "fix:" in lower and "## 1." not in response)
        )
        if looks_like_template_echo:
            retry_user = (
                f"{build_user_message(query=code, contexts=contexts)}\n\n"
                "Return ONLY the final answer. Do not repeat instructions, "
                "templates, or example format text."
            )
            response = client.generate_chat_response(
                system=get_system_prompt(),
                user=retry_user,
            )

    except Exception as exc:
        return _md_to_html(f"Error: {exc}\n\nMake sure Ollama is running: ollama serve"), ""
    latency = time.perf_counter() - t0

    rag_note = f"{len(contexts)} chunks" if contexts else "none"
    metrics = (
        f"Latency: {latency:.2f}s  |  Model: {model_name}  |  "
        f"RAG chunks: {rag_note}  |  Words: {len(response.split())}"
    )
    return _md_to_html(response), metrics


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def launch_ui(share: bool = False, port: int = 7860, open_browser: bool = True) -> None:
    with gr.Blocks(title="DevMentor") as demo:

        # Header
        gr.HTML("""
        <div id="dm-title">
            <h1>DevMentor</h1>
            <p>privacy-first local code review &amp; debugging assistant</p>
        </div>
        """)

        # Row 1: horizontal modes | analyze button
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
                )
            with gr.Column(scale=1, elem_id="btn-tile"):
                analyze_btn = gr.Button(
                    "Analyze with DevMentor",
                    variant="primary",
                    elem_id="analyze-btn",
                )

        # Row 2: code input | metrics + analysis
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
                )
            with gr.Column(scale=1):
                with gr.Group(elem_id="metrics-tile"):
                    metrics_out = gr.Textbox(
                        label="Metrics",
                        interactive=False,
                        lines=2,
                    )
                with gr.Group(elem_id="analysis-tile"):
                    gr.HTML('<div class="tile-label">DevMentor Analysis</div>')
                    response_out = gr.HTML(value="")

        # Wire
        analyze_btn.click(
            fn=analyze,
            inputs=[code_input, mode],
            outputs=[response_out, metrics_out],
        )

        # Examples
        gr.Examples(
            examples=_EXAMPLES,
            inputs=[code_input, mode],
            label="Example Bug Cases",
        )

    demo.launch(share=share, server_port=port, debug=False, css=_CSS, inbrowser=open_browser)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DevMentor Gradio UI")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Disable automatic browser opening on launch.",
    )
    args = parser.parse_args()
    launch_ui(share=args.share, port=args.port, open_browser=not args.no_open)
