"""DevMentor hardware benchmarking and latency/memory profiling.

Measures response latency, token throughput, and memory usage across the
three model configurations defined in the project proposal:

  Config A — Base model + prompt engineering only (no RAG context)
  Config B — Base model + prompt engineering + RAG context injected
  Config C — Base model + prompt engineering + RAG + LoRA adapter
             (set LORA_MODEL env var or --lora-model flag when adapter is ready)

Designed to run on Kamal's GPU workstation for representative numbers.
Works on CPU-only machines too — VRAM columns will show N/A.

Usage
-----
    python utils/benchmark.py               # all configs, 3 runs each
    python utils/benchmark.py --config A    # single config
    python utils/benchmark.py --runs 5      # more runs per prompt
    python utils/benchmark.py --output results/bench_$(date +%F).json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import requests

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config/settings imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from prompts.devmentor_prompt import build_user_message, get_system_prompt  # noqa: E402

# ---------------------------------------------------------------------------
# Standardised test prompts (3 difficulty levels × varied bug types)
# ---------------------------------------------------------------------------
TEST_PROMPTS: list[dict[str, str]] = [
    # --- Syntax / NameError ---
    {
        "id": "syntax_01",
        "difficulty": "syntax",
        "code": (
            "def greet(name):\n"
            "    print('Hello, ' + nane)\n\n"
            "greet('Alice')"
        ),
    },
    {
        "id": "syntax_02",
        "difficulty": "syntax",
        "code": (
            "numbers = [1, 2, 3]\n"
            "for num in numbers\n"
            "    print(num)"
        ),
    },
    # --- Logic errors ---
    {
        "id": "logic_01",
        "difficulty": "logic",
        "code": (
            "def is_palindrome(s):\n"
            "    return s == s[1:-1]\n\n"
            "print(is_palindrome('racecar'))  # Expected True, got False"
        ),
    },
    {
        "id": "logic_02",
        "difficulty": "logic",
        "code": (
            "def factorial(n):\n"
            "    result = 0\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result\n\n"
            "print(factorial(5))  # Expected 120, got 0"
        ),
    },
    # --- Performance issues ---
    {
        "id": "perf_01",
        "difficulty": "performance",
        "code": (
            "def find_duplicates(lst):\n"
            "    duplicates = []\n"
            "    for i in range(len(lst)):\n"
            "        for j in range(len(lst)):\n"
            "            if i != j and lst[i] == lst[j]:\n"
            "                if lst[i] not in duplicates:\n"
            "                    duplicates.append(lst[i])\n"
            "    return duplicates"
        ),
    },
    {
        "id": "perf_02",
        "difficulty": "performance",
        "code": (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
            "print(fibonacci(40))  # Takes a very long time"
        ),
    },
]

# Minimal mock RAG context used for Config B when chroma_db is empty
_MOCK_RAG_CONTEXT: list[dict[str, str]] = [
    {
        "type": "doc",
        "source": "python-builtins",
        "content": (
            "range(stop) returns integers from 0 up to but not including stop. "
            "range(start, stop) returns integers from start up to but not including stop."
        ),
    },
    {
        "type": "error",
        "source": "common-errors",
        "content": (
            "IndexError: list index out of range — raised when you access a list "
            "index that does not exist. Check loop bounds and list length."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    prompt_id: str
    difficulty: str
    config: str
    latency_s: float
    prompt_tokens: int
    eval_tokens: int
    tokens_per_sec: float
    ram_delta_mb: float
    vram_before_mb: Optional[float]
    vram_after_mb: Optional[float]
    vram_delta_mb: Optional[float]
    response_preview: str


@dataclass
class BenchmarkReport:
    model: str
    lora_model: str
    date: str
    runs_per_prompt: int
    hardware: str
    results: list[RunResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _get_vram_mb() -> Optional[float]:
    """Query current GPU memory usage via nvidia-smi. Returns None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def _detect_hardware() -> str:
    """Return a human-readable hardware summary string."""
    cpu = f"{psutil.cpu_count(logical=False)}C/{psutil.cpu_count()}T"
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    vram = _get_vram_mb()
    gpu_str = f"{round(vram / 1024, 1)} GB VRAM detected" if vram is not None else "No GPU detected (CPU-only)"
    return f"CPU {cpu}, RAM {ram_gb} GB, {gpu_str}"


def _ram_used_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
) -> dict:
    """POST to /api/chat and return the full response JSON."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def _run_once(
    prompt_meta: dict,
    config: str,
    model: str,
    base_url: str,
    timeout: int,
    use_rag_context: bool,
    live_rag_contexts: list[dict],
) -> RunResult:
    code = prompt_meta["code"]
    system = get_system_prompt()

    if use_rag_context:
        contexts = live_rag_contexts if live_rag_contexts else _MOCK_RAG_CONTEXT
    else:
        contexts = []

    user = build_user_message(query=code, contexts=contexts)

    ram_before = _ram_used_mb()
    vram_before = _get_vram_mb()

    t0 = time.perf_counter()
    data = _call_ollama_chat(base_url, model, system, user, timeout)
    latency = time.perf_counter() - t0

    ram_after = _ram_used_mb()
    vram_after = _get_vram_mb()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    response_text = data.get("message", {}).get("content", "")

    vram_delta: Optional[float] = None
    if vram_before is not None and vram_after is not None:
        vram_delta = vram_after - vram_before

    return RunResult(
        prompt_id=prompt_meta["id"],
        difficulty=prompt_meta["difficulty"],
        config=config,
        latency_s=round(latency, 3),
        prompt_tokens=prompt_eval_count,
        eval_tokens=eval_count,
        tokens_per_sec=round(tokens_per_sec, 1),
        ram_delta_mb=round(ram_after - ram_before, 1),
        vram_before_mb=round(vram_before, 1) if vram_before is not None else None,
        vram_after_mb=round(vram_after, 1) if vram_after is not None else None,
        vram_delta_mb=round(vram_delta, 1) if vram_delta is not None else None,
        response_preview=response_text[:120].replace("\n", " "),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _print_report(report: BenchmarkReport) -> None:
    configs = ["A", "B", "C"]
    config_labels = {
        "A": "Config A — Base + Prompt Engineering",
        "B": "Config B — Base + Prompt Engineering + RAG",
        "C": "Config C — Base + Prompt Engineering + RAG + LoRA",
    }

    print("\n" + "=" * 72)
    print("  DevMentor Benchmark Results")
    print("=" * 72)
    print(f"  Date     : {report.date}")
    print(f"  Hardware : {report.hardware}")
    print(f"  Model    : {report.model}")
    print(f"  LoRA     : {report.lora_model}")
    print(f"  Runs/prompt: {report.runs_per_prompt}")
    print("=" * 72)

    for cfg in configs:
        cfg_results = [r for r in report.results if r.config == cfg]
        if not cfg_results:
            continue

        print(f"\n{config_labels[cfg]}")
        print(f"{'Prompt':<14} {'Diff':<12} {'Avg Lat':>9} {'Tok/s':>7} {'RAM Δ':>8} {'VRAM Δ':>9}")
        print("-" * 62)

        for pid in {r.prompt_id for r in cfg_results}:
            runs = [r for r in cfg_results if r.prompt_id == pid]
            diff = runs[0].difficulty
            avg_lat = _avg([r.latency_s for r in runs])
            avg_tps = _avg([r.tokens_per_sec for r in runs])
            avg_ram = _avg([r.ram_delta_mb for r in runs])
            vram_vals = [r.vram_delta_mb for r in runs if r.vram_delta_mb is not None]
            vram_str = f"{_avg(vram_vals):+.1f} MB" if vram_vals else "N/A"
            print(
                f"{pid:<14} {diff:<12} {avg_lat:>8.2f}s {avg_tps:>7.1f} "
                f"{avg_ram:>+7.1f} MB {vram_str:>9}"
            )

        # Config-level averages
        all_lat = [r.latency_s for r in cfg_results]
        all_tps = [r.tokens_per_sec for r in cfg_results]
        vram_all = [r.vram_delta_mb for r in cfg_results if r.vram_delta_mb is not None]
        print("-" * 62)
        print(
            f"{'AVERAGE':<14} {'':<12} {_avg(all_lat):>8.2f}s {_avg(all_tps):>7.1f} "
            f"{'':>9} {'N/A' if not vram_all else f'{_avg(vram_all):+.1f} MB':>9}"
        )

    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DevMentor benchmarking tool")
    parser.add_argument(
        "--config", choices=["A", "B", "C", "all"], default="all",
        help="Which configuration to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per prompt (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save full results as JSON",
    )
    parser.add_argument(
        "--lora-model", type=str, default=None,
        help="Ollama model name for Config C (LoRA adapter). "
             "Falls back to LORA_MODEL env var, then base model.",
    )
    args = parser.parse_args()

    settings = get_settings()
    base_model = settings.ollama_model
    base_url = settings.ollama_base_url

    lora_model = (
        args.lora_model
        or os.environ.get("LORA_MODEL")
        or base_model  # placeholder until Bhanu's adapter is ready
    )

    configs_to_run = ["A", "B", "C"] if args.config == "all" else [args.config]
    config_settings = {
        "A": {"model": base_model, "use_rag": False},
        "B": {"model": base_model, "use_rag": True},
        "C": {"model": lora_model, "use_rag": True},
    }

    # Try to get live RAG context from ChromaDB for Config B/C
    live_contexts: list[dict] = []
    try:
        from rag.retriever import retrieve_context
        live_contexts = retrieve_context(query="common Python bugs", k=3)
    except Exception:
        pass  # falls back to _MOCK_RAG_CONTEXT in _run_once

    report = BenchmarkReport(
        model=base_model,
        lora_model=lora_model if lora_model != base_model else "(not set — using base model)",
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        runs_per_prompt=args.runs,
        hardware=_detect_hardware(),
    )

    total = len(configs_to_run) * len(TEST_PROMPTS) * args.runs
    done = 0

    for cfg in configs_to_run:
        model = config_settings[cfg]["model"]
        use_rag = config_settings[cfg]["use_rag"]
        print(f"\nRunning Config {cfg} with model '{model}' ...")

        for prompt_meta in TEST_PROMPTS:
            for run_i in range(args.runs):
                done += 1
                print(
                    f"  [{done}/{total}] {prompt_meta['id']} run {run_i + 1}/{args.runs}",
                    end=" ... ",
                    flush=True,
                )
                try:
                    result = _run_once(
                        prompt_meta=prompt_meta,
                        config=cfg,
                        model=model,
                        base_url=base_url,
                        timeout=settings.top_k * 100 + 300,  # generous ceiling
                        use_rag_context=use_rag,
                        live_rag_contexts=live_contexts,
                    )
                    report.results.append(result)
                    print(f"{result.latency_s:.2f}s  {result.tokens_per_sec:.1f} tok/s")
                except Exception as exc:
                    print(f"FAILED: {exc}")

    _print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {**asdict(report), "results": [asdict(r) for r in report.results]},
                f,
                indent=2,
            )
        print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
