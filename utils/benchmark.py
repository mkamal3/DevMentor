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
# Standardised test prompts — Ketu's 30-case ablation benchmark
# Syntax (1-10) | Logic (11-20) | Performance/Runtime (21-30)
# ---------------------------------------------------------------------------
TEST_PROMPTS: list[dict[str, str]] = [
    # ── SYNTAX ERRORS (Cases 1–10) ──────────────────────────────────────────
    {
        "id": "syntax_01", "difficulty": "syntax",
        "keywords": ["colon", "syntax", "def"],
        "code": "def greet(name)\n    print('Hello', name)",
    },
    {
        "id": "syntax_02", "difficulty": "syntax",
        "keywords": ["colon", "syntax", "if"],
        "code": "x = 10\nif x > 5\n    print('big')",
    },
    {
        "id": "syntax_03", "difficulty": "syntax",
        "keywords": ["typo", "primt", "NameError", "print"],
        "code": "primt('Hello World')",
    },
    {
        "id": "syntax_04", "difficulty": "syntax",
        "keywords": ["indent", "IndentationError", "return"],
        "code": "def add(a, b):\nreturn a + b",
    },
    {
        "id": "syntax_05", "difficulty": "syntax",
        "keywords": ["bracket", "SyntaxError", "closing"],
        "code": "nums = [1, 2, 3\nprint(nums)",
    },
    {
        "id": "syntax_06", "difficulty": "syntax",
        "keywords": ["quote", "string", "SyntaxError", "unterminated"],
        "code": "msg = 'Hello World\nprint(msg)",
    },
    {
        "id": "syntax_07", "difficulty": "syntax",
        "keywords": ["colon", "for", "syntax"],
        "code": "for i in range(5)\n    print(i)",
    },
    {
        "id": "syntax_08", "difficulty": "syntax",
        "keywords": ["colon", "class", "syntax"],
        "code": "class Dog\n    def bark(self):\n        print('Woof')",
    },
    {
        "id": "syntax_09", "difficulty": "syntax",
        "keywords": ["parenthesis", "SyntaxError", "unclosed"],
        "code": "result = (10 + 5\nprint(result)",
    },
    {
        "id": "syntax_10", "difficulty": "syntax",
        "keywords": ["typo", "retrn", "return", "NameError"],
        "code": "def multiply(a, b):\n    retrn a * b",
    },
    # ── LOGIC ERRORS (Cases 11–20) ───────────────────────────────────────────
    {
        "id": "logic_11", "difficulty": "logic",
        "keywords": ["condition", "modulo", "odd", "wrong"],
        "code": "def is_even(n):\n    return n % 2 == 1",
    },
    {
        "id": "logic_12", "difficulty": "logic",
        "keywords": ["base case", "recursion", "infinite", "RecursionError"],
        "code": "def factorial(n):\n    return n * factorial(n - 1)",
    },
    {
        "id": "logic_13", "difficulty": "logic",
        "keywords": ["swap", "overwrite", "temp", "original"],
        "code": "def swap(a, b):\n    a = b\n    b = a\n    return a, b",
    },
    {
        "id": "logic_14", "difficulty": "logic",
        "keywords": ["off-by-one", "index", "range", "IndexError"],
        "code": "items = [1,2,3]\nfor i in range(len(items)+1):\n    print(items[i])",
    },
    {
        "id": "logic_15", "difficulty": "logic",
        "keywords": ["formula", "precedence", "fahrenheit", "celsius"],
        "code": (
            "def celsius_to_fahrenheit(c):\n"
            "    return c * 9 / 5 + 32\n\n"
            "print(celsius_to_fahrenheit(0))    # expected 32\n"
            "print(celsius_to_fahrenheit(100))  # expected 212"
        ),
    },
    {
        "id": "logic_16", "difficulty": "logic",
        "keywords": ["uppercase", "lower", "case", "vowel"],
        "code": (
            "def count_vowels(s):\n"
            "    count = 0\n"
            "    for ch in s:\n"
            "        if ch in 'aeiou':\n"
            "            count += 1\n"
            "    return count\n"
            "# Fails on uppercase vowels like 'A','E'"
        ),
    },
    {
        "id": "logic_17", "difficulty": "logic",
        "keywords": ["negative", "initialize", "max_val", "zero"],
        "code": (
            "def find_max(lst):\n"
            "    max_val = 0\n"
            "    for x in lst:\n"
            "        if x > max_val:\n"
            "            max_val = x\n"
            "    return max_val"
        ),
    },
    {
        "id": "logic_18", "difficulty": "logic",
        "keywords": ["infinite loop", "lo", "mid", "binary search"],
        "code": (
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: lo = mid\n"
            "        else: hi = mid\n"
            "    return -1"
        ),
    },
    {
        "id": "logic_19", "difficulty": "logic",
        "keywords": ["indent", "loop", "body", "outside"],
        "code": "total = 0\nfor i in range(1, 11):\ntotal += i\nprint(total)",
    },
    {
        "id": "logic_20", "difficulty": "logic",
        "keywords": ["TypeError", "type", "integer", "string"],
        "code": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
            "# Works for strings but called with integer:\n"
            "is_palindrome(12321)"
        ),
    },
    # ── PERFORMANCE / RUNTIME ISSUES (Cases 21–30) ───────────────────────────
    {
        "id": "perf_21", "difficulty": "performance",
        "keywords": ["ZeroDivisionError", "zero", "division", "guard"],
        "code": "def divide(a, b):\n    return a / b",
    },
    {
        "id": "perf_22", "difficulty": "performance",
        "keywords": ["IndexError", "empty", "bounds", "check"],
        "code": "def first(lst):\n    return lst[0]",
    },
    {
        "id": "perf_23", "difficulty": "performance",
        "keywords": ["NoneType", "None", "AttributeError", "TypeError"],
        "code": "d = None\nprint(d['key'])",
    },
    {
        "id": "perf_24", "difficulty": "performance",
        "keywords": ["close", "context manager", "with", "leak"],
        "code": "import os\nf = open('data.txt')\ndata = f.read()",
    },
    {
        "id": "perf_25", "difficulty": "performance",
        "keywords": ["O(n", "quadratic", "count", "performance", "set"],
        "code": (
            "def has_dup(lst):\n"
            "    for i in lst:\n"
            "        if lst.count(i) > 1:\n"
            "            return True\n"
            "    return False"
        ),
    },
    {
        "id": "perf_26", "difficulty": "performance",
        "keywords": ["exponential", "memoization", "cache", "performance", "lru_cache"],
        "code": (
            "def fib(n):\n"
            "    if n <= 1: return n\n"
            "    return fib(n-1) + fib(n-2)\n\n"
            "fib(50)  # extremely slow"
        ),
    },
    {
        "id": "perf_27", "difficulty": "performance",
        "keywords": ["join", "concatenation", "performance", "O(n"],
        "code": (
            "result = ''\n"
            "for word in ['Hello', 'World', 'from', 'Python']:\n"
            "    result += word + ' '"
        ),
    },
    {
        "id": "perf_28", "difficulty": "performance",
        "keywords": ["race condition", "lock", "thread", "mutex", "concurrent"],
        "code": (
            "import threading\n"
            "counter = 0\n"
            "def increment():\n"
            "    global counter\n"
            "    for _ in range(100000):\n"
            "        counter += 1\n"
            "t1 = threading.Thread(target=increment)\n"
            "t2 = threading.Thread(target=increment)\n"
            "t1.start(); t2.start()\n"
            "t1.join(); t2.join()\n"
            "print(counter)"
        ),
    },
    {
        "id": "perf_29", "difficulty": "performance",
        "keywords": ["connection", "pool", "every call", "reuse"],
        "code": (
            "def get_user(user_id):\n"
            "    db = connect_to_database()  # opens new connection every call\n"
            "    return db.query(f'SELECT * FROM users WHERE id={user_id}')"
        ),
    },
    {
        "id": "perf_30", "difficulty": "performance",
        "keywords": ["extend", "new list", "performance", "concatenation", "memory"],
        "code": (
            "def flatten(nested):\n"
            "    flat = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            flat = flat + flatten(item)  # creates new list each time\n"
            "        else:\n"
            "            flat.append(item)\n"
            "    return flat"
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
    detected: bool          # keyword-based auto-score (matches Ketu's eval method)
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

    keywords = prompt_meta.get("keywords", [])
    detected = any(kw.lower() in response_text.lower() for kw in keywords)

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
        detected=detected,
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
        print(f"{'Prompt':<14} {'Diff':<12} {'Det':>4} {'Avg Lat':>9} {'Tok/s':>7} {'VRAM Δ':>9}")
        print("-" * 66)

        seen: set[str] = set()
        for r in cfg_results:
            if r.prompt_id in seen:
                continue
            seen.add(r.prompt_id)
            runs = [x for x in cfg_results if x.prompt_id == r.prompt_id]
            avg_lat = _avg([x.latency_s for x in runs])
            avg_tps = _avg([x.tokens_per_sec for x in runs])
            det_count = sum(1 for x in runs if x.detected)
            det_str = f"{det_count}/{len(runs)}"
            vram_vals = [x.vram_delta_mb for x in runs if x.vram_delta_mb is not None]
            vram_str = f"{_avg(vram_vals):+.1f} MB" if vram_vals else "N/A"
            print(
                f"{r.prompt_id:<14} {r.difficulty:<12} {det_str:>4} "
                f"{avg_lat:>8.2f}s {avg_tps:>7.1f} {vram_str:>9}"
            )

        # Config-level summary
        all_lat = [r.latency_s for r in cfg_results]
        all_tps = [r.tokens_per_sec for r in cfg_results]
        total_det = sum(1 for r in cfg_results if r.detected)
        vram_all = [r.vram_delta_mb for r in cfg_results if r.vram_delta_mb is not None]
        acc_pct = 100 * total_det / len(cfg_results) if cfg_results else 0
        print("-" * 66)
        print(
            f"{'SUMMARY':<14} {'':<12} "
            f"{total_det}/{len(cfg_results)} ({acc_pct:.0f}%)  "
            f"{_avg(all_lat):>6.2f}s {_avg(all_tps):>7.1f} "
            f"{'N/A' if not vram_all else f'{_avg(vram_all):+.1f} MB':>9}"
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
