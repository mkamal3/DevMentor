"""Run Config A/B evaluation on 30 benchmark cases.

Metrics reported:
1. Bug Detection Accuracy
2. Fix Correctness
3. Explanation Clarity
4. Hallucination Rate
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings, Settings
from llm.ollama_client import OllamaClient
from prompts.devmentor_prompt import build_user_message, get_system_prompt
from rag.retriever import retrieve_context
from utils.benchmark import TEST_PROMPTS


LEAK_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "[response]")


@dataclass
class CaseEval:
    config: str
    prompt_id: str
    difficulty: str
    latency_s: float
    bug_detected: bool
    fix_correct: bool
    clarity_score: float
    hallucinated: bool


def _has_code_block(text: str) -> bool:
    return bool(re.search(r"```[a-zA-Z]*\n[\s\S]*?\n```", text))


def _extract_first_code_block(text: str) -> str:
    match = re.search(r"```[a-zA-Z]*\n([\s\S]*?)\n```", text)
    return match.group(1).strip() if match else ""


def _bug_detection_accuracy(response: str, keywords: list[str]) -> bool:
    lower = response.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _fix_correctness(prompt_code: str, response: str) -> bool:
    lower = response.lower()
    if not _has_code_block(response):
        return False
    code = _extract_first_code_block(response)
    if not code:
        return False
    if code.strip() == prompt_code.strip():
        return False
    return any(token in lower for token in ("fix", "correct", "suggested"))


def _clarity_score(response: str) -> float:
    score = 1.0
    lower = response.lower()
    if "## 1." in response and "## 2." in response and "## 3." in response:
        score += 2.0
    if any(word in lower for word in ("because", "occurs", "cause", "reason")):
        score += 1.0
    if len(response.split()) >= 80:
        score += 1.0
    return min(score, 5.0)


def _hallucination_flag(response: str) -> bool:
    lower = response.lower()
    if any(token in response for token in LEAK_TOKENS):
        return True
    if len(response.strip()) < 12:
        return True
    if "please provide a response in this format" in lower:
        return True
    return False


def _run_case(
    config_name: str,
    prompt_meta: dict[str, Any],
    settings: Settings,
    model_name: str,
    use_rag: bool,
) -> CaseEval:
    code = prompt_meta["code"]
    contexts = retrieve_context(query=code, k=settings.top_k) if use_rag else []

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

    client = OllamaClient(settings=patched)
    t0 = time.perf_counter()
    response = client.generate_chat_response(
        system=get_system_prompt(),
        user=build_user_message(query=code, contexts=contexts),
    )
    latency = time.perf_counter() - t0

    return CaseEval(
        config=config_name,
        prompt_id=prompt_meta["id"],
        difficulty=prompt_meta["difficulty"],
        latency_s=round(latency, 3),
        bug_detected=_bug_detection_accuracy(response, prompt_meta["keywords"]),
        fix_correct=_fix_correctness(code, response),
        clarity_score=_clarity_score(response),
        hallucinated=_hallucination_flag(response),
    )


def _summary(cases: list[CaseEval]) -> dict[str, Any]:
    total = len(cases)
    if total == 0:
        return {}
    bug_acc = sum(1 for c in cases if c.bug_detected) / total
    fix_acc = sum(1 for c in cases if c.fix_correct) / total
    clarity = sum(c.clarity_score for c in cases) / total
    halluc_rate = sum(1 for c in cases if c.hallucinated) / total
    latency = sum(c.latency_s for c in cases) / total
    return {
        "total_cases": total,
        "bug_detection_accuracy": round(bug_acc * 100, 2),
        "fix_correctness": round(fix_acc * 100, 2),
        "explanation_clarity_avg_1_to_5": round(clarity, 2),
        "hallucination_rate": round(halluc_rate * 100, 2),
        "avg_latency_s": round(latency, 2),
    }


def _summary_by_difficulty(cases: list[CaseEval]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for difficulty in ("syntax", "logic", "performance"):
        subset = [c for c in cases if c.difficulty == difficulty]
        out[difficulty] = _summary(subset)
    return out


def main() -> None:
    settings = get_settings()
    model = settings.ollama_model

    all_cases: list[CaseEval] = []
    for cfg_name, use_rag in (("A", False), ("B", True)):
        print(f"Running config {cfg_name}...")
        for idx, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  [{cfg_name} {idx}/30] {prompt['id']}")
            all_cases.append(_run_case(cfg_name, prompt, settings, model, use_rag))

    config_a = [c for c in all_cases if c.config == "A"]
    config_b = [c for c in all_cases if c.config == "B"]

    report = {
        "model": model,
        "config_A": {
            "overall": _summary(config_a),
            "by_difficulty": _summary_by_difficulty(config_a),
        },
        "config_B": {
            "overall": _summary(config_b),
            "by_difficulty": _summary_by_difficulty(config_b),
        },
        "cases": [asdict(c) for c in all_cases],
    }

    out = Path("results/ab_eval_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to: {out}")
    print(json.dumps({"config_A": report["config_A"]["overall"], "config_B": report["config_B"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
