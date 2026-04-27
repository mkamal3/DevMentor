"""
fetch_and_convert.py
--------------------
Downloads HuggingFace datasets + GitHub repos and converts them into
plain .md / .json files ready for your ChromaDB ingestion pipeline.

Output layout:
    data/
    ├── docs/
    │   ├── conala_docs.md
    │   ├── codesearchnet_python.md
    │   ├── codesearchnet_javascript.md
    │   ├── codesearchnet_java.md
    │   └── tldr_pages/          ← cloned markdown files
    └── errors/
        ├── stackoverflow_qa.json
        ├── conala_errors.json
        └── codesearchnet_errors.json

Requirements:
    pip install datasets huggingface_hub tqdm gitpython
"""

import os
import json
import re
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR   = Path("data/docs")
ERRORS_DIR = Path("data/errors")
MAX_ROWS   = 5_000   # cap per dataset to keep things manageable; raise as needed

# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Strip null bytes, excess whitespace, and HTML tags."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)          # remove HTML
    text = re.sub(r"\x00", "", text)              # null bytes
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    return text.strip()


def write_md(path: Path, blocks: list[str]):
    """Write a list of markdown blocks to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(blocks))
    print(f"  ✔  Wrote {len(blocks):,} blocks → {path}")


def write_json(path: Path, records: list[dict]):
    """Write a list of dicts as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✔  Wrote {len(records):,} records → {path}")


def load_hf(dataset_id: str, split: str = "train", **kwargs):
    """Load a HuggingFace dataset with a friendly error message."""
    try:
        from datasets import load_dataset
        return load_dataset(dataset_id, split=split, trust_remote_code=True, **kwargs)
    except Exception as e:
        print(f"  ✗  Could not load '{dataset_id}': {e}")
        return None

# ── Dataset handlers ──────────────────────────────────────────────────────────

def fetch_conala():
    """
    neulab/conala  —  Stack Overflow mined intent+code pairs.
    Splits into docs (intent → code explanation) and errors (intent as symptom).
    """
    print("\n[1/4] Fetching CoNaLa (Stack Overflow mined) …")
    ds = load_hf("neulab/conala", name="mined")
    if ds is None:
        return

    docs_blocks  = []
    error_records = []

    for row in tqdm(ds.select(range(min(MAX_ROWS, len(ds)))), desc="  CoNaLa"):
        intent  = clean_text(str(row.get("intent", "")))
        snippet = clean_text(str(row.get("snippet", "")))
        if not intent or not snippet:
            continue

        # ── docs: intent as question, snippet as answer ──
        docs_blocks.append(
            f"## {intent}\n\n```python\n{snippet}\n```"
        )

        # ── errors: treat intent as a symptom description ──
        error_records.append({
            "symptom" : intent,
            "cause"   : "Common Python pattern question",
            "fix"     : snippet,
            "source"  : "CoNaLa/StackOverflow",
            "language": "python",
        })

    write_md(DOCS_DIR / "conala_docs.md", docs_blocks)
    write_json(ERRORS_DIR / "conala_errors.json", error_records)


def fetch_codesearchnet():
    """
    code_search_net  —  GitHub function docstrings + code across 6 languages.
    Produces per-language doc files and an errors file for functions with
    docstrings that describe failure cases.
    """
    print("\n[2/4] Fetching CodeSearchNet …")

    languages = ["python", "javascript", "java"]

    for lang in languages:
        ds = load_hf("code_search_net", name=lang)
        if ds is None:
            continue

        docs_blocks   = []
        error_records = []

        for row in tqdm(
            ds.select(range(min(MAX_ROWS, len(ds)))),
            desc=f"  CodeSearchNet/{lang}",
        ):
            doc  = clean_text(str(row.get("func_documentation_string", "")))
            code = clean_text(str(row.get("whole_func_string", "")))
            url  = row.get("func_code_url", "")
            if not doc or not code:
                continue

            # ── docs ──
            docs_blocks.append(
                f"## Function Documentation\n\n"
                f"**Source**: {url}\n\n"
                f"{doc}\n\n"
                f"```{lang}\n{code}\n```"
            )

            # ── errors: pull docstrings that mention errors/exceptions ──
            error_keywords = {"error", "exception", "raise", "fail", "invalid", "cannot", "bug"}
            if any(kw in doc.lower() for kw in error_keywords):
                error_records.append({
                    "symptom" : doc,
                    "cause"   : f"See function implementation in {lang}",
                    "fix"     : code,
                    "source"  : f"CodeSearchNet/{lang}",
                    "language": lang,
                    "url"     : url,
                })

        write_md(DOCS_DIR / f"codesearchnet_{lang}.md", docs_blocks)
        if error_records:
            write_json(ERRORS_DIR / f"codesearchnet_{lang}_errors.json", error_records)


def fetch_stackoverflow():
    """
    kye/stack-overflow-questions-answers  —  Raw Q&A pairs.
    Goes into errors/ because it's the closest to symptom→fix format.
    """
    print("\n[3/4] Fetching Stack Overflow Q&A …")
    ds = load_hf("kye/stack-overflow-questions-answers")
    if ds is None:
        return

    records = []
    for row in tqdm(
        ds.select(range(min(MAX_ROWS, len(ds)))),
        desc="  StackOverflow",
    ):
        question = clean_text(str(row.get("question", "")))
        answer   = clean_text(str(row.get("answer",   "")))
        if not question or not answer:
            continue

        records.append({
            "symptom" : question,
            "cause"   : "",          # not available in this dataset
            "fix"     : answer,
            "source"  : "StackOverflow",
            "language": "mixed",
        })

    write_json(ERRORS_DIR / "stackoverflow_qa.json", records)


def clone_tldr():
    """
    tldr-pages/tldr  —  Short, structured command examples in markdown.
    Already clean markdown → copy pages/common + pages/linux into data/docs/tldr/.
    """
    print("\n[4/4] Cloning TLDR pages …")
    clone_dir = Path("data/_tmp_tldr")
    dest_dir  = DOCS_DIR / "tldr_pages"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/tldr-pages/tldr.git", str(clone_dir)],
            check=True,
        )
    else:
        print("  → Already cloned, skipping git clone.")

    # Copy relevant platform folders
    count = 0
    for platform in ("common", "linux", "osx"):
        src = clone_dir / "pages" / platform
        if not src.exists():
            continue
        for md_file in src.glob("*.md"):
            target = dest_dir / f"{platform}_{md_file.name}"
            target.write_text(md_file.read_text(encoding="utf-8"), encoding="utf-8")
            count += 1

    print(f"  ✔  Copied {count} TLDR markdown files → {dest_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    print("=" * 60)
    print("  RAG Data Fetcher — starting …")
    print(f"  MAX_ROWS per dataset : {MAX_ROWS:,}")
    print(f"  docs/  → {DOCS_DIR.resolve()}")
    print(f"  errors/→ {ERRORS_DIR.resolve()}")
    print("=" * 60)

    fetch_conala()
    fetch_codesearchnet()
    fetch_stackoverflow()
    clone_tldr()

    print("\n" + "=" * 60)
    print("  ✅  All done! Your folders are ready for ingestion.")
    print("=" * 60)

    # ── Summary ──
    print("\nFile summary:")
    for folder in (DOCS_DIR, ERRORS_DIR):
        files = list(folder.rglob("*"))
        files = [f for f in files if f.is_file()]
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"  {folder}: {len(files)} files  ({total_mb:.1f} MB)")


if __name__ == "__main__":
    # Sanity-check dependencies before running
    missing = []
    for pkg in ("datasets", "huggingface_hub", "tqdm", "git"):
        try:
            __import__(pkg if pkg != "git" else "git")
        except ImportError:
            missing.append(pkg if pkg != "git" else "gitpython")

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)

    main()
