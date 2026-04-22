"""
DevMentor — Dataset preparation for LoRA fine-tuning.

Pipeline:
  Stage 1: Download the code_x_glue_cc_code_refinement dataset.
  Stage 2: Format raw {buggy, fixed} pairs into {prompt, completion}.
  Stage 3: Save train.jsonl, val.jsonl, and a small sample.jsonl.

Output files (in data/bugfix/processed/):
  - train.jsonl   : 1000 examples for LoRA training
  - val.jsonl     :  100 examples for validation during training
  - sample.jsonl  :    3 examples (committed to Git so teammates see format)
"""

import json
from pathlib import Path

from datasets import load_dataset


# --- Config ---
NUM_TRAIN_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 100
NUM_SAMPLE_EXAMPLES = 3

OUTPUT_DIR = Path("data/bugfix/processed")

# Prompt components
SYSTEM_INSTRUCTION = (
    "You are DevMentor, an expert code reviewer. Analyze the following "
    "Java code, identify the bug, and provide a fix."
)

RESPONSE_FORMAT = (
    "Provide your response in this format:\n"
    "Bug: <one-line description of the bug>\n"
    "Fix:\n"
    "```java\n"
    "<corrected code>\n"
    "```"
)

GENERIC_BUG_DESCRIPTION = (
    "The code contains a defect that has been corrected in the fix below."
)


def format_example(buggy_code: str, fixed_code: str) -> dict:
    """Convert a raw (buggy, fixed) pair into a (prompt, completion) pair."""
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Buggy code:\n"
        f"```java\n"
        f"{buggy_code}\n"
        f"```\n\n"
        f"{RESPONSE_FORMAT}"
    )

    completion = (
        f"Bug: {GENERIC_BUG_DESCRIPTION}\n"
        f"Fix:\n"
        f"```java\n"
        f"{fixed_code}\n"
        f"```"
    )

    return {"prompt": prompt, "completion": completion}


def write_jsonl(examples: list[dict], output_path: Path) -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def build_formatted_examples(raw_split, num_examples: int) -> list[dict]:
    """Format the first N raw examples into training-ready dicts."""
    formatted = []
    for i in range(min(num_examples, len(raw_split))):
        raw = raw_split[i]
        formatted.append(
            format_example(
                buggy_code=raw["buggy"],
                fixed_code=raw["fixed"],
            )
        )
    return formatted


def main() -> None:
    """Full pipeline: download, format, save."""

    print("=" * 60)
    print("DevMentor Dataset Prep — Stages 1, 2, 3")
    print("=" * 60)

    print("\nLoading dataset (from cache if available)...")
    dataset = load_dataset(
        "google/code_x_glue_cc_code_refinement",
        "medium",
    )
    print("Dataset loaded.\n")

    # --- Format ---
    print(f"Formatting {NUM_TRAIN_EXAMPLES} training examples...")
    train_examples = build_formatted_examples(
        raw_split=dataset["train"],
        num_examples=NUM_TRAIN_EXAMPLES,
    )

    print(f"Formatting {NUM_VAL_EXAMPLES} validation examples...")
    val_examples = build_formatted_examples(
        raw_split=dataset["validation"],
        num_examples=NUM_VAL_EXAMPLES,
    )

    sample_examples = train_examples[:NUM_SAMPLE_EXAMPLES]

    # --- Save ---
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    sample_path = OUTPUT_DIR / "sample.jsonl"

    write_jsonl(examples=train_examples, output_path=train_path)
    write_jsonl(examples=val_examples, output_path=val_path)
    write_jsonl(examples=sample_examples, output_path=sample_path)

    # --- Report ---
    print("\n" + "=" * 60)
    print("Files written:")
    print("=" * 60)
    for path, count in [
        (train_path, len(train_examples)),
        (val_path, len(val_examples)),
        (sample_path, len(sample_examples)),
    ]:
        size_kb = path.stat().st_size / 1024
        print(f"  {str(path):45s}  {count:>5,} examples  ({size_kb:>7.1f} KB)")

    print("\n✅ Step 4 complete. Training data is ready for Colab.")


if __name__ == "__main__":
    main()
