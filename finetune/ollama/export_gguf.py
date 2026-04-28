"""
Optional helper: HF merged model → GGUF using llama.cpp's convert_hf_to_gguf.py

Prerequisites:
  1. Clone/build llama.cpp: https://github.com/ggerganov/llama.cpp
  2. Set environment variable LLAMA_CPP (folder containing convert_hf_to_gguf.py)
     Example (PowerShell):
       $env:LLAMA_CPP = "C:\\src\\llama.cpp"

Usage:
  python finetune/ollama/export_gguf.py \\
      --merged-dir models/devmentor_merged \\
      --out-file models/devmentor-f16.gguf \\
      --outtype f16

Then quantize (optional, recommended for inference):
  <llama.cpp>\\quantize.exe models\\devmentor-f16.gguf models\\devmentor-q4_k_m.gguf Q4_K_M

Or on Unix:
  ./llama-quantize model-f16.gguf model-q4.gguf q4_K_M

Then register with Ollama using Modelfile (see README.md in this folder).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_convert_script(llama_cpp: Path) -> Path:
    cand = llama_cpp / "convert_hf_to_gguf.py"
    if cand.exists():
        return cand
    # Some layouts use gguf-py
    cand2 = llama_cpp / "tools" / "convert_hf_to_gguf.py"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Cannot find convert_hf_to_gguf.py under {llama_cpp}")


def main() -> None:
    p = argparse.ArgumentParser(description="HF → GGUF via llama.cpp")
    p.add_argument(
        "--merged-dir",
        type=Path,
        default=Path("models/devmentor_merged"),
        help="Output from merge_lora.py.",
    )
    p.add_argument(
        "--out-file",
        type=Path,
        default=Path("models/devmentor-f16.gguf"),
        help="Destination GGUF file.",
    )
    p.add_argument(
        "--outtype",
        default="f16",
        help="convert_hf_to_gguf.py --outtype (e.g. f16, bf16, q8_0)",
    )
    args = p.parse_args()

    root = os.environ.get("LLAMA_CPP")
    if not root:
        print(
            "Set LLAMA_CPP to your llama.cpp clone root (folder containing "
            "convert_hf_to_gguf.py).\n"
            "Example: $env:LLAMA_CPP = 'C:\\\\src\\\\llama.cpp'",
            file=sys.stderr,
        )
        sys.exit(1)

    llama = Path(root).resolve()
    script = find_convert_script(llama)

    merged = args.merged_dir.resolve()
    if not (merged / "config.json").exists():
        raise FileNotFoundError(f"Expected HF merged config at {merged}/config.json — run merge_lora.py first.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    outfile = args.out_file.resolve()

    cmd = [
        sys.executable,
        str(script),
        str(merged),
        "--outfile",
        str(outfile),
        "--outtype",
        args.outtype,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nGGUF written: {outfile}")


if __name__ == "__main__":
    main()
