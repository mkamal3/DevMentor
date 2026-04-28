"""
Merge a PEFT LoRA adapter into the base Qwen2.5-Coder model (full HF weights).

Required for Option A → GGUF → Ollama: Ollama does not consume raw adapter safetensors;
you must merge (or bake) LoRA weights into the base model first.

Usage (GPU with ~12 GB+ VRAM, 4-bit base — recommended):

    python finetune/merge_lora.py \\
        --adapter-path finetune/adapter_test \\
        --output-dir models/devmentor_merged \\
        --use-4bit-base

Alternative (needs more VRAM, ~14 GB BF16):

    python finetune/merge_lora.py \\
        --adapter-path finetune/adapter_test \\
        --output-dir models/devmentor_merged \\
        --dtype bf16

Next: convert merged folder to GGUF using llama.cpp (see finetune/ollama/README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base causal LM")
    p.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help="Hugging Face model ID for base weights.",
    )
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("finetune/adapter_test"),
        help="Folder containing adapter_config.json + adapter weights.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/devmentor_merged"),
        help="Where to save merged model + tokenizer (large).",
    )
    p.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Ignored when --use-4bit-base is set.",
    )
    p.add_argument(
        "--use-4bit-base",
        action="store_true",
        help="Load base in 4-bit (recommended for merge on 12 GB GPUs). Then merge_and_unload().",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for base model loading (e.g. auto, cpu).",
    )
    return p.parse_args()


def _dtype_from_flag(name: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    return mapping[name]


def main() -> None:
    args = parse_args()
    adapter_path = args.adapter_path.resolve()
    out_dir = args.output_dir.resolve()

    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Missing adapter_config.json under {adapter_path}. "
            "Point --adapter-path at the adapter root (or a checkpoint-* folder)."
        )

    print("Loading tokenizer...")
    if (adapter_path / "tokenizer_config.json").exists() or (adapter_path / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}")

    if args.use_4bit_base:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb,
            device_map=args.device_map,
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=_dtype_from_flag(args.dtype),
            device_map=args.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    print(f"Attaching adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, str(adapter_path), trust_remote_code=True)

    print("Merging and unloading adapter...")
    merged = model.merge_and_unload()

    out_dir.mkdir(parents=True)

    print(f"Saving merged model to {out_dir} ...")
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("\nMerge complete.")
    print(f"   Output: {out_dir}")
    print("   Next:   See finetune/ollama/README.md for GGUF conversion + ollama create.")


if __name__ == "__main__":
    main()
