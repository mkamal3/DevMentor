"""
DevMentor — LoRA fine-tuning script.

Stage A: Load config, model, tokenizer, and data.
Stage B: Attach LoRA adapter.
Stage C: Run training loop with TRL's SFTTrainer.

Usage (on Colab with GPU):
    python finetune/train_lora.py --config finetune/configs/lora_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ----------------------------- Config loading -----------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML config file into a dict."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------- Model loading -----------------------------

def build_quantization_config(quant_cfg: dict[str, Any]) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig for 4-bit QLoRA loading."""
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(model_cfg: dict[str, Any], quant_cfg: dict[str, Any]):
    """Load the tokenizer and the 4-bit quantized base model."""
    model_name = model_cfg["name"]
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Many code models don't ship a pad token; reuse EOS for padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading 4-bit quantized model from {model_name}...")
    print("(First run: will download ~5 GB of weights. Subsequent runs: cached.)")
    bnb_config = build_quantization_config(quant_cfg=quant_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False            # Required for gradient checkpointing
    model.config.pretraining_tp = 1

    return model, tokenizer


# ----------------------------- LoRA attach -----------------------------

def attach_lora(model, lora_cfg: dict[str, Any]) -> PeftModel:
    """Wrap the frozen base model with a trainable LoRA adapter.

    This performs two steps:
      1. Prepare the 4-bit model for k-bit training (freezes base, enables
         gradient checkpointing, casts layer-norm weights to fp32 for stability).
      2. Build a LoraConfig from the YAML settings and inject adapter weights
         into the model's attention projection layers.

    Args:
        model: The 4-bit base model returned by load_base_model().
        lora_cfg: The 'lora' section of the YAML config.

    Returns:
        A PeftModel — the base model with LoRA adapters attached.
    """
    # Step 1: freeze base weights + enable gradient-friendly config.
    model = prepare_model_for_kbit_training(model)

    # Step 2: build the LoRA recipe from config.
    peft_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    # Step 3: inject LoRA adapters into the target modules.
    peft_model = get_peft_model(model, peft_config)

    return peft_model


def report_trainable_parameters(model) -> None:
    """Print the LoRA adapter size vs. the full model size."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(
        f"Trainable parameters: {trainable:>15,}  ({pct:.3f}% of {total:,} total)")


# ----------------------------- Data loading -----------------------------

def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load a JSONL file into a list of dicts (one per line)."""
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_datasets(dataset_cfg: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """Load train + validation JSONL files as HuggingFace Datasets."""
    train_path = Path(dataset_cfg["train_file"])
    val_path = Path(dataset_cfg["val_file"])

    print(f"Loading training data from {train_path}...")
    train_records = load_jsonl(path=train_path)
    print(f"Loading validation data from {val_path}...")
    val_records = load_jsonl(path=val_path)

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    return train_dataset, val_dataset


# ----------------------------- Training -----------------------------

def formatting_prompts_func(example: dict[str, str]) -> str:
    """Combine prompt + completion into a single text string for SFTTrainer.

    SFTTrainer in TRL >=0.9 expects each dataset row to be converted into
    a single 'text' string that the tokenizer can directly process.
    """
    return f"{example['prompt']}\n\n{example['completion']}"


def build_sft_config(training_cfg: dict[str, Any], max_seq_length: int) -> SFTConfig:
    """Build an SFTConfig (the TRL wrapper around HuggingFace TrainingArguments)."""
    return SFTConfig(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        optim=training_cfg["optim"],
        logging_steps=training_cfg["logging_steps"],
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=training_cfg["eval_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        bf16=training_cfg["bf16"],
        seed=training_cfg["seed"],
        max_seq_length=max_seq_length,
        report_to="none",             # Disable wandb/tensorboard by default
        gradient_checkpointing=True,  # Trade compute for memory — needed on L4
    )


def run_training(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    sft_config: SFTConfig,
) -> SFTTrainer:
    """Build the trainer and run the training loop."""
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    print("\n" + "=" * 60)
    print("Training complete. Saving final adapter...")
    print("=" * 60)
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)

    return trainer


# ----------------------------- Main -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DevMentor LoRA trainer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("finetune/configs/lora_config.yaml"),
        help="Path to the LoRA YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("DevMentor LoRA Trainer — Stage A: Setup")
    print("=" * 60)

    # --- 1. Load config ---
    print(f"\nLoading config from {args.config}...")
    config = load_config(config_path=args.config)
    print(f"Experiment name: {config['experiment']['name']}")

    # --- 2. Load model + tokenizer ---
    model, tokenizer = load_base_model(
        model_cfg=config["model"],
        quant_cfg=config["quantization"],
    )

    # --- 3. Load datasets ---
    train_dataset, val_dataset = build_datasets(dataset_cfg=config["dataset"])

    # --- 4. Attach LoRA adapter (Stage B) ---
    print("\nAttaching LoRA adapter...")
    model = attach_lora(model=model, lora_cfg=config["lora"])

    # --- 5. Report what we loaded ---
    print("\n" + "=" * 60)
    print("Stages A + B complete — setup summary")
    print("=" * 60)
    print(f"Base model          : {config['model']['name']}")
    print(f"Quantization        : 4-bit NF4")
    print(f"Training examples   : {len(train_dataset):,}")
    print(f"Validation examples : {len(val_dataset):,}")

    report_trainable_parameters(model=model)

    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory in use   : {gpu_mem_gb:.2f} GB")
    else:
        print("GPU memory in use   : N/A (no CUDA)")

    # --- 6. Run training (Stage C) ---
    sft_config = build_sft_config(
        training_cfg=config["training"],
        max_seq_length=config["model"]["max_seq_length"],
    )

    trainer = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        sft_config=sft_config,
    )

    # --- 7. Final report ---
    print("\n" + "=" * 60)
    print("✅ Training finished successfully!")
    print("=" * 60)
    print(f"Adapter saved to  : {sft_config.output_dir}")
    print(f"Total train steps : {trainer.state.global_step:,}")
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        if "loss" in last_log:
            print(f"Final training loss: {last_log['loss']:.4f}")


if __name__ == "__main__":
    main()
