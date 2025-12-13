"""
Train standard DPO baseline using TRL library.
Uses our prepared HH-RLHF dataset (helpful + harmless combined).
No pessimistic aggregation - standard single-objective DPO.
"""

import os
import sys
import torch
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from utils import set_seed, get_device


def load_config(config_path: str = "configs/policy_config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dpo_dataset(approach: str = "hh_truthfulqa"):
    """Load and combine helpful + harmless datasets for standard DPO."""
    print("\n[*] Loading datasets...")

    # Load helpful and harmless datasets
    helpful_path = f"./data/processed/{approach}/helpful"
    harmless_path = f"./data/processed/{approach}/harmless"

    helpful_data = load_from_disk(helpful_path)
    harmless_data = load_from_disk(harmless_path)

    # Combine train sets
    train_dataset = concatenate_datasets([helpful_data['train'], harmless_data['train']])
    eval_dataset = concatenate_datasets([helpful_data['validation'], harmless_data['validation']])

    print(f"[OK] Combined dataset prepared")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Eval: {len(eval_dataset):,} examples")

    return train_dataset, eval_dataset


def train_standard_dpo():
    """Train standard DPO model using TRL."""
    print("\n" + "=" * 80)
    print("Training Standard DPO Baseline (TRL)")
    print("=" * 80)

    # Setup
    set_seed(42)
    device = get_device()

    # Load config
    config = load_config()

    # Load model and tokenizer
    model_name = config['model']['name']
    print(f"\n[*] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load reference model
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"[OK] Models loaded")

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dpo_dataset()

    # DPO training arguments
    training_args = DPOConfig(
        output_dir="./5_baselines/models/standard_dpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        logging_dir="./logs/baseline_dpo",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        seed=42,
        beta=0.1,  # KL penalty coefficient
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False
    )

    # Initialize DPO trainer
    print("\n[*] Initializing DPO trainer...")

    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\n[*] Starting training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

    # Save final model
    print("\n[*] Saving final model...")
    trainer.save_model("./5_baselines/models/standard_dpo/final")

    print("\n" + "=" * 80)
    print("[OK] Standard DPO training complete!")
    print("=" * 80)
    print(f"Model saved to: ./5_baselines/models/standard_dpo/final")
    print("\nNext step: Evaluate standard DPO on benchmarks")


if __name__ == "__main__":
    train_standard_dpo()
