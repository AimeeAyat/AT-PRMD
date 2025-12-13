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
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    print(f"[OK] Model loaded")

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dpo_dataset()

    # DPO training configuration (using newer TRL DPOConfig API)
    dpo_config = DPOConfig(
        output_dir="./5_baselines/models/standard_dpo",
        logging_dir="./logs/baseline_dpo",

        # Training hyperparameters
        num_train_epochs=1,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # Batch settings
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,

        # DPO-specific parameters
        beta=0.1,  # KL penalty coefficient (lower than 0.5 for conservative optimization)
        label_smoothing=0.0,
        max_prompt_length=512,
        max_length=1024,

        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,

        # Checkpointing and logging
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=10,
        logging_strategy="steps",

        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,

        # Reporting
        report_to=["tensorboard"],
        push_to_hub=False,

        # Reproducibility
        seed=42,
    )

    # Initialize DPO trainer (ref_model=None auto-creates reference from base model)
    print("\n[*] Initializing DPO trainer...")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Auto-creates frozen reference model
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # New TRL API (replaces 'tokenizer')
    )

    # Train
    print("\n[*] Starting training...")
    print(f"  Model: {model_name}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Epochs: {dpo_config.num_train_epochs}")
    print(f"  Batch size: {dpo_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {dpo_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}")
    print(f"  Beta: {dpo_config.beta}")
    print(f"  Learning rate: {dpo_config.learning_rate}")

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
