"""
Train a reward model for a specific objective using TRL RewardTrainer.
Generic script that works for any objective (helpful, harmless, honest).
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    set_seed
)
from trl import RewardTrainer

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_device, print_model_size, print_gpu_utilization
from visualization import RewardVisualizer, save_step_results


def load_config(config_path: str = "configs/reward_model_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset_for_reward_modeling(dataset, tokenizer, max_length: int = 512):
    """
    Prepare dataset for reward modeling.
    TRL RewardTrainer expects: input_ids_chosen, attention_mask_chosen,
                                input_ids_rejected, attention_mask_rejected
    """
    def tokenize_function(examples):
        # Tokenize chosen responses
        chosen_encodings = tokenizer(
            examples['chosen'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )

        # Tokenize rejected responses
        rejected_encodings = tokenizer(
            examples['rejected'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )

        return {
            'input_ids_chosen': chosen_encodings['input_ids'],
            'attention_mask_chosen': chosen_encodings['attention_mask'],
            'input_ids_rejected': rejected_encodings['input_ids'],
            'attention_mask_rejected': rejected_encodings['attention_mask'],
        }

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )


def evaluate_and_visualize(
    model,
    tokenizer,
    dataset,
    objective: str,
    split: str,
    device: torch.device,
    output_dir: str,
    num_samples: int = 100
):
    """
    Evaluate model and create visualizations.
    Returns metrics dict.
    """
    model.eval()

    rewards_chosen = []
    rewards_rejected = []
    examples_with_rewards = []

    # Sample subset for evaluation
    eval_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in eval_indices:
            example = dataset[int(idx)]

            # Get rewards for chosen
            inputs_chosen = {
                'input_ids': torch.tensor([example['input_ids_chosen']]).to(device),
                'attention_mask': torch.tensor([example['attention_mask_chosen']]).to(device)
            }
            reward_chosen = model(**inputs_chosen).logits[0, 0].item()

            # Get rewards for rejected
            inputs_rejected = {
                'input_ids': torch.tensor([example['input_ids_rejected']]).to(device),
                'attention_mask': torch.tensor([example['attention_mask_rejected']]).to(device)
            }
            reward_rejected = model(**inputs_rejected).logits[0, 0].item()

            rewards_chosen.append(reward_chosen)
            rewards_rejected.append(reward_rejected)

            # Decode for examples
            if len(examples_with_rewards) < 10:
                chosen_text = tokenizer.decode(example['input_ids_chosen'], skip_special_tokens=True)
                rejected_text = tokenizer.decode(example['input_ids_rejected'], skip_special_tokens=True)

                examples_with_rewards.append({
                    'chosen': chosen_text,
                    'rejected': rejected_text,
                    'reward_chosen': reward_chosen,
                    'reward_rejected': reward_rejected,
                    'margin': reward_chosen - reward_rejected
                })

    # Calculate metrics
    margins = [c - r for c, r in zip(rewards_chosen, rewards_rejected)]
    accuracy = sum(1 for m in margins if m > 0) / len(margins)

    metrics = {
        'accuracy': accuracy,
        'mean_reward_chosen': float(np.mean(rewards_chosen)),
        'mean_reward_rejected': float(np.mean(rewards_rejected)),
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins)),
    }

    # Create visualizations
    visualizer = RewardVisualizer(output_dir=output_dir)

    reward_dist = {
        'chosen': rewards_chosen,
        'rejected': rewards_rejected
    }
    visualizer.plot_reward_distributions(reward_dist, objective, split)
    visualizer.plot_reward_margins(margins, objective, split)
    visualizer.save_top_bottom_examples(examples_with_rewards, objective, split, n=10)

    return metrics


def train_reward_model(objective: str, approach: str = "hh_truthfulqa"):
    """
    Train a reward model for a specific objective.

    Args:
        objective: One of 'helpful', 'harmless', 'honest'
        approach: Data approach ('hh_truthfulqa' or 'pku_safe')
    """
    print("\n" + "=" * 80)
    print(f"Training Reward Model: {objective.upper()}")
    print(f"Approach: {approach}")
    print("=" * 80)

    # Load config
    config = load_config()
    set_seed(config['training']['seed'])

    # Setup device
    device = get_device()

    # Load tokenizer and model
    model_name = config['model']['name']
    print(f"\n[*] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=config['model']['cache_dir']
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Reward model outputs scalar
        cache_dir=config['model']['cache_dir'],
        torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float32
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    print_model_size(model)
    print_gpu_utilization()

    # Load datasets
    data_path = f"./data/processed/{approach}/{objective}"
    print(f"\n[*] Loading datasets from {data_path}")

    dataset = load_from_disk(data_path)
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Prepare datasets
    print("\n[*] Tokenizing datasets...")
    max_length = config['training']['max_length']

    train_dataset = prepare_dataset_for_reward_modeling(dataset['train'], tokenizer, max_length)
    eval_dataset = prepare_dataset_for_reward_modeling(dataset['validation'], tokenizer, max_length)
    test_dataset = prepare_dataset_for_reward_modeling(dataset['test'], tokenizer, max_length)

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./models/reward_models/{objective}_rm_{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],
        bf16=config['training']['bf16'],
        tf32=config['training']['tf32'],
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        logging_dir=f"{config['training']['logging_dir']}/{objective}_rm_{timestamp}",
        report_to=config['training']['report_to'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training'].get('greater_is_better', True),
        seed=config['training']['seed'],
        data_seed=config['training']['data_seed'],
        remove_unused_columns=False,
        run_name=f"{objective}_rm_{approach}",
    )

    # Initialize trainer
    print("\n[*] Initializing RewardTrainer...")

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\n[*] Starting training...")
    print("=" * 80)

    train_result = trainer.train()

    print("\n[*] Training complete!")
    print("=" * 80)

    # Save final model
    final_model_path = f"./models/reward_models/{objective}_rm"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n[*] Saved final model to {final_model_path}")

    # Evaluate on test set
    print("\n[*] Evaluating on test set...")
    test_metrics = evaluate_and_visualize(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        objective=objective,
        split="test",
        device=device,
        output_dir="./outputs/reward_analysis"
    )

    print(f"\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results = {
        'objective': objective,
        'approach': approach,
        'model_name': model_name,
        'output_dir': output_dir,
        'final_model_path': final_model_path,
        'train_metrics': {
            'train_loss': train_result.training_loss,
            'train_steps': train_result.global_step,
        },
        'test_metrics': test_metrics,
        'config': config['training']
    }

    save_step_results(f'train_{objective}_rm', results, approach)

    print("\n" + "=" * 80)
    print(f"[OK] {objective.upper()} reward model training complete!")
    print("=" * 80)
    print(f"\nModel saved to: {final_model_path}")
    print(f"Visualizations: ./outputs/reward_analysis/")
    print(f"TensorBoard logs: {training_args.logging_dir}")
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Mean Margin: {test_metrics['mean_margin']:.4f}")

    return final_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--objective",
        type=str,
        required=True,
        choices=['helpful', 'harmless', 'honest'],
        help="Objective to train reward model for"
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="hh_truthfulqa",
        choices=['hh_truthfulqa', 'pku_safe'],
        help="Data approach"
    )

    args = parser.parse_args()

    train_reward_model(args.objective, args.approach)
