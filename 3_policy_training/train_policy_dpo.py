"""
Train policy using DPO (Direct Preference Optimization) with pessimistic multi-objective reward distillation.
Supports: baseline DPO, pessimistic DPO (hard_min, CVaR), hierarchical pessimistic DPO.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    set_seed
)
from trl import DPOTrainer

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_device, print_model_size, print_gpu_utilization
from visualization import save_step_results, TrainingVisualizer


def load_config(config_path: str = "configs/policy_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_reward_models(config: Dict, device: torch.device) -> Dict[str, AutoModelForSequenceClassification]:
    """
    Load trained reward models for each objective.

    Returns:
        Dict mapping objective name to reward model
    """
    print("\n[*] Loading reward models...")

    reward_models = {}
    objectives = ['helpful', 'harmless', 'honest']

    for objective in objectives:
        model_path = config['reward_models'][f'{objective}_path']
        print(f"  Loading {objective} RM from {model_path}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float32
        )
        model.to(device)
        model.eval()

        reward_models[objective] = model

    print(f"[OK] Loaded {len(reward_models)} reward models")
    return reward_models


def compute_ensemble_rewards(
    reward_models: Dict,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512
) -> Dict[str, List[float]]:
    """
    Compute rewards from all reward models for given texts.

    Returns:
        Dict mapping objective to list of reward scores
    """
    rewards = {obj: [] for obj in reward_models.keys()}

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(device)

            for obj, model in reward_models.items():
                reward = model(**inputs).logits[0, 0].item()
                rewards[obj].append(reward)

    return rewards


def apply_pessimism(
    rewards: Dict[str, List[float]],
    method: str = "hard_min",
    cvar_alpha: float = 0.1,
    weights: Optional[Dict[str, float]] = None
) -> List[float]:
    """
    Apply pessimistic aggregation across objectives.

    Args:
        rewards: {objective: [scores]} for all objectives
        method: "hard_min", "cvar", "weighted_min"
        cvar_alpha: For CVaR method, quantile level
        weights: Objective weights for weighted methods

    Returns:
        List of aggregated reward scores
    """
    n_samples = len(next(iter(rewards.values())))
    objectives = list(rewards.keys())

    if method == "hard_min":
        # Take minimum across all objectives for each sample
        aggregated = []
        for i in range(n_samples):
            sample_rewards = [rewards[obj][i] for obj in objectives]
            aggregated.append(min(sample_rewards))
        return aggregated

    elif method == "cvar":
        # CVaR: average of worst alpha-fraction across objectives
        aggregated = []
        for i in range(n_samples):
            sample_rewards = [rewards[obj][i] for obj in objectives]
            sample_rewards.sort()
            k = max(1, int(len(sample_rewards) * cvar_alpha))
            aggregated.append(np.mean(sample_rewards[:k]))
        return aggregated

    elif method == "weighted_min":
        # Weighted minimum
        if weights is None:
            weights = {obj: 1.0 / len(objectives) for obj in objectives}

        aggregated = []
        for i in range(n_samples):
            weighted_rewards = [rewards[obj][i] * weights[obj] for obj in objectives]
            aggregated.append(min(weighted_rewards))
        return aggregated

    else:
        raise ValueError(f"Unknown pessimism method: {method}")


class PessimisticDPOTrainer(DPOTrainer):
    """
    Custom DPO Trainer that uses pessimistic ensemble of reward models.
    """

    def __init__(self, reward_models, pessimism_method="hard_min", cvar_alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_models = reward_models
        self.pessimism_method = pessimism_method
        self.cvar_alpha = cvar_alpha

    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        """
        Compute pessimistic rewards for texts.
        Override this if using reward model evaluation during training.
        """
        device = next(self.model.parameters()).device

        # Get rewards from all models
        ensemble_rewards = compute_ensemble_rewards(
            self.reward_models,
            self.tokenizer,
            texts,
            device,
            max_length=self.max_length
        )

        # Apply pessimism
        aggregated_rewards = apply_pessimism(
            ensemble_rewards,
            method=self.pessimism_method,
            cvar_alpha=self.cvar_alpha
        )

        return torch.tensor(aggregated_rewards, device=device)


def prepare_policy_dataset(approach: str, max_samples: Optional[int] = None):
    """
    Load and combine preference data for policy training.
    Can use all objectives combined or a specific subset.
    """
    print(f"\n[*] Loading policy training data (approach: {approach})...")

    datasets = []
    objectives = ['helpful', 'harmless', 'honest']

    for objective in objectives:
        data_path = f"./data/processed/{approach}/{objective}"
        dataset = load_from_disk(data_path)
        datasets.append(dataset['train'])
        print(f"  {objective}: {len(dataset['train'])} examples")

    # Combine all objectives
    combined = concatenate_datasets(datasets)
    print(f"  Combined: {len(combined)} examples")

    if max_samples and len(combined) > max_samples:
        indices = np.random.choice(len(combined), max_samples, replace=False)
        combined = combined.select(indices)
        print(f"  Sampled to: {len(combined)} examples")

    return combined


def train_policy_with_dpo(
    method: str = "baseline",
    pessimism_type: str = "hard_min",
    approach: str = "hh_truthfulqa"
):
    """
    Train policy using DPO.

    Args:
        method: "baseline" (standard DPO) or "pessimistic" (with ensemble RMs)
        pessimism_type: "hard_min", "cvar", "hierarchical"
        approach: Data approach
    """
    print("\n" + "=" * 80)
    print(f"Training Policy with DPO")
    print(f"Method: {method}")
    if method == "pessimistic":
        print(f"Pessimism Type: {pessimism_type}")
    print(f"Approach: {approach}")
    print("=" * 80)

    # Load config
    config = load_config()
    set_seed(config['training']['seed'])

    # Setup device
    device = get_device()

    # Load tokenizer and model
    model_name = config['model']['name']
    print(f"\n[*] Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=config['model']['cache_dir']
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=config['model']['cache_dir'],
        torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float32
    )

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=config['model']['cache_dir'],
        torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float32
    )

    print_model_size(model)
    print_gpu_utilization()

    # Load reward models if pessimistic
    reward_models = None
    if method == "pessimistic":
        reward_models = load_reward_models(config, device)

    # Load dataset
    max_samples = config['data'].get('max_samples', 50000)
    train_dataset = prepare_policy_dataset(approach, max_samples)

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = f"{method}_{pessimism_type}" if method == "pessimistic" else method
    output_dir = f"./models/policy_models/{method_name}_{timestamp}"

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
        logging_dir=f"{config['training']['logging_dir']}/{method_name}_{timestamp}",
        report_to=config['training']['report_to'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=5,
        seed=config['training']['seed'],
        run_name=f"{method_name}_{approach}",
        remove_unused_columns=False,
    )

    # Initialize trainer
    print("\n[*] Initializing DPO Trainer...")

    dpo_config = config['dpo']

    if method == "baseline":
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=dpo_config['beta'],
            max_length=config['training']['max_length'],
            max_prompt_length=config['training']['max_prompt_length'],
        )
    else:
        trainer = PessimisticDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=dpo_config['beta'],
            max_length=config['training']['max_length'],
            max_prompt_length=config['training']['max_prompt_length'],
            reward_models=reward_models,
            pessimism_method=pessimism_type,
            cvar_alpha=dpo_config['pessimism'].get('cvar_alpha', 0.1),
        )

    # Train
    print("\n[*] Starting training...")
    print("=" * 80)

    train_result = trainer.train()

    print("\n[*] Training complete!")
    print("=" * 80)

    # Save final model
    final_model_path = f"./models/policy_models/{method_name}"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n[*] Saved final model to {final_model_path}")

    # Save results
    results = {
        'method': method,
        'pessimism_type': pessimism_type if method == "pessimistic" else None,
        'approach': approach,
        'model_name': model_name,
        'output_dir': output_dir,
        'final_model_path': final_model_path,
        'train_metrics': {
            'train_loss': train_result.training_loss,
            'train_steps': train_result.global_step,
        },
        'config': config['training']
    }

    save_step_results(f'train_policy_{method_name}', results, approach)

    print("\n" + "=" * 80)
    print(f"[OK] Policy training complete!")
    print("=" * 80)
    print(f"\nModel saved to: {final_model_path}")
    print(f"TensorBoard logs: {training_args.logging_dir}")

    return final_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train policy with DPO")
    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=['baseline', 'pessimistic'],
        help="Training method"
    )
    parser.add_argument(
        "--pessimism",
        type=str,
        default="hard_min",
        choices=['hard_min', 'cvar', 'hierarchical'],
        help="Pessimism aggregation method (for pessimistic mode)"
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="hh_truthfulqa",
        choices=['hh_truthfulqa', 'pku_safe'],
        help="Data approach"
    )

    args = parser.parse_args()

    train_policy_with_dpo(args.method, args.pessimism, args.approach)
