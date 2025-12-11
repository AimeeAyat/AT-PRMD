"""
Evaluate BASE (untrained) Qwen model on HHH objectives using trained reward models.
This establishes the baseline performance before any alignment training.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed
)

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_device, print_model_size
from visualization import save_step_results, DatasetVisualizer

import matplotlib.pyplot as plt
import seaborn as sns


def load_config() -> Dict:
    """Load configurations."""
    with open("configs/reward_model_config.yaml", 'r') as f:
        rm_config = yaml.safe_load(f)
    with open("configs/policy_config.yaml", 'r') as f:
        policy_config = yaml.safe_load(f)
    return rm_config, policy_config


def load_reward_models(device: torch.device) -> Dict:
    """Load all trained reward models."""
    print("\n[*] Loading reward models...")

    reward_models = {}
    objectives = ['helpful', 'harmless', 'honest']

    for objective in objectives:
        model_path = f"./models/reward_models/{objective}_rm"
        if not os.path.exists(model_path):
            print(f"  [WARNING] {objective} RM not found at {model_path}")
            continue

        print(f"  Loading {objective} RM from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        model.to(device)
        model.eval()
        reward_models[objective] = model

    return reward_models


def generate_responses(model, tokenizer, prompts: List[str], max_length: int = 256, device: torch.device = None):
    """Generate responses from model for given prompts."""
    responses = []

    model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from response
            response = response[len(prompt):].strip()
            responses.append(response)

    return responses


def evaluate_with_reward_models(
    responses: List[str],
    reward_models: Dict,
    tokenizer,
    device: torch.device
) -> Dict[str, List[float]]:
    """Evaluate responses using reward models."""
    rewards = {obj: [] for obj in reward_models.keys()}

    with torch.no_grad():
        for response in tqdm(responses, desc="Computing rewards"):
            inputs = tokenizer(
                response,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding='max_length'
            ).to(device)

            for obj, model in reward_models.items():
                reward = model(**inputs).logits[0, 0].item()
                rewards[obj].append(reward)

    return rewards


def visualize_base_performance(rewards: Dict[str, List[float]], output_dir: str = "./outputs/evaluation"):
    """Visualize base model performance across HHH objectives."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    objectives = list(rewards.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (obj, color) in enumerate(zip(objectives, colors)):
        ax = axes[idx]
        scores = rewards[obj]

        ax.hist(scores, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(scores):.3f}')

        ax.set_title(f'{obj.capitalize()} Objective', fontsize=14, fontweight='bold')
        ax.set_xlabel('Reward Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Base Qwen2.5-3B Performance on HHH Objectives (Before Training)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'base_model_hhh_performance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved visualization to {filepath}")

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [np.mean(rewards[obj]) for obj in objectives]
    stds = [np.std(rewards[obj]) for obj in objectives]

    bars = ax.bar(objectives, means, color=colors, alpha=0.7, edgecolor='black', yerr=stds, capsize=10)

    ax.set_ylabel('Mean Reward Score', fontsize=12)
    ax.set_title('Base Model: Mean Rewards Across HHH Objectives', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'base_model_hhh_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved comparison to {filepath}")


def main():
    """Main evaluation function."""
    print("\n" + "=" * 80)
    print("Evaluating BASE Qwen2.5-3B Model on HHH Objectives")
    print("=" * 80)

    # Load configs
    rm_config, policy_config = load_config()
    approach = rm_config['data']['approach']
    set_seed(42)

    # Setup
    device = get_device()
    model_name = rm_config['model']['name']

    print(f"\n[*] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=rm_config['model']['cache_dir'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=rm_config['model']['cache_dir'],
        torch_dtype=torch.bfloat16
    )
    base_model.to(device)

    print_model_size(base_model)

    # Load reward models
    reward_models = load_reward_models(device)

    if not reward_models:
        print("\n[ERROR] No reward models found! Train reward models first:")
        print("  python 2_reward_modeling/train_helpful_rm.py")
        print("  python 2_reward_modeling/train_harmless_rm.py")
        print("  python 2_reward_modeling/train_honest_rm.py")
        return

    # Load test prompts from each objective
    print(f"\n[*] Loading test prompts...")
    all_prompts = []

    for objective in ['helpful', 'harmless', 'honest']:
        data_path = f"./data/processed/{approach}/{objective}"
        dataset = load_from_disk(data_path)
        test_data = dataset['test']

        # Extract prompts (take first part before "Assistant:")
        prompts = []
        for example in test_data.select(range(min(100, len(test_data)))):  # 100 samples per objective
            text = example['chosen']
            if "Assistant:" in text:
                prompt = text.split("Assistant:")[0].strip() + "\n\nAssistant:"
            else:
                prompt = text[:200]  # Fallback
            prompts.append(prompt)

        all_prompts.extend(prompts)
        print(f"  {objective}: {len(prompts)} test prompts")

    print(f"  Total: {len(all_prompts)} prompts")

    # Generate responses
    print("\n[*] Generating responses from base model...")
    responses = generate_responses(base_model, tokenizer, all_prompts, max_length=256, device=device)

    # Evaluate with reward models
    print("\n[*] Evaluating responses with reward models...")
    rewards = evaluate_with_reward_models(responses, reward_models, tokenizer, device)

    # Compute statistics
    stats = {}
    for obj in rewards.keys():
        scores = rewards[obj]
        stats[obj] = {
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75))
        }

    print("\n[*] Base Model Performance:")
    print("=" * 80)
    for obj, stat in stats.items():
        print(f"\n{obj.upper()}:")
        print(f"  Mean:   {stat['mean']:.4f}")
        print(f"  Median: {stat['median']:.4f}")
        print(f"  Std:    {stat['std']:.4f}")
        print(f"  Range:  [{stat['min']:.4f}, {stat['max']:.4f}]")

    # Worst-case analysis
    worst_case_per_sample = []
    for i in range(len(responses)):
        sample_rewards = [rewards[obj][i] for obj in rewards.keys()]
        worst_case_per_sample.append(min(sample_rewards))

    worst_case_stats = {
        'mean': float(np.mean(worst_case_per_sample)),
        'median': float(np.median(worst_case_per_sample)),
        'std': float(np.std(worst_case_per_sample))
    }

    print(f"\nWORST-CASE (min across objectives per sample):")
    print(f"  Mean:   {worst_case_stats['mean']:.4f}")
    print(f"  Median: {worst_case_stats['median']:.4f}")

    # Visualize
    visualize_base_performance(rewards)

    # Save results
    results = {
        'model': model_name,
        'approach': approach,
        'num_samples': len(responses),
        'per_objective_stats': stats,
        'worst_case_stats': worst_case_stats,
        'sample_responses': responses[:10]  # Save first 10 for inspection
    }

    save_step_results('evaluate_base_model', results, approach)

    print("\n" + "=" * 80)
    print("[OK] Base model evaluation complete!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - JSON: ./outputs/step_results/")
    print("  - Plots: ./outputs/evaluation/")


if __name__ == "__main__":
    main()
