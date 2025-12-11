"""
Compare performance across:
1. Base (untrained) Qwen model
2. Baseline DPO policy
3. Pessimistic DPO policy

Shows improvement from alignment training and benefits of pessimistic approach.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

sys.path.append(str(Path(__file__).parent.parent))
from evaluate_base_model import (
    load_reward_models,
    generate_responses,
    evaluate_with_reward_models,
    load_config
)
from utils import get_device
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk


def load_trained_policy(model_path: str, device: torch.device):
    """Load a trained policy model."""
    print(f"[*] Loading policy from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    return model, tokenizer


def load_previous_results(approach: str) -> Dict:
    """Load base model evaluation results if available."""
    pattern = f"./outputs/step_results/evaluate_base_model_{approach}_*.json"
    files = glob(pattern)

    if files:
        # Load most recent
        latest = max(files, key=os.path.getctime)
        with open(latest, 'r') as f:
            return json.load(f)
    return None


def compare_all_models(prompts: List[str], approach: str = "hh_truthfulqa"):
    """
    Evaluate and compare all models.

    Returns:
        Dict with results for each model type
    """
    print("\n" + "=" * 80)
    print("Comparing Models on HHH Objectives")
    print("=" * 80)

    rm_config, policy_config = load_config()
    device = get_device()

    # Load reward models
    reward_models = load_reward_models(device)

    results = {}

    # 1. Base model (load fresh or use cached results)
    print("\n[1/3] Evaluating BASE model...")
    base_results = load_previous_results(approach)

    if base_results:
        print("  [*] Using cached base model results")
        results['base'] = base_results['per_objective_stats']
        results['base_worst_case'] = base_results['worst_case_stats']
    else:
        print("  [*] No cached results, evaluating base model...")
        model_name = rm_config['model']['name']
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=rm_config['model']['cache_dir'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=rm_config['model']['cache_dir'],
            torch_dtype=torch.bfloat16
        )
        base_model.to(device)

        responses = generate_responses(base_model, tokenizer, prompts, device=device)
        rewards = evaluate_with_reward_models(responses, reward_models, tokenizer, device)

        results['base'] = {obj: {'mean': float(np.mean(scores))} for obj, scores in rewards.items()}
        worst_case = [min([rewards[obj][i] for obj in rewards.keys()]) for i in range(len(responses))]
        results['base_worst_case'] = {'mean': float(np.mean(worst_case))}

    # 2. Baseline DPO
    baseline_path = "./models/policy_models/baseline"
    if os.path.exists(baseline_path):
        print("\n[2/3] Evaluating BASELINE DPO model...")
        baseline_model, baseline_tokenizer = load_trained_policy(baseline_path, device)

        responses = generate_responses(baseline_model, baseline_tokenizer, prompts, device=device)
        rewards = evaluate_with_reward_models(responses, reward_models, baseline_tokenizer, device)

        results['baseline'] = {obj: {'mean': float(np.mean(scores))} for obj, scores in rewards.items()}
        worst_case = [min([rewards[obj][i] for obj in rewards.keys()]) for i in range(len(responses))]
        results['baseline_worst_case'] = {'mean': float(np.mean(worst_case))}
    else:
        print(f"\n[2/3] Baseline DPO model not found at {baseline_path}")
        results['baseline'] = None

    # 3. Pessimistic DPO
    pessimistic_path = None
    for method in ['hard_min', 'cvar', 'hierarchical']:
        path = f"./models/policy_models/pessimistic_{method}"
        if os.path.exists(path):
            pessimistic_path = path
            break

    if pessimistic_path:
        print(f"\n[3/3] Evaluating PESSIMISTIC DPO model from {pessimistic_path}...")
        pessimistic_model, pessimistic_tokenizer = load_trained_policy(pessimistic_path, device)

        responses = generate_responses(pessimistic_model, pessimistic_tokenizer, prompts, device=device)
        rewards = evaluate_with_reward_models(responses, reward_models, pessimistic_tokenizer, device)

        results['pessimistic'] = {obj: {'mean': float(np.mean(scores))} for obj, scores in rewards.items()}
        worst_case = [min([rewards[obj][i] for obj in rewards.keys()]) for i in range(len(responses))]
        results['pessimistic_worst_case'] = {'mean': float(np.mean(worst_case))}
    else:
        print("\n[3/3] Pessimistic DPO model not found")
        results['pessimistic'] = None

    return results


def visualize_comparison(results: Dict, output_dir: str = "./outputs/evaluation"):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    objectives = ['helpful', 'harmless', 'honest']
    models = ['base', 'baseline', 'pessimistic']
    model_labels = ['Base\n(Unaligned)', 'Baseline DPO', 'Pessimistic DPO']
    colors = ['#95a5a6', '#3498db', '#2ecc71']

    # 1. Per-objective comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, obj in enumerate(objectives):
        ax = axes[idx]

        means = []
        available_models = []
        available_colors = []

        for model, label, color in zip(models, model_labels, colors):
            if results.get(model) and obj in results[model]:
                means.append(results[model][obj]['mean'])
                available_models.append(label)
                available_colors.append(color)

        bars = ax.bar(range(len(available_models)), means, color=available_colors, alpha=0.8, edgecolor='black')

        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels(available_models, fontsize=10)
        ax.set_ylabel('Mean Reward Score', fontsize=11)
        ax.set_title(f'{obj.capitalize()}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Model Comparison Across HHH Objectives', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'model_comparison_per_objective.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved comparison to {filepath}")

    # 2. Worst-case comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    worst_case_means = []
    available_models = []
    available_colors = []

    for model, label, color in zip(models, model_labels, colors):
        if results.get(f'{model}_worst_case'):
            worst_case_means.append(results[f'{model}_worst_case']['mean'])
            available_models.append(label)
            available_colors.append(color)

    bars = ax.bar(range(len(available_models)), worst_case_means, color=available_colors, alpha=0.8, edgecolor='black', width=0.6)

    ax.set_xticks(range(len(available_models)))
    ax.set_xticklabels(available_models, fontsize=12)
    ax.set_ylabel('Mean Worst-Case Reward', fontsize=13)
    ax.set_title('Worst-Case Performance Comparison\n(Min reward across HHH per sample)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, worst_case_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'worst_case_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved worst-case comparison to {filepath}")

    # 3. Improvement heatmap
    if results.get('base') and (results.get('baseline') or results.get('pessimistic')):
        fig, ax = plt.subplots(figsize=(10, 6))

        improvement_data = []
        row_labels = []

        for model in ['baseline', 'pessimistic']:
            if results.get(model):
                improvements = []
                for obj in objectives:
                    if obj in results['base'] and obj in results[model]:
                        base_score = results['base'][obj]['mean']
                        model_score = results[model][obj]['mean']
                        improvement = ((model_score - base_score) / abs(base_score)) * 100 if base_score != 0 else 0
                        improvements.append(improvement)
                    else:
                        improvements.append(0)

                improvement_data.append(improvements)
                row_labels.append('Baseline DPO' if model == 'baseline' else 'Pessimistic DPO')

        if improvement_data:
            im = ax.imshow(improvement_data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=150)

            ax.set_xticks(range(len(objectives)))
            ax.set_xticklabels([obj.capitalize() for obj in objectives], fontsize=11)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=11)

            # Add text annotations
            for i in range(len(row_labels)):
                for j in range(len(objectives)):
                    text = ax.text(j, i, f'{improvement_data[i][j]:.1f}%',
                                 ha="center", va="center", color="black", fontsize=11, fontweight='bold')

            ax.set_title('Improvement Over Base Model (%)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Improvement (%)')
            plt.tight_layout()

            filepath = os.path.join(output_dir, 'improvement_heatmap.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[*] Saved improvement heatmap to {filepath}")


def main():
    """Main comparison function."""
    print("\n" + "=" * 80)
    print("Model Comparison: Base vs Baseline DPO vs Pessimistic DPO")
    print("=" * 80)

    rm_config, _ = load_config()
    approach = rm_config['data']['approach']

    # Load test prompts
    print("\n[*] Loading test prompts...")
    all_prompts = []

    for objective in ['helpful', 'harmless', 'honest']:
        data_path = f"./data/processed/{approach}/{objective}"
        dataset = load_from_disk(data_path)
        test_data = dataset['test']

        prompts = []
        for example in test_data.select(range(min(50, len(test_data)))):  # 50 per objective
            text = example['chosen']
            if "Assistant:" in text:
                prompt = text.split("Assistant:")[0].strip() + "\n\nAssistant:"
            else:
                prompt = text[:200]
            prompts.append(prompt)

        all_prompts.extend(prompts)

    print(f"  Total test prompts: {len(all_prompts)}")

    # Compare models
    results = compare_all_models(all_prompts, approach)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for model in ['base', 'baseline', 'pessimistic']:
        if results.get(model):
            print(f"\n{model.upper()}:")
            for obj in ['helpful', 'harmless', 'honest']:
                if obj in results[model]:
                    print(f"  {obj.capitalize():12s}: {results[model][obj]['mean']:.4f}")

            if results.get(f'{model}_worst_case'):
                print(f"  {'Worst-Case':12s}: {results[f'{model}_worst_case']['mean']:.4f}")

    # Visualize
    visualize_comparison(results)

    # Save results
    from visualization import save_step_results
    save_step_results('compare_models', results, approach)

    print("\n" + "=" * 80)
    print("[OK] Model comparison complete!")
    print("=" * 80)
    print("\nVisualizations saved to: ./outputs/evaluation/")


if __name__ == "__main__":
    main()
