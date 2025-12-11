"""
Visualization and logging utilities for AT-PRMD project.
Includes dataset visualization, reward analysis, and training plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def save_json(data: Dict, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[*] Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class DatasetVisualizer:
    """Visualize dataset statistics and samples."""

    def __init__(self, output_dir: str = "./outputs/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_dataset_statistics(self, stats: Dict[str, Dict], approach: str):
        """
        Plot dataset statistics (train/val/test sizes per objective).

        Args:
            stats: {objective: {train: N, validation: M, test: K}}
            approach: Approach name for title
        """
        objectives = list(stats.keys())
        splits = ['train', 'validation', 'test']

        data = {split: [stats[obj][split] for obj in objectives] for split in splits}

        x = np.arange(len(objectives))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, split in enumerate(splits):
            ax.bar(x + i * width, data[split], width, label=split.capitalize())

        ax.set_xlabel('Objective')
        ax.set_ylabel('Number of Examples')
        ax.set_title(f'Dataset Statistics - {approach}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(objectives)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, split in enumerate(splits):
            for j, v in enumerate(data[split]):
                ax.text(j + i * width, v + max(data[split]) * 0.02,
                       f'{v:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'dataset_stats_{approach}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved plot to {filepath}")

    def plot_text_length_distribution(self, lengths: Dict[str, List[int]], approach: str):
        """
        Plot text length distributions for each objective.

        Args:
            lengths: {objective: [len1, len2, ...]}
            approach: Approach name
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, (objective, lens) in enumerate(lengths.items()):
            ax = axes[idx]
            ax.hist(lens, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(f'{objective.capitalize()} - Text Length')
            ax.set_xlabel('Character Count')
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(lens), color='red', linestyle='--',
                      label=f'Mean: {np.mean(lens):.0f}')
            ax.axvline(np.median(lens), color='green', linestyle='--',
                      label=f'Median: {np.median(lens):.0f}')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(f'Text Length Distributions - {approach}')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'text_lengths_{approach}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved plot to {filepath}")

    def plot_objective_distribution(self, stats: Dict[str, Dict], approach: str):
        """
        Plot histogram of data distribution across HHH objectives.

        Args:
            stats: {objective: {train: N, validation: M, test: K}}
            approach: Approach name
        """
        objectives = list(stats.keys())
        train_counts = [stats[obj]['train'] for obj in objectives]
        val_counts = [stats[obj]['validation'] for obj in objectives]
        test_counts = [stats[obj]['test'] for obj in objectives]
        total_counts = [t + v + te for t, v, te in zip(train_counts, val_counts, test_counts)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Total distribution
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        ax1.bar(objectives, total_counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Total Examples')
        ax1.set_title(f'Total Data Distribution - {approach}')
        ax1.grid(axis='y', alpha=0.3)

        for i, (obj, count) in enumerate(zip(objectives, total_counts)):
            ax1.text(i, count + max(total_counts) * 0.02,
                    f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Split breakdown
        x = np.arange(len(objectives))
        width = 0.25

        ax2.bar(x - width, train_counts, width, label='Train', color='#3498db', alpha=0.7, edgecolor='black')
        ax2.bar(x, val_counts, width, label='Validation', color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.bar(x + width, test_counts, width, label='Test', color='#2ecc71', alpha=0.7, edgecolor='black')

        ax2.set_ylabel('Number of Examples')
        ax2.set_title(f'Data Distribution by Split - {approach}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(objectives)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'objective_distribution_{approach}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved objective distribution plot to {filepath}")

    def save_sample_examples(self, samples: Dict[str, List[Dict]], approach: str):
        """
        Save sample examples to JSON and create readable text file.

        Args:
            samples: {objective: [{chosen: ..., rejected: ...}, ...]}
            approach: Approach name
        """
        # Save JSON
        json_path = os.path.join(self.output_dir, f'sample_examples_{approach}.json')
        save_json(samples, json_path)

        # Create readable text file
        text_path = os.path.join(self.output_dir, f'sample_examples_{approach}.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample Examples - {approach}\n")
            f.write("=" * 80 + "\n\n")

            for objective, examples in samples.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"OBJECTIVE: {objective.upper()}\n")
                f.write(f"{'=' * 80}\n\n")

                for i, example in enumerate(examples, 1):
                    f.write(f"--- Example {i} ---\n\n")
                    f.write(f"CHOSEN:\n{example['chosen']}\n\n")
                    f.write(f"REJECTED:\n{example['rejected']}\n\n")
                    f.write("-" * 80 + "\n\n")

        print(f"[*] Saved sample examples to {text_path}")


class RewardVisualizer:
    """Visualize reward model predictions and analysis."""

    def __init__(self, output_dir: str = "./outputs/reward_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_reward_distributions(self, rewards: Dict[str, List[float]], objective: str, split: str = "validation"):
        """
        Plot distribution of reward scores.

        Args:
            rewards: {type: [scores]} e.g., {chosen: [...], rejected: [...]}
            objective: Objective name
            split: Data split name
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for reward_type, scores in rewards.items():
            ax.hist(scores, bins=50, alpha=0.6, label=reward_type.capitalize(), edgecolor='black')

        ax.set_xlabel('Reward Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Reward Distribution - {objective.capitalize()} ({split})')
        ax.legend()
        ax.grid(alpha=0.3)

        filepath = os.path.join(self.output_dir, f'reward_dist_{objective}_{split}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved plot to {filepath}")

    def plot_reward_margins(self, margins: List[float], objective: str, split: str = "validation"):
        """
        Plot distribution of reward margins (chosen - rejected).

        Args:
            margins: List of margin values
            objective: Objective name
            split: Data split name
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(margins, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='Margin = 0')
        ax.axvline(np.mean(margins), color='green', linestyle='--',
                  label=f'Mean = {np.mean(margins):.3f}')

        ax.set_xlabel('Reward Margin (Chosen - Rejected)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Reward Margins - {objective.capitalize()} ({split})')
        ax.legend()
        ax.grid(alpha=0.3)

        filepath = os.path.join(self.output_dir, f'reward_margins_{objective}_{split}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved plot to {filepath}")

    def save_top_bottom_examples(self, examples: List[Dict], objective: str,
                                 split: str = "validation", n: int = 10):
        """
        Save top and bottom scoring examples.

        Args:
            examples: List of dicts with 'chosen', 'rejected', 'reward_chosen', 'reward_rejected', 'margin'
            objective: Objective name
            split: Data split name
            n: Number of top/bottom examples
        """
        # Sort by margin
        sorted_examples = sorted(examples, key=lambda x: x['margin'], reverse=True)

        top_examples = sorted_examples[:n]
        bottom_examples = sorted_examples[-n:]

        data = {
            'objective': objective,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'top_examples': top_examples,
            'bottom_examples': bottom_examples
        }

        json_path = os.path.join(self.output_dir, f'top_bottom_{objective}_{split}.json')
        save_json(data, json_path)

        # Create readable text file
        text_path = os.path.join(self.output_dir, f'top_bottom_{objective}_{split}.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Top and Bottom Examples - {objective.capitalize()} ({split})\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"\nTOP {n} EXAMPLES (Highest Margins)\n")
            f.write("=" * 80 + "\n")
            for i, ex in enumerate(top_examples, 1):
                f.write(f"\n--- Example {i} ---\n")
                f.write(f"Margin: {ex['margin']:.3f} | Chosen: {ex['reward_chosen']:.3f} | Rejected: {ex['reward_rejected']:.3f}\n\n")
                f.write(f"CHOSEN:\n{ex['chosen']}\n\n")
                f.write(f"REJECTED:\n{ex['rejected']}\n\n")
                f.write("-" * 80 + "\n")

            f.write(f"\n\nBOTTOM {n} EXAMPLES (Lowest Margins)\n")
            f.write("=" * 80 + "\n")
            for i, ex in enumerate(bottom_examples, 1):
                f.write(f"\n--- Example {i} ---\n")
                f.write(f"Margin: {ex['margin']:.3f} | Chosen: {ex['reward_chosen']:.3f} | Rejected: {ex['reward_rejected']:.3f}\n\n")
                f.write(f"CHOSEN:\n{ex['chosen']}\n\n")
                f.write(f"REJECTED:\n{ex['rejected']}\n\n")
                f.write("-" * 80 + "\n")

        print(f"[*] Saved top/bottom examples to {text_path}")


class TrainingVisualizer:
    """Visualize training metrics and progress."""

    def __init__(self, output_dir: str = "./outputs/training_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_curves(self, metrics: Dict[str, List], objective: str, model_type: str = "reward"):
        """
        Plot training curves (loss, accuracy, etc.).

        Args:
            metrics: {metric_name: [values]}
            objective: Objective name
            model_type: 'reward' or 'policy'
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.grid(alpha=0.3)

        plt.suptitle(f'Training Metrics - {objective.capitalize()} ({model_type.upper()})')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'training_{model_type}_{objective}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved training plot to {filepath}")

    def plot_multi_objective_comparison(self, metrics: Dict[str, Dict], metric_name: str = "accuracy"):
        """
        Plot comparison of a metric across multiple objectives.

        Args:
            metrics: {objective: {metric_name: [values]}}
            metric_name: Name of metric to plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for objective, obj_metrics in metrics.items():
            if metric_name in obj_metrics:
                ax.plot(obj_metrics[metric_name], label=objective.capitalize(), linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Across Objectives')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'multi_objective_{metric_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved comparison plot to {filepath}")


def save_step_results(step_name: str, results: Dict, approach: str):
    """
    Save results from a processing step to JSON.

    Args:
        step_name: Name of the step (e.g., 'data_download', 'reward_training')
        results: Dictionary of results to save
        approach: Approach name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{step_name}_{approach}_{timestamp}.json"
    filepath = os.path.join("./outputs/step_results", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    results['timestamp'] = timestamp
    results['step'] = step_name
    results['approach'] = approach

    save_json(results, filepath)
    return filepath


if __name__ == "__main__":
    print("Visualization utilities loaded")
    print("\nAvailable classes:")
    print("  - DatasetVisualizer: For dataset statistics and samples")
    print("  - RewardVisualizer: For reward analysis and distributions")
    print("  - TrainingVisualizer: For training metrics and progress")
