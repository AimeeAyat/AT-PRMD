"""
Compare all baseline models and generate comprehensive analysis.
Compares: Untrained, Standard DPO, Qwen-Instruct, and (later) AT-PRMD.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: str = "./5_baselines/results") -> Dict:
    """Load all available result JSONs."""
    results = {}

    for json_file in Path(results_dir).glob("*.json"):
        model_name = json_file.stem
        with open(json_file, 'r') as f:
            results[model_name] = json.load(f)

    print(f"[*] Loaded results for {len(results)} models:")
    for name in results.keys():
        print(f"  - {name}")

    return results


def plot_hhh_comparison(results: Dict, output_dir: str = "./5_baselines/results"):
    """Create comprehensive HHH comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    # Plot 1: Response lengths by objective
    ax = axes[0, 0]
    x = np.arange(3)
    width = 0.2

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        data = results[model_name]
        helpful_len = data.get('helpful', data.get('hh_rlhf_helpful', {})).get('avg_response_length', 0)
        harmless_len = data.get('harmless', data.get('hh_rlhf_harmless', {})).get('avg_response_length', 0)
        honest_samples = data.get('honest', data.get('truthfulqa', {})).get('num_samples', 0)

        lengths = [helpful_len, harmless_len, honest_samples * 50]  # Placeholder for honest
        ax.bar(x + i * width, lengths, width, label=model_name, color=color)

    ax.set_ylabel('Avg Response Length (chars)')
    ax.set_title('Response Lengths by Objective')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(['Helpful', 'Harmless', 'Honest'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: TruthfulQA Accuracy
    ax = axes[0, 1]
    accuracies = []
    for model_name in model_names:
        data = results[model_name]
        acc = data.get('honest', data.get('truthfulqa', {})).get('accuracy', 0)
        accuracies.append(acc * 100)

    ax.bar(model_names, accuracies, color=colors)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('TruthfulQA Accuracy Comparison')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Correct vs Incorrect breakdown
    ax = axes[1, 0]
    x = np.arange(len(model_names))
    correct_counts = []
    incorrect_counts = []

    for model_name in model_names:
        data = results[model_name]
        tqa = data.get('honest', data.get('truthfulqa', {}))
        correct_counts.append(tqa.get('correct', 0))
        incorrect_counts.append(tqa.get('incorrect', 0))

    ax.bar(x, correct_counts, label='Correct', color='#06A77D', alpha=0.8)
    ax.bar(x, incorrect_counts, bottom=correct_counts, label='Incorrect', color='#D62246', alpha=0.8)

    ax.set_ylabel('Count')
    ax.set_title('TruthfulQA: Correct vs Incorrect Responses')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Model', 'Helpful (len)', 'Harmless (len)', 'TruthfulQA (acc)'])

    for model_name in model_names:
        data = results[model_name]
        helpful_len = data.get('helpful', data.get('hh_rlhf_helpful', {})).get('avg_response_length', 0)
        harmless_len = data.get('harmless', data.get('hh_rlhf_harmless', {})).get('avg_response_length', 0)
        acc = data.get('honest', data.get('truthfulqa', {})).get('accuracy', 0)

        table_data.append([
            model_name,
            f"{helpful_len:.0f}",
            f"{harmless_len:.0f}",
            f"{acc:.2%}"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle('HHH Baseline Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Saved comparison plot to {output_path}")
    plt.close()


def generate_summary_report(results: Dict, output_dir: str = "./5_baselines/results"):
    """Generate text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BASELINE COMPARISON SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")

    for model_name, data in results.items():
        report_lines.append(f"\n{model_name.upper()}")
        report_lines.append("-" * 80)

        # Helpful
        helpful = data.get('helpful', data.get('hh_rlhf_helpful', {}))
        report_lines.append(f"  Helpful:")
        report_lines.append(f"    Samples: {helpful.get('num_samples', 0)}")
        report_lines.append(f"    Avg Response Length: {helpful.get('avg_response_length', 0):.1f} chars")

        # Harmless
        harmless = data.get('harmless', data.get('hh_rlhf_harmless', {}))
        report_lines.append(f"  Harmless:")
        report_lines.append(f"    Samples: {harmless.get('num_samples', 0)}")
        report_lines.append(f"    Avg Response Length: {harmless.get('avg_response_length', 0):.1f} chars")

        # Honest
        honest = data.get('honest', data.get('truthfulqa', {}))
        report_lines.append(f"  TruthfulQA:")
        report_lines.append(f"    Samples: {honest.get('num_samples', 0)}")
        report_lines.append(f"    Correct: {honest.get('correct', 0)}")
        report_lines.append(f"    Incorrect: {honest.get('incorrect', 0)}")
        report_lines.append(f"    Accuracy: {honest.get('accuracy', 0):.2%}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file
    output_path = os.path.join(output_dir, 'baseline_summary.txt')
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n[OK] Summary report saved to {output_path}")


def main():
    """Main comparison pipeline."""
    print("\n" + "=" * 80)
    print("Baseline Comparison")
    print("=" * 80)

    results_dir = "./5_baselines/results"

    # Load all results
    results = load_results(results_dir)

    if not results:
        print("\n[ERROR] No results found. Run evaluation scripts first.")
        return

    # Generate visualizations
    plot_hhh_comparison(results, results_dir)

    # Generate summary report
    generate_summary_report(results, results_dir)

    print("\n" + "=" * 80)
    print("[OK] Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
