"""
Split datasets by objectives for reward model training.
Supports two approaches:
- Approach A: HH-RLHF (helpful, harmless) + TruthfulQA (honest)
- Approach B: PKU-SafeRLHF (native 3-objective annotations)
"""

import os
import sys
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from typing import Dict
import yaml
import numpy as np
from tqdm import tqdm
import random

sys.path.append(str(Path(__file__).parent.parent))
from visualization import DatasetVisualizer, save_step_results


def load_config(config_path: str = "configs/reward_model_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# Approach A: HH-RLHF + TruthfulQA
# ============================================================================

def process_hh_rlhf_helpful(hh_dataset: Dict, config: Dict) -> DatasetDict:
    """Extract helpful examples from HH-RLHF."""
    print("\n[*] Processing HH-RLHF for HELPFUL objective...")

    helpful_config = config['data']['hh_truthfulqa']['helpful']
    subsets = helpful_config['subsets']
    max_samples = helpful_config['max_samples']

    # Combine specified subsets
    all_examples = []
    for subset in subsets:
        if subset in hh_dataset:
            # Each subset is a DatasetDict with 'train' and 'test' splits
            train_examples = list(hh_dataset[subset]['train'])
            test_examples = list(hh_dataset[subset]['test'])
            subset_examples = train_examples + test_examples
            print(f"  {subset}: {len(subset_examples):,} examples (train: {len(train_examples)}, test: {len(test_examples)})")
            all_examples.extend(subset_examples)

    print(f"  Total helpful examples: {len(all_examples):,}")

    # Shuffle and limit
    random.shuffle(all_examples)
    if len(all_examples) > max_samples:
        all_examples = all_examples[:max_samples]
        print(f"  Sampled down to: {len(all_examples):,}")

    # Create dataset
    dataset = Dataset.from_list(all_examples)
    return dataset


def process_hh_rlhf_harmless(hh_dataset: Dict, config: Dict) -> DatasetDict:
    """Extract harmless examples from HH-RLHF."""
    print("\n[*] Processing HH-RLHF for HARMLESS objective...")

    harmless_config = config['data']['hh_truthfulqa']['harmless']
    subsets = harmless_config['subsets']
    max_samples = harmless_config['max_samples']

    # Combine specified subsets
    all_examples = []
    for subset in subsets:
        if subset in hh_dataset:
            # Each subset is a DatasetDict with 'train' and 'test' splits
            train_examples = list(hh_dataset[subset]['train'])
            test_examples = list(hh_dataset[subset]['test'])
            subset_examples = train_examples + test_examples
            print(f"  {subset}: {len(subset_examples):,} examples (train: {len(train_examples)}, test: {len(test_examples)})")
            all_examples.extend(subset_examples)

    print(f"  Total harmless examples: {len(all_examples):,}")

    # Shuffle and limit
    random.shuffle(all_examples)
    if len(all_examples) > max_samples:
        all_examples = all_examples[:max_samples]
        print(f"  Sampled down to: {len(all_examples):,}")

    # Create dataset
    dataset = Dataset.from_list(all_examples)
    return dataset


def convert_truthful_qa_to_preferences(tqa_dataset: Dict, config: Dict) -> Dataset:
    """
    Convert TruthfulQA to preference pairs for honest objective.

    Format:
    - chosen: truthful answer (correct)
    - rejected: false/misleading answer (incorrect)
    """
    print("\n[*] Converting TruthfulQA to preference pairs for HONEST objective...")

    honest_config = config['data']['hh_truthfulqa']['honest']
    max_samples = honest_config['max_samples']

    validation_data = tqa_dataset['validation']

    preference_pairs = []

    for example in tqdm(validation_data, desc="Converting TruthfulQA"):
        question = example['question']
        best_answer = example['best_answer']
        correct_answers = example['correct_answers']
        incorrect_answers = example['incorrect_answers']

        # Create multiple pairs from each question
        # Use best answer or random correct answer as "chosen"
        chosen_answer = best_answer if best_answer else random.choice(correct_answers)

        # Create pairs with incorrect answers as "rejected"
        for incorrect in incorrect_answers[:3]:  # Use up to 3 incorrect per question
            pair = {
                'chosen': f"Human: {question}\n\nAssistant: {chosen_answer}",
                'rejected': f"Human: {question}\n\nAssistant: {incorrect}"
            }
            preference_pairs.append(pair)

    print(f"  Generated {len(preference_pairs):,} preference pairs")

    # Shuffle and limit
    random.shuffle(preference_pairs)
    if len(preference_pairs) > max_samples:
        preference_pairs = preference_pairs[:max_samples]
        print(f"  Sampled down to: {len(preference_pairs):,}")

    # Create dataset
    dataset = Dataset.from_list(preference_pairs)
    return dataset


def process_approach_a(config: Dict) -> Dict[str, Dataset]:
    """Process datasets for Approach A: HH-RLHF + TruthfulQA."""
    print("\n" + "=" * 80)
    print("Processing Approach A: HH-RLHF + TruthfulQA")
    print("=" * 80)

    # Load raw datasets
    print("\n[*] Loading raw datasets...")

    # Load HH-RLHF subsets (each subset is saved separately)
    hh_base_path = "./data/raw/hh_rlhf"
    hh_dataset = {}

    subset_names = ['helpful-base', 'helpful-online', 'helpful-rejection-sampled', 'harmless-base']
    for subset_name in subset_names:
        subset_path = os.path.join(hh_base_path, subset_name)
        if os.path.exists(subset_path):
            hh_dataset[subset_name] = load_from_disk(subset_path)
            print(f"  Loaded {subset_name}")

    tqa_dataset = load_from_disk("./data/raw/truthful_qa")

    # Process each objective
    datasets = {}

    # Helpful
    datasets['helpful'] = process_hh_rlhf_helpful(hh_dataset, config)

    # Harmless
    datasets['harmless'] = process_hh_rlhf_harmless(hh_dataset, config)

    # Honest (from TruthfulQA)
    datasets['honest'] = convert_truthful_qa_to_preferences(tqa_dataset, config)

    return datasets


# ============================================================================
# Approach B: PKU-SafeRLHF
# ============================================================================

def process_pku_safe_rlhf(config: Dict) -> Dict[str, Dataset]:
    """
    Process PKU-SafeRLHF dataset for 3 objectives.

    PKU-SafeRLHF has native annotations for:
    - Helpfulness (is_response_0_helpful, is_response_1_helpful)
    - Harmlessness (safer_response_id)
    - Honesty/Truthfulness (implicit in overall better response)
    """
    print("\n" + "=" * 80)
    print("Processing Approach B: PKU-SafeRLHF")
    print("=" * 80)

    # Load raw dataset
    print("\n[*] Loading PKU-SafeRLHF dataset...")
    pku_dataset = load_from_disk("./data/raw/pku_safe_rlhf")

    train_data = pku_dataset['train']

    pku_config = config['data']['pku_safe']
    max_samples = pku_config['max_samples_per_objective']

    # Initialize objective datasets
    objective_data = {
        'helpful': [],
        'harmless': [],
        'honest': []
    }

    print("\n[*] Extracting preference pairs by objective...")

    for example in tqdm(train_data, desc="Processing PKU-SafeRLHF"):
        prompt = example['prompt']
        response_0 = example['response_0']
        response_1 = example['response_1']

        # Helpfulness
        if 'is_response_0_helpful' in example and 'is_response_1_helpful' in example:
            helpful_0 = example['is_response_0_helpful']
            helpful_1 = example['is_response_1_helpful']

            if helpful_0 and not helpful_1:
                objective_data['helpful'].append({
                    'chosen': f"Human: {prompt}\n\nAssistant: {response_0}",
                    'rejected': f"Human: {prompt}\n\nAssistant: {response_1}"
                })
            elif helpful_1 and not helpful_0:
                objective_data['helpful'].append({
                    'chosen': f"Human: {prompt}\n\nAssistant: {response_1}",
                    'rejected': f"Human: {prompt}\n\nAssistant: {response_0}"
                })

        # Harmlessness (safety)
        if 'safer_response_id' in example:
            safer_id = example['safer_response_id']
            if safer_id == 0:
                objective_data['harmless'].append({
                    'chosen': f"Human: {prompt}\n\nAssistant: {response_0}",
                    'rejected': f"Human: {prompt}\n\nAssistant: {response_1}"
                })
            elif safer_id == 1:
                objective_data['harmless'].append({
                    'chosen': f"Human: {prompt}\n\nAssistant: {response_1}",
                    'rejected': f"Human: {prompt}\n\nAssistant: {response_0}"
                })

        # Honesty/Overall better
        if 'better_response_id' in example:
            better_id = example['better_response_id']
            if better_id == 0:
                objective_data['honest'].append({
                    'chosen': f"Human: {prompt}\n\nAssistant: {response_0}",
                    'rejected': f"Human: {prompt}\n\nAssistant: {response_1}"
                })
            elif better_id == 1:
                objective_data['honest'].append({
                    'chosen': f"Human: {prompt}\n\nAssistant: {response_1}",
                    'rejected': f"Human: {prompt}\n\nAssistant: {response_0}"
                })

    # Print statistics
    print("\n[*] Dataset statistics:")
    for obj, data in objective_data.items():
        print(f"  {obj}: {len(data):,} examples")

    # Sample and create datasets
    datasets = {}
    for obj, data in objective_data.items():
        random.shuffle(data)
        if len(data) > max_samples:
            data = data[:max_samples]
            print(f"  {obj} sampled to: {len(data):,}")

        datasets[obj] = Dataset.from_list(data)

    return datasets


# ============================================================================
# Common: Train/Val/Test Split
# ============================================================================

def create_splits(datasets: Dict[str, Dataset], val_percentage: float = 0.1) -> Dict[str, DatasetDict]:
    """Create train/validation/test splits for each objective."""
    print("\n[*] Creating train/validation/test splits...")

    split_datasets = {}

    for objective, dataset in datasets.items():
        # Split into train/temp (90/10)
        split1 = dataset.train_test_split(test_size=val_percentage, seed=42)
        # Split temp into val/test (50/50)
        split2 = split1['test'].train_test_split(test_size=0.5, seed=42)

        split_datasets[objective] = DatasetDict({
            'train': split1['train'],
            'validation': split2['train'],
            'test': split2['test']
        })

        print(f"  {objective}:")
        print(f"    Train:      {len(split1['train']):,}")
        print(f"    Validation: {len(split2['train']):,}")
        print(f"    Test:       {len(split2['test']):,}")

    return split_datasets


def save_processed_datasets(datasets: Dict[str, DatasetDict], approach: str, output_dir: str = "./data/processed"):
    """Save processed datasets to disk."""
    print(f"\n[*] Saving processed datasets for approach: {approach}...")

    approach_dir = os.path.join(output_dir, approach)
    os.makedirs(approach_dir, exist_ok=True)

    for objective, dataset_dict in datasets.items():
        obj_path = os.path.join(approach_dir, objective)
        dataset_dict.save_to_disk(obj_path)
        print(f"  [OK] Saved {objective} to {obj_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Step 1: Data Preparation - Split by Objectives")
    print("=" * 80)

    # Load config
    config = load_config()
    approach = config['data']['approach']
    val_percentage = config['data']['validation_split_percentage'] / 100

    print(f"\n[*] Selected approach: {approach}")

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Process based on approach
    if approach == "hh_truthfulqa":
        datasets = process_approach_a(config)
    elif approach == "pku_safe":
        datasets = process_pku_safe_rlhf(config)
    else:
        print(f"\n[ERROR] Unknown approach: {approach}")
        return

    # Create splits
    split_datasets = create_splits(datasets, val_percentage)

    # Save datasets
    save_processed_datasets(split_datasets, approach)

    # Visualization and Analysis
    print("\n" + "=" * 80)
    print("[*] Creating visualizations and saving results...")
    print("=" * 80)

    visualizer = DatasetVisualizer()

    # Collect statistics
    stats = {}
    text_lengths = {}
    samples = {}

    for objective, dataset_dict in split_datasets.items():
        stats[objective] = {
            'train': len(dataset_dict['train']),
            'validation': len(dataset_dict['validation']),
            'test': len(dataset_dict['test'])
        }

        # Calculate text lengths (chosen + rejected)
        lengths = []
        for example in dataset_dict['train']:
            lengths.append(len(example['chosen']) + len(example['rejected']))
        text_lengths[objective] = lengths[:1000]  # Sample 1000 for speed

        # Save 10 sample examples
        sample_indices = np.random.choice(len(dataset_dict['train']),
                                         min(10, len(dataset_dict['train'])),
                                         replace=False)
        samples[objective] = [dataset_dict['train'][int(i)] for i in sample_indices]

    # Create visualizations
    visualizer.plot_dataset_statistics(stats, approach)
    visualizer.plot_objective_distribution(stats, approach)
    visualizer.plot_text_length_distribution(text_lengths, approach)
    visualizer.save_sample_examples(samples, approach)

    # Save results to JSON
    results = {
        'approach': approach,
        'statistics': stats,
        'text_length_stats': {
            obj: {
                'mean': float(np.mean(lens)),
                'median': float(np.median(lens)),
                'min': float(np.min(lens)),
                'max': float(np.max(lens)),
                'std': float(np.std(lens))
            }
            for obj, lens in text_lengths.items()
        }
    }

    save_step_results('split_objectives', results, approach)

    print("\n" + "=" * 80)
    print("[OK] Dataset processing complete!")
    print("=" * 80)
    print(f"\nApproach: {approach}")
    print(f"Processed datasets saved to: ./data/processed/{approach}/")
    print(f"Visualizations saved to: ./outputs/visualizations/")
    print(f"Results saved to: ./outputs/step_results/")
    print("\nObjectives:")
    print("  - helpful/")
    print("  - harmless/")
    print("  - honest/")
    print("\nNext step: Train reward models (Step 2)")


if __name__ == "__main__":
    main()
