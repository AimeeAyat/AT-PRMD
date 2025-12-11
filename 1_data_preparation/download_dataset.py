"""
Download datasets for AT-PRMD project.
Supports two approaches:
- Approach A: HH-RLHF + TruthfulQA
- Approach B: PKU-SafeRLHF
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from typing import Dict
import yaml

sys.path.append(str(Path(__file__).parent.parent))


def load_config(config_path: str = "configs/reward_model_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_hh_rlhf(cache_dir: str) -> Dict:
    """Download Anthropic HH-RLHF dataset with all subsets."""
    print("\n" + "=" * 80)
    print("Downloading Anthropic HH-RLHF Dataset")
    print("=" * 80)

    # HH-RLHF has separate subsets that need to be loaded individually
    subsets_to_load = [
        'helpful-base',
        'helpful-online',
        'helpful-rejection-sampled',
        'harmless-base'
    ]

    all_subsets = {}

    for subset_name in subsets_to_load:
        print(f"\n[*] Loading {subset_name}...")
        try:
            subset_data = load_dataset("Anthropic/hh-rlhf", data_dir=subset_name, cache_dir=cache_dir)
            all_subsets[subset_name] = subset_data
            print(f"  [OK] {subset_name}: {len(subset_data['train'])} train, {len(subset_data['test'])} test")
        except Exception as e:
            print(f"  [ERROR] Failed to load {subset_name}: {e}")

    print(f"\n[OK] HH-RLHF dataset downloaded!")
    print(f"Loaded {len(all_subsets)} subsets")

    return all_subsets


def download_truthful_qa(cache_dir: str) -> Dict:
    """Download TruthfulQA dataset for honesty objective."""
    print("\n" + "=" * 80)
    print("Downloading TruthfulQA Dataset")
    print("=" * 80)

    dataset = load_dataset("truthful_qa", "generation", cache_dir=cache_dir)

    print(f"\n[OK] TruthfulQA dataset downloaded!")
    print(f"Available splits: {list(dataset.keys())}")

    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])} examples")

    return dataset


def download_pku_safe_rlhf(cache_dir: str) -> Dict:
    """Download PKU-SafeRLHF dataset."""
    print("\n" + "=" * 80)
    print("Downloading PKU-SafeRLHF Dataset")
    print("=" * 80)

    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", cache_dir=cache_dir)

    print(f"\n[OK] PKU-SafeRLHF dataset downloaded!")
    print(f"Available splits: {list(dataset.keys())}")

    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])} examples")

    # Show available annotations
    if 'train' in dataset:
        print(f"\nFeatures: {dataset['train'].features}")

    return dataset


def inspect_hh_rlhf(dataset: Dict):
    """Inspect HH-RLHF dataset structure."""
    print("\n" + "=" * 80)
    print("HH-RLHF Dataset Inspection")
    print("=" * 80)

    # Count examples per subset
    subset_counts = {}
    for split_name in dataset.keys():
        subset_counts[split_name] = len(dataset[split_name])

    print("\nDataset breakdown:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count:,} examples")

    # Show example
    if 'train' in dataset:
        print("\nExample from 'train' split:")
        example = dataset['train'][0]
        print(f"  Chosen (first 200 chars): {example['chosen'][:200]}...")
        print(f"  Rejected (first 200 chars): {example['rejected'][:200]}...")


def inspect_truthful_qa(dataset: Dict):
    """Inspect TruthfulQA dataset structure."""
    print("\n" + "=" * 80)
    print("TruthfulQA Dataset Inspection")
    print("=" * 80)

    validation_data = dataset['validation']
    print(f"\nTotal examples: {len(validation_data)}")
    print(f"Features: {validation_data.features.keys()}")

    # Show example
    print("\nExample:")
    example = validation_data[0]
    print(f"  Question: {example['question']}")
    print(f"  Best answer: {example['best_answer']}")
    print(f"  Correct answers: {example['correct_answers'][:2]}")
    print(f"  Incorrect answers: {example['incorrect_answers'][:2]}")


def inspect_pku_safe(dataset: Dict):
    """Inspect PKU-SafeRLHF dataset structure."""
    print("\n" + "=" * 80)
    print("PKU-SafeRLHF Dataset Inspection")
    print("=" * 80)

    train_data = dataset['train']
    print(f"\nTotal examples: {len(train_data)}")
    print(f"Features: {list(train_data.features.keys())}")

    # Show example
    print("\nExample:")
    example = train_data[0]
    print(f"  Prompt: {example['prompt'][:200]}...")
    print(f"  Response 0: {example['response_0'][:150]}...")
    print(f"  Response 1: {example['response_1'][:150]}...")

    # Show annotation keys
    annotation_keys = [k for k in example.keys() if 'better' in k.lower() or 'safer' in k.lower()]
    print(f"\nAnnotation fields: {annotation_keys}")


def save_datasets(approach: str, datasets: Dict, output_dir: str = "./data/raw"):
    """Save datasets locally."""
    print(f"\n[*] Saving datasets for approach: {approach}...")
    os.makedirs(output_dir, exist_ok=True)

    if approach == "hh_truthfulqa":
        # Save HH-RLHF subsets (each subset is a separate DatasetDict)
        hh_base_path = os.path.join(output_dir, "hh_rlhf")
        os.makedirs(hh_base_path, exist_ok=True)

        for subset_name, subset_data in datasets['hh_rlhf'].items():
            subset_path = os.path.join(hh_base_path, subset_name)
            subset_data.save_to_disk(subset_path)
            print(f"  [OK] Saved HH-RLHF {subset_name} to {subset_path}")

        # Save TruthfulQA
        tqa_path = os.path.join(output_dir, "truthful_qa")
        datasets['truthful_qa'].save_to_disk(tqa_path)
        print(f"  [OK] Saved TruthfulQA to {tqa_path}")

    elif approach == "pku_safe":
        # Save PKU-SafeRLHF
        pku_path = os.path.join(output_dir, "pku_safe_rlhf")
        datasets['pku_safe'].save_to_disk(pku_path)
        print(f"  [OK] Saved PKU-SafeRLHF to {pku_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Step 1: Data Preparation - Download Datasets")
    print("=" * 80)

    # Load config
    config = load_config()
    approach = config['data']['approach']
    cache_dir = config['data']['cache_dir']

    print(f"\n[*] Selected approach: {approach}")

    datasets = {}

    if approach == "hh_truthfulqa":
        print("\n[*] Downloading datasets for Approach A: HH-RLHF + TruthfulQA")

        # Download HH-RLHF
        hh_dataset = download_hh_rlhf(cache_dir)
        inspect_hh_rlhf(hh_dataset)
        datasets['hh_rlhf'] = hh_dataset

        # Download TruthfulQA
        tqa_dataset = download_truthful_qa(cache_dir)
        inspect_truthful_qa(tqa_dataset)
        datasets['truthful_qa'] = tqa_dataset

    elif approach == "pku_safe":
        print("\n[*] Downloading datasets for Approach B: PKU-SafeRLHF")

        # Download PKU-SafeRLHF
        pku_dataset = download_pku_safe_rlhf(cache_dir)
        inspect_pku_safe(pku_dataset)
        datasets['pku_safe'] = pku_dataset

    else:
        print(f"\n[ERROR] Unknown approach: {approach}")
        print("Valid options: 'hh_truthfulqa' or 'pku_safe'")
        return

    # Save datasets
    save_datasets(approach, datasets)

    print("\n" + "=" * 80)
    print("[OK] Data download complete!")
    print("=" * 80)
    print(f"\nApproach: {approach}")
    print("Next step: Run split_objectives.py to process datasets")


if __name__ == "__main__":
    main()
