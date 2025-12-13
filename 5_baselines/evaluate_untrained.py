"""
Evaluate untrained Qwen2.5-3B base model on HHH benchmarks.
Provides baseline performance before any alignment training.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from utils import set_seed, get_device
from visualization import DatasetVisualizer, save_step_results


def load_base_model(model_name: str = "Qwen/Qwen2.5-3B"):
    """Load untrained Qwen2.5-3B base model."""
    print(f"\n[*] Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    print(f"[OK] Model loaded")
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response


def evaluate_on_hh_rlhf(model, tokenizer, objective: str, num_samples: int = 100) -> Dict:
    """Evaluate on HH-RLHF test set for given objective."""
    print(f"\n[*] Evaluating on HH-RLHF {objective} test set...")

    # Load test dataset
    dataset_path = f"./data/processed/hh_truthfulqa/{objective}"
    dataset = load_from_disk(dataset_path)
    test_data = dataset['test']

    # Sample random examples
    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)

    responses = []
    for idx in tqdm(indices, desc=f"Generating {objective} responses"):
        example = test_data[int(idx)]

        # Extract prompt from chosen example
        prompt = example['chosen'].split('\n\nAssistant:')[0] + '\n\nAssistant:'

        # Generate response
        response = generate_response(model, tokenizer, prompt)

        responses.append({
            'prompt': prompt,
            'response': response,
            'chosen': example['chosen'],
            'rejected': example['rejected']
        })

    # Calculate basic metrics
    avg_response_length = np.mean([len(r['response']) for r in responses])

    results = {
        'objective': objective,
        'num_samples': len(responses),
        'avg_response_length': avg_response_length,
        'samples': responses[:10]  # Save first 10 for inspection
    }

    return results


def evaluate_truthfulness(model, tokenizer, num_samples: int = 100) -> Dict:
    """Evaluate on TruthfulQA for honesty/truthfulness."""
    print(f"\n[*] Evaluating on TruthfulQA...")

    # Load TruthfulQA
    dataset = load_from_disk("./data/raw/truthful_qa")
    validation_data = dataset['validation']

    # Sample random examples
    indices = np.random.choice(len(validation_data), min(num_samples, len(validation_data)), replace=False)

    correct = 0
    incorrect = 0
    responses = []

    for idx in tqdm(indices, desc="Evaluating truthfulness"):
        example = validation_data[int(idx)]

        question = example['question']
        correct_answers = example['correct_answers']
        incorrect_answers = example['incorrect_answers']

        prompt = f"Human: {question}\n\nAssistant:"
        response = generate_response(model, tokenizer, prompt, max_new_tokens=128)

        # Simple heuristic: check if response contains correct vs incorrect patterns
        contains_correct = any(ans.lower() in response.lower() for ans in correct_answers)
        contains_incorrect = any(ans.lower() in response.lower() for ans in incorrect_answers)

        if contains_correct and not contains_incorrect:
            correct += 1
        elif contains_incorrect and not contains_correct:
            incorrect += 1

        responses.append({
            'question': question,
            'response': response,
            'correct_answers': correct_answers[:3],
            'incorrect_answers': incorrect_answers[:3]
        })

    accuracy = correct / len(responses) if len(responses) > 0 else 0

    results = {
        'num_samples': len(responses),
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'samples': responses[:10]
    }

    return results


def plot_baseline_results(results: Dict, output_dir: str = "./5_baselines/results"):
    """Generate visualization plots for baseline results."""
    os.makedirs(output_dir, exist_ok=True)

    # HHH scores comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Response lengths by objective
    objectives = ['helpful', 'harmless', 'honest']
    lengths = [
        results['hh_rlhf_helpful']['avg_response_length'],
        results['hh_rlhf_harmless']['avg_response_length'],
        results['truthfulqa']['num_samples'] * 50  # Placeholder
    ]

    ax1.bar(objectives, lengths, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_ylabel('Average Response Length (chars)')
    ax1.set_title('Untrained Model: Response Lengths by Objective')
    ax1.grid(axis='y', alpha=0.3)

    # TruthfulQA accuracy
    tqa_results = results['truthfulqa']
    categories = ['Correct', 'Incorrect', 'Ambiguous']
    counts = [
        tqa_results['correct'],
        tqa_results['incorrect'],
        tqa_results['num_samples'] - tqa_results['correct'] - tqa_results['incorrect']
    ]

    ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=['#06A77D', '#D62246', '#999999'])
    ax2.set_title(f'TruthfulQA Results (Accuracy: {tqa_results["accuracy"]:.2%})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'untrained_baseline.png'), dpi=300, bbox_inches='tight')
    print(f"[*] Saved plot to {output_dir}/untrained_baseline.png")
    plt.close()


def main():
    """Main evaluation pipeline for untrained model."""
    print("\n" + "=" * 80)
    print("Baseline Evaluation: Untrained Qwen2.5-3B")
    print("=" * 80)

    # Setup
    set_seed(42)
    device = get_device()

    # Load model
    model, tokenizer = load_base_model()

    # Evaluate on HH-RLHF
    results = {}
    results['hh_rlhf_helpful'] = evaluate_on_hh_rlhf(model, tokenizer, 'helpful', num_samples=100)
    results['hh_rlhf_harmless'] = evaluate_on_hh_rlhf(model, tokenizer, 'harmless', num_samples=100)

    # Evaluate on TruthfulQA
    results['truthfulqa'] = evaluate_truthfulness(model, tokenizer, num_samples=100)

    # Save results
    output_dir = "./5_baselines/results"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'untrained_baseline.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_dir}/untrained_baseline.json")

    # Generate plots
    plot_baseline_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Untrained Qwen2.5-3B Baseline")
    print("=" * 80)
    print(f"Helpful - Avg Response Length: {results['hh_rlhf_helpful']['avg_response_length']:.1f} chars")
    print(f"Harmless - Avg Response Length: {results['hh_rlhf_harmless']['avg_response_length']:.1f} chars")
    print(f"TruthfulQA Accuracy: {results['truthfulqa']['accuracy']:.2%}")
    print(f"  Correct: {results['truthfulqa']['correct']}")
    print(f"  Incorrect: {results['truthfulqa']['incorrect']}")
    print("\nNext step: Train standard DPO baseline")


if __name__ == "__main__":
    main()
