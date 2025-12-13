"""
General evaluation script for any model checkpoint.
Can evaluate: untrained, standard DPO, Qwen-Instruct, AT-PRMD, etc.
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
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from utils import set_seed, get_device


def load_model_and_tokenizer(model_path: str, use_instruct: bool = False):
    """Load model and tokenizer from path or HF hub."""
    print(f"\n[*] Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    print(f"[OK] Model loaded")
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

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
    if prompt in response:
        response = response.replace(prompt, '').strip()

    return response


def evaluate_on_hh_rlhf(model, tokenizer, objective: str, num_samples: int = 100) -> Dict:
    """Evaluate on HH-RLHF test set."""
    print(f"\n[*] Evaluating on {objective}...")

    dataset_path = f"./data/processed/hh_truthfulqa/{objective}"
    dataset = load_from_disk(dataset_path)
    test_data = dataset['test']

    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)

    responses = []
    for idx in tqdm(indices, desc=f"{objective}"):
        example = test_data[int(idx)]

        # Extract prompt
        prompt = example['chosen'].split('\n\nAssistant:')[0] + '\n\nAssistant:'

        # Generate
        response = generate_response(model, tokenizer, prompt)

        responses.append({
            'prompt': prompt,
            'response': response,
            'chosen': example['chosen'],
            'rejected': example['rejected']
        })

    # Metrics
    avg_length = np.mean([len(r['response']) for r in responses])

    return {
        'objective': objective,
        'num_samples': len(responses),
        'avg_response_length': avg_length,
        'samples': responses[:10]
    }


def evaluate_truthfulqa(model, tokenizer, num_samples: int = 100) -> Dict:
    """Evaluate on TruthfulQA."""
    print(f"\n[*] Evaluating on TruthfulQA...")

    dataset = load_from_disk("./data/raw/truthful_qa")
    validation_data = dataset['validation']

    indices = np.random.choice(len(validation_data), min(num_samples, len(validation_data)), replace=False)

    correct = 0
    incorrect = 0
    responses = []

    for idx in tqdm(indices, desc="TruthfulQA"):
        example = validation_data[int(idx)]

        question = example['question']
        correct_answers = example['correct_answers']
        incorrect_answers = example['incorrect_answers']

        prompt = f"Human: {question}\n\nAssistant:"
        response = generate_response(model, tokenizer, prompt, max_new_tokens=128)

        # Simple heuristic check
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

    return {
        'num_samples': len(responses),
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'samples': responses[:10]
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on HHH benchmarks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model or HF model name')
    parser.add_argument('--output_name', type=str, required=True, help='Name for output files')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples per benchmark')
    parser.add_argument('--use_instruct', action='store_true', help='Use instruct variant')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"Evaluating Model: {args.output_name}")
    print("=" * 80)

    # Setup
    set_seed(42)
    device = get_device()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_instruct)

    # Evaluate
    results = {}
    results['model_path'] = args.model_path
    results['model_name'] = args.output_name

    results['helpful'] = evaluate_on_hh_rlhf(model, tokenizer, 'helpful', args.num_samples)
    results['harmless'] = evaluate_on_hh_rlhf(model, tokenizer, 'harmless', args.num_samples)
    results['honest'] = evaluate_truthfulqa(model, tokenizer, args.num_samples)

    # Save results
    output_dir = "./5_baselines/results"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {args.output_name}")
    print("=" * 80)
    print(f"Helpful - Avg Length: {results['helpful']['avg_response_length']:.1f} chars")
    print(f"Harmless - Avg Length: {results['harmless']['avg_response_length']:.1f} chars")
    print(f"TruthfulQA Accuracy: {results['honest']['accuracy']:.2%}")
    print(f"  Correct: {results['honest']['correct']}")
    print(f"  Incorrect: {results['honest']['incorrect']}")


if __name__ == "__main__":
    main()
