"""
Run the complete AT-PRMD pipeline.
Executes all steps sequentially with user confirmations.
"""

import os
import subprocess
import sys
from datetime import datetime


def run_step(description: str, command: str, wait_for_approval: bool = False):
    """Run a pipeline step with proper logging."""
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)

    if wait_for_approval:
        response = input("\nProceed with this step? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping step...")
            return False

    print(f"\nRunning: {command}\n")

    start_time = datetime.now()

    result = subprocess.run(command, shell=True)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[*] Step completed in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")

    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with code {result.returncode}")
        response = input("Continue with remaining steps? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)

    return True


def main():
    """Run full pipeline."""
    print("\n" + "=" * 80)
    print("AT-PRMD: Complete Training Pipeline")
    print("=" * 80)
    print("\nThis will run:")
    print("  1. Data download & preparation")
    print("  2. Train 3 reward models (helpful, harmless, honest)")
    print("  3. Evaluate base model performance")
    print("  4. Train baseline DPO policy")
    print("  5. Train pessimistic DPO policy")
    print("  6. Compare all models")
    print("\nEstimated time: 15-25 hours on RTX 5090")

    response = input("\nStart pipeline? [y/N]: ")
    if response.lower() != 'y':
        print("Pipeline cancelled")
        return

    start_time = datetime.now()

    # Step 1: Data Preparation
    run_step(
        "1. Download datasets",
        f"{sys.executable} 1_data_preparation/download_dataset.py",
        wait_for_approval=False
    )

    run_step(
        "2. Split datasets by objectives",
        f"{sys.executable} 1_data_preparation/split_objectives.py",
        wait_for_approval=False
    )

    # Step 2: Train Reward Models
    run_step(
        "3. Train HELPFUL reward model (~2-4 hours)",
        f"{sys.executable} 2_reward_modeling/train_helpful_rm.py",
        wait_for_approval=True
    )

    run_step(
        "4. Train HARMLESS reward model (~2-4 hours)",
        f"{sys.executable} 2_reward_modeling/train_harmless_rm.py",
        wait_for_approval=True
    )

    run_step(
        "5. Train HONEST reward model (~2-4 hours)",
        f"{sys.executable} 2_reward_modeling/train_honest_rm.py",
        wait_for_approval=True
    )

    # Step 3: Evaluate Base Model
    run_step(
        "6. Evaluate base (untrained) model",
        f"{sys.executable} 4_evaluation/evaluate_base_model.py",
        wait_for_approval=True
    )

    # Step 4: Train Policies
    run_step(
        "7. Train BASELINE DPO policy (~3-5 hours)",
        f"{sys.executable} 3_policy_training/train_baseline_dpo.py",
        wait_for_approval=True
    )

    run_step(
        "8. Train PESSIMISTIC DPO policy (~3-5 hours)",
        f"{sys.executable} 3_policy_training/train_pessimistic_dpo.py",
        wait_for_approval=True
    )

    # Step 5: Final Comparison
    run_step(
        "9. Compare all models",
        f"{sys.executable} 4_evaluation/compare_models.py",
        wait_for_approval=True
    )

    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nTotal time: {total_time/3600:.1f} hours")
    print("\nResults saved to:")
    print("  - Models: ./models/")
    print("  - Visualizations: ./outputs/")
    print("  - Logs: ./logs/")
    print("\nView training progress:")
    print("  tensorboard --logdir logs/")


if __name__ == "__main__":
    main()
