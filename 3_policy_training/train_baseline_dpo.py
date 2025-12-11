"""
Train baseline DPO policy (no pessimism, standard DPO).
"""

from train_policy_dpo import train_policy_with_dpo
import yaml


def main():
    # Load config to get approach
    with open("configs/policy_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Get approach from data config
    with open("configs/reward_model_config.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    approach = data_config['data']['approach']

    print("Training BASELINE DPO Policy")
    print(f"Using approach: {approach}")

    train_policy_with_dpo(method='baseline', approach=approach)


if __name__ == "__main__":
    main()
