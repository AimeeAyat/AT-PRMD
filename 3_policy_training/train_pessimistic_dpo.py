"""
Train pessimistic DPO policy with ensemble reward models.
"""

from train_policy_dpo import train_policy_with_dpo
import yaml


def main():
    # Load config to get approach and pessimism method
    with open("configs/policy_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Get approach from data config
    with open("configs/reward_model_config.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    approach = data_config['data']['approach']
    pessimism_method = config['dpo']['pessimism']['method']

    print("Training PESSIMISTIC DPO Policy")
    print(f"Using approach: {approach}")
    print(f"Pessimism method: {pessimism_method}")

    train_policy_with_dpo(
        method='pessimistic',
        pessimism_type=pessimism_method,
        approach=approach
    )


if __name__ == "__main__":
    main()
