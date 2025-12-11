"""
Train the HELPFUL reward model.
"""

from train_reward_model import train_reward_model
import yaml


def main():
    # Load config to get approach
    with open("configs/reward_model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    approach = config['data']['approach']

    print("Training HELPFUL Reward Model")
    print(f"Using approach: {approach}")

    train_reward_model(objective='helpful', approach=approach)


if __name__ == "__main__":
    main()
