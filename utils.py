"""
Utility functions for the AT-PRMD project.
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Optional
import yaml
import json
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the available device (CUDA if available, else CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    return device


def load_yaml_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def print_model_size(model):
    """
    Print model size information.

    Args:
        model: PyTorch model
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n📊 Model Information:")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {param_count - trainable_params:,}")

    # Estimate memory usage
    param_memory = param_count * 4 / (1024**3)  # Assuming fp32
    print(f"  Estimated memory (FP32): {param_memory:.2f} GB")
    print(f"  Estimated memory (BF16): {param_memory/2:.2f} GB")


def print_gpu_utilization():
    """Print current GPU memory utilization."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\n💾 GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved:  {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_run_name(objective: str, method: str, timestamp: Optional[str] = None) -> str:
    """
    Create a run name for logging.

    Args:
        objective: Objective name (helpful, harmless, honest)
        method: Method name (rm, dpo, pessimistic, etc.)
        timestamp: Optional timestamp string

    Returns:
        Run name string
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{objective}_{method}_{timestamp}"


def check_cuda_setup():
    """
    Verify CUDA setup and print diagnostic information.
    """
    print("\n" + "=" * 80)
    print("CUDA Setup Check")
    print("=" * 80)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi-Processors: {props.multi_processor_count}")

        # Test CUDA operations
        print("\n[*] Testing CUDA operations...")
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("[OK] CUDA operations successful!")
        except Exception as e:
            print(f"[ERROR] CUDA operations failed: {e}")
    else:
        print("\n[WARNING] CUDA not available. Training will be slow on CPU.")

    print("=" * 80)


def prepare_directories():
    """
    Create all necessary directories for the project.
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/cache",
        "models/reward_models",
        "models/policy_models",
        "models/cache",
        "outputs",
        "logs/reward_models",
        "logs/policy_models",
    ]

    print("\n[*] Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("[OK] All directories created!")


class MetricsTracker:
    """Simple metrics tracking utility."""

    def __init__(self):
        self.metrics = {}

    def add(self, name: str, value: float, step: int):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = {'steps': [], 'values': []}
        self.metrics[name]['steps'].append(step)
        self.metrics[name]['values'].append(value)

    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if name in self.metrics and self.metrics[name]['values']:
            return self.metrics[name]['values'][-1]
        return None

    def get_average(self, name: str) -> Optional[float]:
        """Get the average value for a metric."""
        if name in self.metrics and self.metrics[name]['values']:
            return np.mean(self.metrics[name]['values'])
        return None

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        save_json(self.metrics, filepath)

    def load(self, filepath: str):
        """Load metrics from JSON file."""
        self.metrics = load_json(filepath)


if __name__ == "__main__":
    # Run diagnostics when executed directly
    print("\n" + "=" * 80)
    print("AT-PRMD Utilities - System Diagnostics")
    print("=" * 80)

    check_cuda_setup()
    prepare_directories()
    set_seed(42)

    print("\n[OK] Utilities loaded successfully!")
