"""
Test script to verify AT-PRMD setup.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "=" * 80)
    print("Testing Package Imports")
    print("=" * 80)

    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("accelerate", "Accelerate"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]

    failed = []
    for package_name, display_name in packages:
        try:
            __import__(package_name)
            print(f"[OK] {display_name}")
        except ImportError as e:
            print(f"[FAIL] {display_name} - {e}")
            failed.append(display_name)

    if failed:
        print(f"\n[FAIL] Failed to import: {', '.join(failed)}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All required packages imported successfully!")
        return True


def test_cuda():
    """Test CUDA availability."""
    print("\n" + "=" * 80)
    print("Testing CUDA Setup")
    print("=" * 80)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("[OK] CUDA setup successful!")
            return True
        else:
            print("[WARNING] CUDA not available - training will be slow")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False


def test_directories():
    """Test that all required directories exist."""
    print("\n" + "=" * 80)
    print("Testing Directory Structure")
    print("=" * 80)

    required_dirs = [
        "1_data_preparation",
        "2_reward_modeling",
        "3_policy_training",
        "4_evaluation",
        "configs",
        "data",
        "models",
        "logs",
        "outputs",
    ]

    all_exist = True
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"[OK] {dir_name}/")
        else:
            print(f"[FAIL] {dir_name}/ - Missing!")
            all_exist = False

    if all_exist:
        print("\n[OK] All directories exist!")
    else:
        print("\n[FAIL] Some directories are missing. Run: python utils.py")

    return all_exist


def test_configs():
    """Test that configuration files exist and are valid."""
    print("\n" + "=" * 80)
    print("Testing Configuration Files")
    print("=" * 80)

    import yaml

    configs = [
        "configs/reward_model_config.yaml",
        "configs/policy_config.yaml",
    ]

    all_valid = True
    for config_path in configs:
        path = Path(config_path)
        if not path.exists():
            print(f"[FAIL] {config_path} - Missing!")
            all_valid = False
            continue

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] {config_path} - Valid YAML")
        except Exception as e:
            print(f"[FAIL] {config_path} - Invalid: {e}")
            all_valid = False

    if all_valid:
        print("\n[OK] All configuration files valid!")
    else:
        print("\n[FAIL] Some configuration files are invalid!")

    return all_valid


def test_scripts():
    """Test that all main scripts exist."""
    print("\n" + "=" * 80)
    print("Testing Script Files")
    print("=" * 80)

    scripts = [
        "1_data_preparation/download_dataset.py",
        "1_data_preparation/split_objectives.py",
        "utils.py",
        "setup.py",
    ]

    all_exist = True
    for script_path in scripts:
        path = Path(script_path)
        if path.exists():
            print(f"[OK] {script_path}")
        else:
            print(f"[FAIL] {script_path} - Missing!")
            all_exist = False

    if all_exist:
        print("\n[OK] All scripts exist!")
    else:
        print("\n[FAIL] Some scripts are missing!")

    return all_exist


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AT-PRMD Setup Verification")
    print("=" * 80)

    results = {
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Directories": test_directories(),
        "Configs": test_configs(),
        "Scripts": test_scripts(),
    }

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("[OK] All tests passed! Setup is complete.")
        print("\nYou can now run:")
        print("  python 1_data_preparation/download_dataset.py")
    else:
        print("[FAIL] Some tests failed. Please fix the issues above.")
        print("\nMake sure to:")
        print("  1. Install PyTorch with CUDA")
        print("  2. Install requirements: pip install -r requirements.txt")
        print("  3. Run: python utils.py (to create directories)")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
