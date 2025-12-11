"""
Setup script for AT-PRMD project.
Runs initial setup and verification.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    print(f"Running: {command}\n")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("\n" + "=" * 80)
    print("AT-PRMD Project Setup")
    print("=" * 80)

    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("⚠️  Warning: Python 3.10+ recommended")

    # Run utilities to check CUDA and create directories
    print("\n📋 Step 1: Verifying system setup...")
    success = run_command(
        f"{sys.executable} utils.py",
        "System Verification"
    )

    if not success:
        print("\n❌ Setup failed during system verification!")
        return

    # Instructions for manual PyTorch installation
    print("\n" + "=" * 80)
    print("📦 PyTorch Installation")
    print("=" * 80)
    print("\nPlease install PyTorch 2.7 with CUDA 12.8 support manually:")
    print("\nRun this command in your terminal:")
    print("  pip install torch==2.7.0 torchvision==0.20.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128")
    print("\nThen run:")
    print("  pip install -r requirements.txt")

    print("\n" + "=" * 80)
    print("✅ Setup script completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Install PyTorch with CUDA (see above)")
    print("  2. Install requirements: pip install -r requirements.txt")
    print("  3. Run data preparation: python 1_data_preparation/download_dataset.py")
    print("\nSee README.md for detailed instructions.")


if __name__ == "__main__":
    main()
