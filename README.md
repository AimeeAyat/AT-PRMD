# AT-PRMD: Robust Multi-Objective Alignment via Pessimistic Reward Model Distillation

Implementation of "Robust Multi-Objective Alignment via Pessimistic Reward Model Distillation" using TRL and PyTorch 2.7.

## 🎯 Project Overview

This project implements a pessimistic ensemble approach to reward model distillation for language model alignment. The goal is to train models that balance multiple competing objectives (helpfulness, harmlessness, honesty) without falling into degenerate solutions.

### Key Components
- **3 Reward Models**: One per objective (helpful, harmless, honest)
- **Base Model**: Qwen2.5-3B-Instruct
- **Dataset**: Anthropic HH-RLHF (~170k preference pairs)
- **Method**: Pessimistic DPO with ensemble reward models

## 🔧 System Requirements

- **GPU**: RTX 5090 (32GB VRAM) or similar
- **CUDA**: 12.8
- **PyTorch**: 2.7.0
- **OS**: Windows (current setup)
- **Python**: 3.10+

## 📦 Installation

### Step 1: Install PyTorch 2.7 with CUDA 12.8

```bash
# Install PyTorch with CUDA 12.8 support
pip install torch==2.7.0 torchvision==0.20.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Setup

```bash
python utils.py
```

This will check your CUDA setup and create necessary directories.

## 📊 Project Structure

```
ATML_PROJ_IMP/
├── 1_data_preparation/          # Data download and preprocessing
│   ├── download_dataset.py      # Download HH-RLHF dataset
│   └── split_objectives.py      # Split by objectives
├── 2_reward_modeling/           # Reward model training
│   ├── train_helpful_rm.py      # Train helpful reward model
│   ├── train_harmless_rm.py     # Train harmless reward model
│   └── train_honest_rm.py       # Train honest reward model
├── 3_policy_training/           # Policy training with pessimistic DPO
│   ├── baseline_dpo.py          # Baseline DPO
│   ├── pessimistic_dpo.py       # Pessimistic DPO
│   └── hierarchical_dpo.py      # Hierarchical pessimistic DPO
├── 4_evaluation/                # Evaluation scripts
│   ├── compute_metrics.py       # Compute alignment metrics
│   └── benchmark_eval.py        # Run benchmarks
├── configs/                     # Configuration files
│   ├── reward_model_config.yaml
│   └── policy_config.yaml
├── data/                        # Data storage
│   ├── raw/                     # Raw downloaded data
│   ├── processed/               # Processed objective-specific data
│   └── cache/                   # HuggingFace cache
├── models/                      # Model storage
│   ├── reward_models/           # Trained reward models
│   └── policy_models/           # Trained policy models
├── logs/                        # Training logs
└── outputs/                     # Evaluation outputs
```

## 🚀 Usage

### Phase 1: Data Preparation

```bash
# Download the dataset
python 1_data_preparation/download_dataset.py

# Split by objectives
python 1_data_preparation/split_objectives.py
```

### Phase 2: Train Reward Models

```bash
# Train helpful reward model
python 2_reward_modeling/train_helpful_rm.py

# Train harmless reward model
python 2_reward_modeling/train_harmless_rm.py

# Train honest reward model
python 2_reward_modeling/train_honest_rm.py
```

### Phase 3: Train Policy Models

```bash
# Baseline DPO (for comparison)
python 3_policy_training/baseline_dpo.py

# Pessimistic DPO (main method)
python 3_policy_training/pessimistic_dpo.py
```

### Phase 4: Evaluation

```bash
# Compute metrics
python 4_evaluation/compute_metrics.py

# Run benchmarks
python 4_evaluation/benchmark_eval.py
```

## 🔬 Experiment Configurations

### Reward Model Training
- **Learning Rate**: 1e-5
- **Batch Size**: 4 (per device) × 8 (gradient accumulation) = 32 effective
- **Epochs**: 3
- **Max Length**: 512 tokens
- **Precision**: BF16

### Policy Training
- **Learning Rate**: 5e-7
- **Batch Size**: 2 (per device) × 16 (gradient accumulation) = 32 effective
- **Beta (KL penalty)**: 0.1
- **Max Length**: 512 tokens
- **Precision**: BF16

### Pessimism Methods
1. **Hard Minimum**: Take worst-case reward across ensemble
2. **CVaR-10%**: Average worst 10% of rewards
3. **Hierarchical**: Worst within objectives, then worst across

## 📈 Evaluation Metrics

- **Win Rate Against Reference**: Preference-based evaluation
- **Per-Objective Performance**: Separate scores for each objective
- **Worst-Case Performance**: Minimum across all objectives
- **KL Divergence**: Distance from reference policy
- **Benchmarks**:
  - HH-RLHF Holdout
  - MT-Bench
  - TruthfulQA
  - RealToxicityPrompts
  - JailbreakBench

## 🔧 Configuration

Edit `configs/reward_model_config.yaml` and `configs/policy_config.yaml` to adjust hyperparameters.

## 📝 Logging

- **TensorBoard**: `tensorboard --logdir logs/`
- **Weights & Biases**: Configure in YAML files

## ⚠️ Notes

- Each reward model training takes ~2-4 hours on RTX 5090
- Policy training takes ~3-5 hours
- Total dataset size: ~10GB
- Model checkpoints: ~6GB per reward model

## 📚 References

- Paper: "Robust Multi-Objective Alignment via Pessimistic Reward Model Distillation"
- Dataset: [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- Library: [TRL](https://huggingface.co/docs/trl/)
- Model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## 🤝 Contributing

This is a research implementation. Feel free to modify and experiment!

## 📊 Dual-Approach Support

This implementation supports **two independent approaches** for comprehensive comparison:

### Approach A (Primary): HH-RLHF + TruthfulQA
- Helpful & Harmless from Anthropic HH-RLHF (clean, human-annotated)
- Honest from TruthfulQA (converted to preference pairs)
- 3 reward models with diverse data sources

### Approach B (Comparison): PKU-SafeRLHF
- All 3 objectives from PKU-SafeRLHF with native annotations
- Single data source, uniform distribution
- 3 reward models from same dataset

**Switch approaches** by editing `configs/reward_model_config.yaml`:
```yaml
data:
  approach: "hh_truthfulqa"  # or "pku_safe"
```

See [DUAL_APPROACH_GUIDE.md](DUAL_APPROACH_GUIDE.md) for detailed comparison.

## 📈 Visualization & Logging

Every step automatically generates:

### JSON Results
- Step results saved to `./outputs/step_results/`
- Timestamped for tracking experiments
- Includes all metrics and statistics

### Visualizations
1. **Dataset Statistics**: Train/val/test sizes per objective
2. **Text Length Distributions**: Character counts across objectives
3. **Sample Examples**: 10 random examples per objective (JSON + readable text)
4. **Reward Distributions**: Chosen vs rejected scores
5. **Reward Margins**: Score differences analysis
6. **Top/Bottom Examples**: Best and worst performing samples

Saved to: `./outputs/visualizations/` and `./outputs/reward_analysis/`

### TensorBoard Logging
```bash
# View training progress
tensorboard --logdir logs/
```

Tracks:
- Loss curves
- Accuracy
- Learning rate
- Gradient norms
- Custom metrics

### Checkpointing Strategy
- **5 checkpoints saved**: Start, 33%, 66%, End, Best
- **Best model** automatically loaded at end
- Checkpoints in: `./models/reward_models/<objective>_rm/`

## 📄 License

Research and educational use.
