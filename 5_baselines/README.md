# Baseline Evaluations

This directory contains scripts for evaluating baseline models to compare against AT-PRMD.

## Directory Structure

```
5_baselines/
├── evaluate_untrained.py      # Evaluate untrained Qwen2.5-3B
├── train_standard_dpo.py      # Train standard TRL DPO (Option A)
├── evaluate_model.py          # General evaluation script
├── compare_baselines.py       # Generate comparison plots
├── run_all_baselines.sh       # Master script to run all evaluations
├── models/                    # Saved baseline models
│   └── standard_dpo/
└── results/                   # Evaluation results and plots
    ├── untrained_baseline.json
    ├── standard_dpo.json
    ├── qwen_instruct.json
    └── baseline_comparison.png
```

## Baseline Models

### 1. Untrained Qwen2.5-3B
- **Purpose:** Show pre-training performance without any alignment
- **Script:** `evaluate_untrained.py`
- **Metrics:** HHH response quality, TruthfulQA accuracy

### 2. Standard DPO (Option A)
- **Purpose:** Fair comparison - same data, standard method
- **Script:** `train_standard_dpo.py` + `evaluate_model.py`
- **Training:** TRL DPOTrainer on our 42K helpful + 42K harmless dataset
- **Time:** ~3-5 hours

### 3. Qwen2.5-3B-Instruct (Option B)
- **Purpose:** Real-world reference - production aligned model
- **Script:** `evaluate_model.py`
- **Model:** Pre-trained checkpoint from HuggingFace

## Running Evaluations

### Option 1: Run All (Sequential)
```bash
# Evaluate untrained model
python 5_baselines/evaluate_untrained.py

# Train standard DPO
python 5_baselines/train_standard_dpo.py

# Evaluate standard DPO
python 5_baselines/evaluate_model.py --model_path ./5_baselines/models/standard_dpo/final --output_name standard_dpo

# Evaluate Qwen-Instruct
python 5_baselines/evaluate_model.py --model_path Qwen/Qwen2.5-3B-Instruct --output_name qwen_instruct --use_instruct

# Generate comparison
python 5_baselines/compare_baselines.py
```

### Option 2: Individual Scripts

**Untrained Evaluation:**
```bash
python 5_baselines/evaluate_untrained.py
```

**Train Standard DPO:**
```bash
python 5_baselines/train_standard_dpo.py
```

**Evaluate Any Model:**
```bash
python 5_baselines/evaluate_model.py --model_path <path> --output_name <name>
```

**Compare All:**
```bash
python 5_baselines/compare_baselines.py
```

## Evaluation Metrics

### Per-Objective Metrics
- **Helpful:** Average response length, relevance
- **Harmless:** Average response length, safety
- **Honest:** TruthfulQA accuracy, correct/incorrect counts

### Comparison Metrics
- Response quality across HHH dimensions
- TruthfulQA accuracy comparison
- Worst-case performance (min across objectives)

## Expected Results

### Untrained Qwen2.5-3B
- Random/incoherent responses
- Low TruthfulQA accuracy (~30-40%)
- Baseline for improvement measurement

### Standard DPO
- Improved helpfulness and harmlessness
- Potential single-objective bias
- TruthfulQA accuracy ~60-70%
- **Key limitation:** May sacrifice one objective for another

### Qwen-Instruct
- Production-quality responses
- Balanced HHH performance
- TruthfulQA accuracy ~55-65%
- Reference for real-world comparison

### AT-PRMD (To be added)
- **Goal:** Better worst-case HHH scores
- Balanced multi-objective performance
- CVaR pessimism prevents collapse
- Target: Outperform standard DPO on min(H, H, H)

## Output Files

### JSON Results
- `untrained_baseline.json` - Untrained model scores
- `standard_dpo.json` - Standard DPO scores
- `qwen_instruct.json` - Qwen-Instruct scores

### Visualizations
- `untrained_baseline.png` - Untrained model plots
- `baseline_comparison.png` - Multi-model comparison
- `baseline_summary.txt` - Text summary report

## Next Steps

1. Run untrained evaluation (baseline)
2. Train standard DPO (3-5 hours)
3. Evaluate Qwen-Instruct (30 mins)
4. Compare all baselines
5. Train AT-PRMD pessimistic model
6. Add AT-PRMD to comparison
