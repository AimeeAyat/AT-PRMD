# Dual-Approach Implementation Guide

## Overview

This project supports **TWO independent approaches** for training reward models, allowing for rigorous comparison:

### Approach A (Primary): HH-RLHF + TruthfulQA
- **2 Reward Models** from HH-RLHF (helpful, harmless)
- **1 Reward Model** from TruthfulQA (honest)
- **Total**: 3 reward models

### Approach B (Comparison): PKU-SafeRLHF
- **3 Reward Models** with native multi-dimensional annotations
- All objectives (helpful, harmless, honest) from single dataset
- **Total**: 3 reward models

---

## Approach A: HH-RLHF + TruthfulQA (PRIMARY)

### Rationale
- Uses **clean, human-annotated data** from well-established datasets
- HH-RLHF has explicit helpful/harmless splits - no keyword guessing needed
- TruthfulQA provides high-quality truthfulness data
- Different data sources = more diverse reward signals

### Data Sources

#### 1. Helpful Reward Model
**Dataset**: Anthropic HH-RLHF
**Subsets**:
- `helpful-base` (~43k examples)
- `helpful-online` (~22k examples)
- `helpful-rejection-sampled` (~9k examples)

**Total**: ~74k examples → Sample to 80k max

**Format**:
```python
{
    'chosen': "Human: <prompt>\n\nAssistant: <helpful response>",
    'rejected': "Human: <prompt>\n\nAssistant: <less helpful response>"
}
```

#### 2. Harmless Reward Model
**Dataset**: Anthropic HH-RLHF
**Subsets**:
- `harmless-base` (~42k examples)

**Total**: ~42k examples

**Format**:
```python
{
    'chosen': "Human: <prompt>\n\nAssistant: <safe response>",
    'rejected': "Human: <prompt>\n\nAssistant: <unsafe response>"
}
```

#### 3. Honest Reward Model
**Dataset**: TruthfulQA (generation subset)
**Source**: 817 questions with correct/incorrect answers

**Conversion Process**:
1. Take each question from TruthfulQA
2. Create preference pairs:
   - **Chosen**: Question + Correct answer (from `best_answer` or `correct_answers`)
   - **Rejected**: Question + Incorrect answer (from `incorrect_answers`)
3. Generate up to 3 pairs per question (3 different incorrect answers)
4. Total: ~2,400 pairs → Can augment to 30k if needed

**Format**:
```python
{
    'chosen': "Human: <question>\n\nAssistant: <truthful answer>",
    'rejected': "Human: <question>\n\nAssistant: <false/misleading answer>"
}
```

### Pros
- High data quality (human-annotated)
- Well-established datasets used in major papers
- Clear objective separation
- TruthfulQA specifically designed for honesty

### Cons
- TruthfulQA is smaller (~800 examples)
- Different data distributions between datasets
- Need conversion for TruthfulQA

---

## Approach B: PKU-SafeRLHF (COMPARISON)

### Rationale
- **Native multi-objective annotations** - no conversion needed
- All objectives from same data distribution
- Larger dataset (330k+ examples)
- Research-backed (PKU-Alignment team)

### Data Source

**Dataset**: PKU-Alignment/PKU-SafeRLHF
**Size**: ~330k preference pairs with multi-dimensional annotations

**Annotations**:
```python
{
    'prompt': "<user query>",
    'response_0': "<first response>",
    'response_1': "<second response>",
    'is_response_0_helpful': bool,
    'is_response_1_helpful': bool,
    'safer_response_id': 0 or 1,  # Which is safer
    'better_response_id': 0 or 1   # Overall better
}
```

### Objective Extraction

#### 1. Helpful Reward Model
**Criteria**: `is_response_0_helpful` and `is_response_1_helpful`
- If response_0 helpful AND response_1 not helpful → Use response_0 as chosen
- If response_1 helpful AND response_0 not helpful → Use response_1 as chosen
- Skip if both or neither are helpful

#### 2. Harmless Reward Model
**Criteria**: `safer_response_id`
- `safer_response_id = 0` → response_0 is chosen
- `safer_response_id = 1` → response_1 is chosen

#### 3. Honest Reward Model
**Criteria**: `better_response_id` (overall quality)
- `better_response_id = 0` → response_0 is chosen
- `better_response_id = 1` → response_1 is chosen
- Assumes "better" includes truthfulness

### Pros
- Single data distribution across all objectives
- Large dataset size
- Native annotations (no conversion)
- Multi-dimensional ratings available

### Cons
- "Honest" is implicit (from overall "better" rating)
- Newer dataset (less established than HH-RLHF)
- Potential annotation bias/noise

---

## Switching Between Approaches

### Configuration
Edit `configs/reward_model_config.yaml`:

```yaml
data:
  # Choose approach: "hh_truthfulqa" or "pku_safe"
  approach: "hh_truthfulqa"  # Change this line
```

### Running Approach A
```bash
# 1. Set config
# In configs/reward_model_config.yaml: approach: "hh_truthfulqa"

# 2. Download datasets
python 1_data_preparation/download_dataset.py

# 3. Process datasets
python 1_data_preparation/split_objectives.py

# Outputs:
# ./data/raw/hh_rlhf/
# ./data/raw/truthful_qa/
# ./data/processed/hh_truthfulqa/helpful/
# ./data/processed/hh_truthfulqa/harmless/
# ./data/processed/hh_truthfulqa/honest/
```

### Running Approach B
```bash
# 1. Set config
# In configs/reward_model_config.yaml: approach: "pku_safe"

# 2. Download datasets
python 1_data_preparation/download_dataset.py

# 3. Process datasets
python 1_data_preparation/split_objectives.py

# Outputs:
# ./data/raw/pku_safe_rlhf/
# ./data/processed/pku_safe/helpful/
# ./data/processed/pku_safe/harmless/
# ./data/processed/pku_safe/honest/
```

---

## Comparison Experiment Plan

### Phase 1: Complete Approach A (Now)
1. Train 3 reward models on HH-RLHF + TruthfulQA
2. Train pessimistic DPO policy
3. Evaluate on benchmarks

### Phase 2: Run Approach B (Later)
1. Change config to `approach: "pku_safe"`
2. Download and process PKU-SafeRLHF
3. Train 3 reward models on PKU data
4. Train pessimistic DPO policy (same settings)
5. Evaluate on benchmarks

### Phase 3: Comparison
Compare both approaches on:
- **Win rates** against reference
- **Per-objective performance**
- **Worst-case robustness**
- **Benchmark scores** (MT-Bench, TruthfulQA, etc.)
- **Training stability**

---

## Expected Dataset Sizes

### Approach A: HH-RLHF + TruthfulQA

| Objective | Train | Validation | Test | Total |
|-----------|-------|------------|------|-------|
| Helpful   | 72k   | 4k         | 4k   | 80k   |
| Harmless  | 37.8k | 2.1k       | 2.1k | 42k   |
| Honest    | 27k   | 1.5k       | 1.5k | 30k   |

### Approach B: PKU-SafeRLHF

| Objective | Train | Validation | Test | Total |
|-----------|-------|------------|------|-------|
| Helpful   | 45k   | 2.5k       | 2.5k | 50k   |
| Harmless  | 45k   | 2.5k       | 2.5k | 50k   |
| Honest    | 45k   | 2.5k       | 2.5k | 50k   |

Note: Actual sizes depend on annotation overlap in PKU dataset

---

## Recommended Workflow

### For Complete Implementation (Current Task)
**Use Approach A**: HH-RLHF + TruthfulQA
- Complete all steps with this approach first
- Well-established datasets
- Clear results to compare against literature

### For Future Experiments
**Switch to Approach B**: PKU-SafeRLHF
- Easy to switch (just change config)
- Provides comparison data
- Tests robustness to different data sources

---

## Key Differences Summary

| Aspect | Approach A | Approach B |
|--------|------------|------------|
| Data Sources | 2 datasets | 1 dataset |
| Annotation Quality | High (human) | High (human) |
| Honest Data | TruthfulQA (explicit) | PKU (implicit) |
| Dataset Size | Mixed (large + small) | Uniform (large) |
| Data Distribution | Different per objective | Same across objectives |
| Conversion Needed | Yes (TruthfulQA) | No |
| Established | Very (HH-RLHF) | Moderate (PKU) |

---

## Files Modified

1. **configs/reward_model_config.yaml**: Added dual approach support
2. **1_data_preparation/download_dataset.py**: Downloads both approaches
3. **1_data_preparation/split_objectives.py**: Processes both approaches

All reward model training scripts (Step 2+) will automatically work with either approach - they just load from `./data/processed/<approach>/` directory.

---

**Current Status**: Ready to run Approach A (HH-RLHF + TruthfulQA)

**To Start**: Run `python 1_data_preparation/download_dataset.py`
