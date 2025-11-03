# HydraGNN GFM Fine-Tuning

Tools for fine-tuning a **single HydraGNN Graph Foundation Model (GFM)**  
(e.g., `gfm_0.229.pk`) on datasets such as **QM9**, **MD17**, and **Transition1x**.

---

## Overview

This repository provides scripts for fine-tuning Graph Foundation Models (GFM) using HydraGNN on molecular property prediction tasks. It supports:

- **Full fine-tuning**: Train both backbone and head layers
- **Frozen-backbone fine-tuning**: Train only the head while freezing the backbone
- **Head reuse**: Keep the pretrained head and only fine-tune it
- **Head from scratch**: Replace the head with a new one trained from scratch
- **Multiple datasets**: QM9, MD17 (revised uracil), and Transition1x
- **Data efficiency experiments**: Training with different dataset fractions (1%, 5%, 10%, 25%, 50%, 100%)

---

## Prerequisites

### Installation

1. **Install HydraGNN**:
   ```bash
   git clone https://github.com/ORNL/HydraGNN
   cd HydraGNN
   pip install -e .
   ```

2. **Install additional dependencies**:
   ```bash
   pip install torch torch-geometric tensorboard
   ```

3. **Download pretrained GFM models**:
   - Visit: https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024
   - Download a checkpoint (e.g., `gfm_0.229.pk`)
   - Create a `checkpoints/` directory in this repository and place the checkpoint there:
     ```bash
     mkdir -p checkpoints
     # Place your downloaded checkpoint in this directory
     ```

---

## Repository Structure

```
.
├── qm9/                          # QM9 dataset scripts
│   ├── split_and_save_qm9.py     # Create train/val/test splits
│   ├── train_qm9_scratch.py      # Train from scratch (no pretrained model)
│   ├── qm9_gfm_finetune.py       # Full fine-tuning (backbone + head)
│   ├── qm9_gfm_reuse_head.py     # Frozen backbone, train existing head
│   └── qm9_gfm_head_scratch.py   # Frozen backbone, new head from scratch
├── md17/                         # MD17 dataset scripts
│   ├── split_and_save_md17.py    # Create train/val/test splits
│   ├── train_md17_scratch.py     # Train from scratch
│   ├── md17_gfm_finetune.py      # Full fine-tuning
│   ├── md17_gfm_reuse_head.py    # Frozen backbone, train existing head
│   └── md17_gfm_head_scratch.py  # Frozen backbone, new head
└── transition1x/                 # Transition1x dataset scripts
    ├── split_and_save_transition1x.py  # Create train/val/test splits
    ├── train_transition1x_scratch.py
    ├── transition1x_gfm_finetune.py
    ├── transition1x_gfm_reuse_head.py
    └── transition1x_gfm_head_scratch.py
```

---

## Usage

### Step 1: Prepare Dataset Splits

For each dataset, first create the train/validation/test splits by running the corresponding script:

#### QM9 Dataset
```bash
cd qm9
python split_and_save_qm9.py
```

This creates a `qm9_splits/` directory with train/val/test splits and subsampled training sets.

#### MD17 Dataset
```bash
cd md17
python split_and_save_md17.py
```

This creates an `md17_splits/` directory with train/val/test splits and subsampled training sets.

#### Transition1x Dataset
```bash
cd transition1x
python split_and_save_transition1x.py
```

This creates a `transition1x_splits/` directory with train/val/test splits and subsampled training sets.

**Common Output for All Datasets:**
- `train_full.pt` (full 70% training split)
- `train_100.pt`, `train_50.pt`, `train_25.pt`, `train_10.pt`, `train_5.pt`, `train_1.pt` (subsampled versions for data efficiency experiments; train_100.pt is identical to train_full.pt)
- `val.pt` (15% validation set)
- `test.pt` (15% test set)
- `label_stats.json` (mean and standard deviation for normalization)

**Note**: The split directories contain generated data and are excluded by `.gitignore`. You'll need to generate them locally by running the appropriate split script.

### Step 2: Fine-tuning

Each script requires:
- `--fraction`: Dataset fraction to use (1, 5, 10, 25, 50, or 100)
- `--ckpt`: Path to the GFM checkpoint (e.g., `../checkpoints/gfm_0.229.pk`)
- `--split_dir`: Directory containing the dataset splits
- `--config`: HydraGNN configuration JSON file, make one based on dataset and checkpoint used
- `--label_stats`: JSON file with normalization statistics (`{"mean": ..., "std": ...}`)
- `--seed`: Random seed (default: 0)

#### Example: Full Fine-tuning on MD17

```bash
cd md17
python md17_gfm_finetune.py \
    --fraction 100 \
    --ckpt ../checkpoints/gfm_0.229.pk \
    --split_dir md17_splits \
    --config config.json \
    --label_stats md17_splits/label_stats.json \
    --seed 42
```

#### Example: Frozen Backbone with Head Reuse

```bash
python md17_gfm_reuse_head.py \
    --fraction 25 \
    --ckpt ../checkpoints/gfm_0.229.pk \
    --split_dir md17_splits \
    --config config.json \
    --label_stats md17_splits/label_stats.json
```

#### Example: Frozen Backbone with New Head

```bash
python md17_gfm_head_scratch.py \
    --fraction 10 \
    --ckpt ../checkpoints/gfm_0.229.pk \
    --split_dir md17_splits \
    --config config.json \
    --label_stats md17_splits/label_stats.json
```

#### Example: Train from Scratch (No Pretrained Model)

```bash
python train_md17_scratch.py \
    --fraction 100 \
    --split_dir md17_splits \
    --config config.json \
    --label_stats md17_splits/label_stats.json
```

---

## Fine-tuning Modes Explained

1. **Full Fine-tuning** (`*_gfm_finetune.py`):
   - Loads the entire pretrained GFM (backbone + head)
   - All parameters are trainable
   - Best for larger datasets where overfitting is less of a concern

2. **Reuse Head** (`*_gfm_reuse_head.py`):
   - Loads pretrained GFM
   - Freezes the backbone (feature extraction layers)
   - Only trains the prediction head
   - Good for transfer learning with limited data

3. **Head from Scratch** (`*_gfm_head_scratch.py`):
   - Loads pretrained backbone
   - Replaces the head with a new randomly initialized one
   - Freezes backbone, trains only the new head
   - Useful when the original head doesn't match your task

4. **Train from Scratch** (`train_*_scratch.py`):
   - No pretrained model used
   - Trains everything from random initialization
   - Baseline comparison for transfer learning effectiveness

---

## Configuration Files

Each script requires a HydraGNN configuration JSON file that specifies:
- Model architecture (number of layers, hidden dimensions, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Task type (regression/classification)
- Output heads configuration

Refer to the HydraGNN documentation for configuration file formats.

---

## Data Efficiency Experiments

The `--fraction` parameter allows you to train with different amounts of data:
- `1`: Use 1% of training data
- `5`: Use 5% of training data
- `10`: Use 10% of training data
- `25`: Use 25% of training data
- `50`: Use 50% of training data
- `100`: Use 100% of training data

This is useful for studying how transfer learning helps with limited data.

---

## References

- **HydraGNN**: https://github.com/ORNL/HydraGNN
- **Pretrained GFM models**: https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024
- **QM9 dataset**: Quantum chemistry dataset with ~134k molecules
- **MD17 dataset**: Molecular dynamics trajectories
- **Transition1x dataset**: Transition state prediction dataset

