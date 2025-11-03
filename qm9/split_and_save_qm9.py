#!/usr/bin/env python
# ------------------------------------------------------------
# Create QM9 train/val/test splits + subsets
# ------------------------------------------------------------
import os
import json
import torch
import random
from torch_geometric.datasets import QM9
from torch_geometric.nn import radius_graph

# ---------- Settings ----------
ROOT_DIR = "dataset/qm9"
OUT_DIR = "qm9_splits"
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SUBSETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

# ---------- Preprocessing ----------
def qm9_pre_transform(data):
    """
    Preprocess QM9 data:
    - Node features: atomic number + 3D coordinates
    - Target: free energy at 298.15K (G, index 10) normalized by number of atoms
    - Edges: radius graph with r=5.0 Angstroms
    - Edge attributes: edge lengths
    """
    atomic_number = data.z.view(-1, 1).float()
    pos = data.pos.float()
    data.x = torch.cat([atomic_number, pos], dim=1)
    
    # Target: free energy at 298.15K (G, index 10), normalized by number of atoms
    data.y = (data.y[:, 10] / len(data.x)).view(1)
    
    # Create edges using radius graph
    data.edge_index = radius_graph(pos, r=5.0, loop=False)
    
    # Edge attributes: edge lengths
    row, col = data.edge_index
    edge_vec = pos[row] - pos[col]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
    data.edge_attr = edge_length
    
    return data

# ---------- Dataset Split ----------
def split_dataset(dataset, perc_train, perc_val, seed=42):
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(perc_train * n)
    n_val = int(perc_val * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]
    return train_set, val_set, test_set

# ---------- Subsampling ----------
def subsample(dataset, fraction, seed=42):
    k = max(1, int(len(dataset) * fraction))
    random.seed(seed)
    return random.sample(dataset, k)

# ---------- Save ----------
def save(dataset, path):
    torch.save(dataset, path)

# ---------- Compute Label Stats ----------
def compute_label_stats(dataset):
    """Compute mean and std of labels for normalization."""
    labels = torch.stack([d.y for d in dataset])
    mean = labels.mean().item()
    std = labels.std().item()
    return mean, std

# ---------- Main ----------
if __name__ == "__main__":
    # Remove old processed data to apply updated pre_transform
    if os.path.exists(os.path.join(ROOT_DIR, "processed")):
        print("Removing old processed data to apply updated pre_transform...")
        import shutil
        shutil.rmtree(os.path.join(ROOT_DIR, "processed"))
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load QM9 dataset with preprocessing
    print("Loading QM9 dataset...")
    dataset = QM9(root=ROOT_DIR, pre_transform=qm9_pre_transform, pre_filter=None)
    
    # Split dataset
    train_set, val_set, test_set = split_dataset(dataset, TRAIN_FRAC, VAL_FRAC, seed=SEED)
    print(f"Dataset size: {len(dataset)}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Compute label statistics on training set only
    print("\nComputing label statistics on training set...")
    mean, std = compute_label_stats(train_set)
    print(f"Mean: {mean:.6f}, Std: {std:.6f}")
    
    # Save label statistics
    stats = {"mean": mean, "std": std}
    with open(os.path.join(OUT_DIR, "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved label statistics to {OUT_DIR}/label_stats.json")
    
    # Save full splits
    save(train_set, os.path.join(OUT_DIR, "train_full.pt"))
    save(val_set, os.path.join(OUT_DIR, "val.pt"))
    save(test_set, os.path.join(OUT_DIR, "test.pt"))
    
    # Create and save subsampled training sets
    print("\nCreating subsampled training sets...")
    for frac in SUBSETS:
        subset = subsample(train_set, frac, seed=SEED)
        tag = str(int(frac * 100))
        save(subset, os.path.join(OUT_DIR, f"train_{tag}.pt"))
        print(f"Saved train_{tag}.pt with {len(subset)} samples")
    
    print(f"\nAll splits saved in {OUT_DIR}")
