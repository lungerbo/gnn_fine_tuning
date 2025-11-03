#!/usr/bin/env python
# ------------------------------------------------------------
# Create Transition1x train/val/test splits + subsets
# ------------------------------------------------------------
import os
import json
import torch
import random
from torch_geometric.nn import radius_graph

# ---------- Settings ----------
DATASET_DIR = "transition1x_dataset"
OUT_DIR = "transition1x_splits"
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SUBSETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

# ---------- Preprocessing ----------
def transition1x_preprocess(data):
    """
    Preprocess Transition1x data to ensure edges and edge attributes exist.
    - Node features: atomic number + 3D coordinates (if not already set)
    - Edges: radius graph with r=5.0 Angstroms (if not already set)
    - Edge attributes: edge lengths (if not already set)
    """
    # Ensure node features exist (atomic number + coordinates)
    if not hasattr(data, 'x') or data.x is None:
        if hasattr(data, 'z') and hasattr(data, 'pos'):
            atomic_number = data.z.view(-1, 1).float()
            pos = data.pos.float()
            data.x = torch.cat([atomic_number, pos], dim=1)
        else:
            raise ValueError("Data must have 'z' (atomic numbers) and 'pos' (coordinates) attributes")
    
    # Ensure positions are available for edge generation
    if not hasattr(data, 'pos') or data.pos is None:
        raise ValueError("Data must have 'pos' (coordinates) attribute for edge generation")
    
    pos = data.pos.float()
    
    # Generate edges using radius graph if not present
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = radius_graph(pos, r=5.0, loop=False)
    
    # Compute edge attributes (edge lengths) if not present
    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
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
    labels = []
    for d in dataset:
        if hasattr(d, 'y') and d.y is not None:
            labels.append(d.y)
    if not labels:
        raise ValueError("No labels found in dataset")
    labels = torch.stack(labels)
    mean = labels.mean().item()
    std = labels.std().item()
    return mean, std

# ---------- Main ----------
if __name__ == "__main__":
    # Load full dataset
    all_path = os.path.join(DATASET_DIR, "all_data.pt")
    if not os.path.exists(all_path):
        raise FileNotFoundError(f"Missing dataset: {all_path}")
    
    print(f"Loading dataset from {all_path}...")
    full_data = torch.load(all_path)
    print(f"Loaded {len(full_data)} total samples")
    
    # Preprocess all data to ensure edges exist
    print("Preprocessing data to ensure edges and edge attributes...")
    for i, data in enumerate(full_data):
        full_data[i] = transition1x_preprocess(data)
        if i == 0:
            # Verify first sample
            print(f"Sample 0 verification:")
            print(f"  - Has x: {hasattr(full_data[0], 'x') and full_data[0].x is not None}")
            print(f"  - Has pos: {hasattr(full_data[0], 'pos') and full_data[0].pos is not None}")
            print(f"  - Has edge_index: {hasattr(full_data[0], 'edge_index') and full_data[0].edge_index is not None}")
            print(f"  - Has edge_attr: {hasattr(full_data[0], 'edge_attr') and full_data[0].edge_attr is not None}")
            if hasattr(full_data[0], 'edge_index') and full_data[0].edge_index is not None:
                print(f"  - Number of edges: {full_data[0].edge_index.shape[1]}")
    
    # Split dataset
    os.makedirs(OUT_DIR, exist_ok=True)
    train_set, val_set, test_set = split_dataset(full_data, TRAIN_FRAC, VAL_FRAC, seed=SEED)
    print(f"\nDataset split:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val: {len(val_set)}")
    print(f"  Test: {len(test_set)}")
    
    # Compute label statistics on training set only
    print("\nComputing label statistics on training set...")
    mean, std = compute_label_stats(train_set)
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    
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
        print(f"  Saved train_{tag}.pt with {len(subset)} samples")
    
    print(f"\n✅ All splits saved to '{OUT_DIR}'")
