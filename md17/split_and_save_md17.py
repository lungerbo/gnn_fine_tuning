
#!/usr/bin/env python
# ------------------------------------------------------------
# Create MD-17 (revised uracil) train/val/test splits + subsets
# ------------------------------------------------------------
import os
import torch
import random
import torch_geometric
from torch_geometric.nn import radius_graph

# ---------- Settings ----------
ROOT_DIR = "dataset/md17"
OUT_DIR = "md17_splits"
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SUBSETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

# ---------- Preprocessing ----------
def md17_pre_transform(data):
    atomic_number = data.z.view(-1, 1).float()
    pos = data.pos.float()
    data.x = torch.cat([atomic_number, pos], dim=1)
    data.y = data.energy.view(1)
    data.edge_index = radius_graph(pos, r=5.0, loop=False)
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

# ---------- Main ----------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = torch_geometric.datasets.MD17(
        root=ROOT_DIR,
        name="revised uracil",
        pre_transform=md17_pre_transform,
        pre_filter=None,
    )

    train_set, val_set, test_set = split_dataset(dataset, TRAIN_FRAC, VAL_FRAC, seed=SEED)
    print(f"Dataset size: {len(dataset)}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    save(train_set, os.path.join(OUT_DIR, "train_full.pt"))
    save(val_set, os.path.join(OUT_DIR, "val.pt"))
    save(test_set, os.path.join(OUT_DIR, "test.pt"))

    for frac in SUBSETS:
        subset = subsample(train_set, frac, seed=SEED)
        tag = str(int(frac * 100))
        save(subset, os.path.join(OUT_DIR, f"train_{tag}.pt"))
        print(f"Saved train_{tag}.pt with {len(subset)} samples")

    print(f"\nAll splits saved in {OUT_DIR}")
