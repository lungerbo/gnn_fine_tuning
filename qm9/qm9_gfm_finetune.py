#!/usr/bin/env python3
"""
Minimal full GFM fine-tuning on QM9 (train backbone + GFM-style head).

- Replaces the model head with a GFM-style MLP
- Loads checkpoint (wrapped or raw state_dict) and loads weights
- All parameters left trainable (no freezing)
- Global label normalization (JSON with "mean" and "std")
- Uses HydraGNN train_validate_test (DDP-compatible as in original)
- Minimal: no extra sanity checks, no best-epoch copy block
"""
import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model, get_comm_size_and_rank
from hydragnn.utils import update_config
from hydragnn.models import create_model_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--label_stats", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    return float(d["mean"]), float(d["std"])


def normalize(ds, mean, std):
    for d in ds:
        d.y = (d.y - mean) / std
    return ds


def replace_head_gfm_style(model):
    # infer input dim from existing head if possible
    in_dim = None
    try:
        first_head = model.heads_NN[0]
        for m in first_head:
            if isinstance(m, nn.Linear):
                in_dim = m.in_features
                break
    except Exception:
        in_dim = getattr(model, "node_out_features", None) or getattr(model, "out_dim", None)
    if in_dim is None:
        raise RuntimeError("Cannot infer head input dim to replace head")

    model.heads_NN = nn.ModuleList([
        nn.Sequential(
            nn.Linear(in_dim, 888),
            nn.ReLU(),
            nn.Linear(888, 888),
            nn.ReLU(),
            nn.Linear(888, 888),
            nn.ReLU(),
            nn.Linear(888, 1)
        )
    ])


if __name__ == "__main__":
    args = parse_args()
    tag = f"qm9_gfm_full_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    world_size, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/{world_size}")

    cfg = json.load(open(args.config))
    mean, std = load_stats(args.label_stats)
    log(f"[STATS] mean={mean:.6f}, std={std:.6f}")

    base = args.split_dir
    train_ds = normalize(torch.load(f"{base}/train_{args.fraction}.pt"), mean, std)
    val_ds = normalize(torch.load(f"{base}/val.pt"), mean, std)
    test_ds = normalize(torch.load(f"{base}/test.pt"), mean, std)

    bs = cfg["NeuralNetwork"]["Training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    update_config(cfg, train_loader, val_loader, test_loader)
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"])

    # Replace head with GFM-style head (we will fine-tune backbone + head)
    replace_head_gfm_style(model)

    # Load checkpoint (wrapped or raw)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        raw = ckpt
    else:
        raise RuntimeError("Unrecognized checkpoint format")

    # strip "module." prefix if present and load (strict=False to allow head mismatch)
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)

    # ensure all params are trainable (full fine-tune)
    for p in model.parameters():
        p.requires_grad = True

    model = get_distributed_model(model.to("cuda"), cfg["Verbosity"]["level"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting full fine-tune → tag={tag}")
    t0 = time.time()

    hydragnn.train.train_validate_test(
        model, optimizer,
        train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True)
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")
