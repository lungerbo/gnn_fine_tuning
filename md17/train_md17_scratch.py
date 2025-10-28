#!/usr/bin/env python3
"""
MD17 scratch training script.
"""

import os
import json
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model, get_comm_size_and_rank
from hydragnn.utils import update_config
from hydragnn.models import create_model_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True, help="1|3|5|10|25|50|100")
    p.add_argument("--split_dir", required=True, help="dir with train_*.pt, val.pt, test.pt, label_stats.json")
    p.add_argument("--config", required=True, help="HydraGNN JSON config")
    p.add_argument("--label_stats", required=False, help="label_stats.json (if absent, compute on the fly)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_stats_or_compute(path_or_none, train_ds):
    if path_or_none and os.path.exists(path_or_none):
        d = json.load(open(path_or_none))
        return float(d["mean"]), float(d["std"])
    ys = [float(d.y.view(-1)[0].item()) for d in train_ds]
    mean = float(np.mean(ys))
    std = float(np.std(ys))
    return mean, std


def normalize_dataset(ds, mean, std):
    if std == 0:
        raise ValueError("Std is zero; cannot normalize labels.")
    for d in ds:
        d.y = (d.y - mean) / std
    return ds


if __name__ == "__main__":
    args = parse_args()
    tag = f"md17_scratch_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    world_size, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/{world_size}")

    base = args.split_dir
    train_file = f"train_{args.fraction}.pt" if args.fraction != "100" else "train_full.pt"
    train_ds = torch.load(os.path.join(base, train_file))
    val_ds = torch.load(os.path.join(base, "val.pt"))
    test_ds = torch.load(os.path.join(base, "test.pt"))
    log(f"Loaded datasets: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    mean, std = load_stats_or_compute(args.label_stats, train_ds)
    log(f"[STATS] mean={mean:.6f}, std={std:.6f}")
    normalize_dataset(train_ds, mean, std)
    normalize_dataset(val_ds, mean, std)
    normalize_dataset(test_ds, mean, std)

    bs = int(json.load(open(args.config))["NeuralNetwork"]["Training"]["batch_size"])

    class YFix:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for b in self.loader:
                if b.y.dim() == 1: b.y = b.y.unsqueeze(1)
                yield b
        def __len__(self): return len(self.loader)
        def __getattr__(self, n): return getattr(self.loader, n)

    train_loader = YFix(DataLoader(train_ds, batch_size=bs, shuffle=True))
    val_loader = YFix(DataLoader(val_ds, batch_size=bs))
    test_loader = YFix(DataLoader(test_ds, batch_size=bs))

    cfg = json.load(open(args.config))
    update_config(cfg, train_loader, val_loader, test_loader)
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"]).to("cuda")
    model = get_distributed_model(model, cfg["Verbosity"]["level"])

    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting (scratch) → tag={tag}")
    t0 = time.time()
    hydragnn.train.train_validate_test(
        model, optimizer, train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True),
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")
