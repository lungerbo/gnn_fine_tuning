#!/usr/bin/env python3
"""
Script to train QM9 from scratch with DDP support.
- Loads pre-saved splits: train_{fraction}.pt, val.pt, test.pt
- Applies global label normalization from a JSON with keys "mean" and "std"
- Runs hydragnn.train.train_validate_test
"""
import argparse
import json
import os
import random
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.utils import update_config
from hydragnn.models import create_model_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--label_stats", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_stats(path):
    d = json.load(open(path))
    return float(d["mean"]), float(d["std"])


def normalize(ds, mean, std):
    for d in ds:
        d.y = (d.y - mean) / std
    return ds


if __name__ == "__main__":
    args = parse_args()
    tag = f"qm9_scratch_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
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
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"]).to("cuda")
    model = get_distributed_model(model, cfg["Verbosity"]["level"])

    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting FROM-SCRATCH training → tag={tag}")
    t0 = time.time()
    hydragnn.train.train_validate_test(
        model, optimizer, train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True),
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")
