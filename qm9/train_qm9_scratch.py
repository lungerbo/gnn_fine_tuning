#!/usr/bin/env python3
"""
Script to train QM9 from scratch (no DDP, no extras).
- Loads pre-saved splits: train_{fraction}.pt, val.pt, test.pt
- Applies global label normalization from a JSON with keys "mean" and "std"
- Runs hydragnn.train.train_validate_test
"""
import argparse
import json
import random
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.load(open(args.config))
    mean, std = load_stats(args.label_stats)

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
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    tag = f"qm9_scratch_{args.fraction}_seed{args.seed}"
    writer = SummaryWriter(log_dir=f"./logs/{tag}")

    import hydragnn.train

    t0 = time.time()
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        cfg["NeuralNetwork"],
        tag,
        cfg["Verbosity"]["level"],
        create_plots=False,
    )
    print("Training done in %.1f sec" % (time.time() - t0))
