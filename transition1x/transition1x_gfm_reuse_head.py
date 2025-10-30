#!/usr/bin/env python3
# Compact reuse-head fine-tuning: load checkpoint, freeze backbone, train heads (no best-copy, no final test eval).

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
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.models import create_model_config
from hydragnn.utils import update_config


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
        if hasattr(d, "y") and d.y is not None:
            d.y = (d.y - mean) / std
    return ds


if __name__ == "__main__":
    args = parse_args()
    tag = f"t1x_gfm_reuse_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
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

    # Load checkpoint (backbone + heads); allow extra keys
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) == 0:
        log("[LOAD] No missing keys.")
    else:
        log(f"[LOAD] Missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if len(unexpected) > 0:
        log(f"[LOAD] Unexpected ckpt keys (ignored): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    # Freeze backbone parameters; train heads only
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if ("backbone" in name) or ("egnn" in name.lower()) or ("graph_convs" in name.lower()):
            p.requires_grad = False
            frozen += p.numel()
        else:
            p.requires_grad = True
            trainable += p.numel()
    log(f"[SANITY] params frozen={frozen}, trainable={trainable}")

    model = get_distributed_model(model, cfg["Verbosity"]["level"])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting reuse-head (frozen-backbone) → tag={tag}")
    t0 = time.time()

    hydragnn.train.train_validate_test(
        model, optimizer,
        train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True)
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")
