#!/usr/bin/env python3
"""
MD17 head-from-scratch, frozen GFM backbone.
- Loads GFM checkpoint (backbone only)
- Replaces head with a fresh GFM-style MLP (scratch init)
- Freezes backbone; trains head only
- Global label normalization (reads label_stats.json)
- Minimal sanity: y shape and forward check
"""

import os
import json
import time
import random
import argparse
import re

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
    p.add_argument("--fraction", required=True, help="1|5|10|25|50|100")
    p.add_argument("--ckpt", required=True, help="GFM checkpoint .pk")
    p.add_argument("--split_dir", required=True, help="dir with train_*.pt, val.pt, test.pt, label_stats.json")
    p.add_argument("--config", required=True, help="HydraGNN JSON (matches expected shared/head dims)")
    p.add_argument("--label_stats", required=True, help="label_stats.json with {mean,std}")
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


def strip_module(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


class YFix:
    def __init__(self, loader): self.loader = loader
    def __iter__(self):
        for b in self.loader:
            if getattr(b, "y", None) is not None and b.y.dim() == 1:
                b.y = b.y.unsqueeze(1)
            yield b
    def __len__(self): return len(self.loader)
    def __getattr__(self, n): return getattr(self.loader, n)


def replace_head_gfm_style(model):
    try:
        in_dim = model.heads_NN[0][0].in_features
    except Exception:
        in_dim = model.graph_shared[-1].out_features
    model.heads_NN = nn.ModuleList([
        nn.Sequential(
            nn.Linear(in_dim, 888), nn.ReLU(),
            nn.Linear(888, 888),    nn.ReLU(),
            nn.Linear(888, 888),    nn.ReLU(),
            nn.Linear(888, 1),
        )
    ])
    log(f"[HEAD] scratch GFM-style head: {in_dim} -> 888 -> 888 -> 888 -> 1")


def freeze_backbone_keep_head(model):
    for name, p in model.named_parameters():
        p.requires_grad = ("heads_NN" in name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[MODEL] Params total={total:,}, trainable={trainable:,} ({100.0*trainable/max(1,total):.1f}%)")


def sanity_loader(loader, name):
    b = next(iter(loader))
    B = int(b.batch.max().item()) + 1 if hasattr(b, "batch") else b.y.shape[0]
    assert b.y.shape == (B, 1), f"{name}.y shape={tuple(b.y.shape)} expected ({B},1)"
    log(f"[SANITY] {name} y.shape OK: {tuple(b.y.shape)}")


if __name__ == "__main__":
    args = parse_args()
    tag = f"md17_gfm_head_scratch_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
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
    train_ds = normalize(torch.load(os.path.join(base, f"train_{args.fraction}.pt")), mean, std)
    val_ds = normalize(torch.load(os.path.join(base, "val.pt")), mean, std)
    test_ds = normalize(torch.load(os.path.join(base, "test.pt")), mean, std)
    log(f"Loaded datasets: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    bs = int(cfg["NeuralNetwork"]["Training"]["batch_size"])
    train_loader = YFix(DataLoader(train_ds, batch_size=bs, shuffle=True))
    val_loader = YFix(DataLoader(val_ds, batch_size=bs))
    test_loader = YFix(DataLoader(test_ds, batch_size=bs))

    sanity_loader(train_loader, "train")
    sanity_loader(val_loader, "val")
    sanity_loader(test_loader, "test")

    update_config(cfg, train_loader, val_loader, test_loader)

    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"])

    # replace head with scratch GFM-style MLP
    replace_head_gfm_style(model)

    # load only backbone from checkpoint (exclude heads_NN)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = strip_module(ckpt.get("model_state_dict", ckpt))
    backbone_state = {k: v for k, v in state.items() if not k.startswith("heads_NN")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    log(f"[CKPT] backbone_keys_loaded={len(backbone_state)} missing={len(missing)} unexpected={len(unexpected)}")
    bad = [k for k in missing if not k.startswith("heads_NN")]
    if bad:
        raise RuntimeError(f"[ERROR] Backbone mismatch on: {bad[:8]} ...")

    # freeze backbone, keep new head trainable
    freeze_backbone_keep_head(model)

    model = get_distributed_model(model.to("cuda"), cfg["Verbosity"]["level"])

    # quick forward sanity (informational)
    try:
        with torch.no_grad():
            b = next(iter(train_loader)).to("cuda")
            out = model(b)
            if isinstance(out, (list, tuple)):
                out = torch.cat([o.reshape(-1, 1) for o in out], dim=1)
            assert out.shape == b.y.shape, f"forward out {tuple(out.shape)} != y {tuple(b.y.shape)}"
            log(f"[SANITY] forward OK: out.shape={tuple(out.shape)}")
    except Exception as e:
        log(f"[WARN] quick forward failed: {e}")

    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting head-from-scratch finetune → tag={tag}")
    t0 = time.time()
    hydragnn.train.train_validate_test(
        model, optimizer, train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True),
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")
