#!/usr/bin/env python3
"""
QM9 reuse-head fine-tuning with frozen GFM backbone.

- Loads GFM checkpoint (accepts either {"model_state_dict": ...} or a raw state_dict)
- Freezes backbone parameters (keeps params with "head"/"output"/"classifier" trainable)
- Applies global label normalization (JSON with "mean" and "std")
- Runs hydragnn.train.train_validate_test (no DDP, single device)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    d = json.load(open(path, "r"))
    return float(d["mean"]), float(d["std"])


def normalize(ds, mean, std):
    for d in ds:
        d.y = (d.y - mean) / std
        if d.y.dim() == 1:
            d.y = d.y.unsqueeze(1)
    return ds


def freeze_backbone_keep_head(model):
    """Freeze all parameters except heads_NN"""
    for name, p in model.named_parameters():
        p.requires_grad = ("heads_NN" in name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Params total={total:,}, trainable={trainable:,} ({100.0*trainable/max(1,total):.1f}%)")


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.load(open(args.config, "r"))
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

    # load checkpoint (support both wrapped ckpt dict and raw state_dict)
    ck = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        state = {k.replace("module.", ""): v for k, v in ck.items()}
    else:
        raise RuntimeError("Unrecognized checkpoint format")

    model.load_state_dict(state, strict=False)

    freeze_backbone_keep_head(model)
    model = model.to(device)

    # quick forward sanity check
    batch = next(iter(train_loader))
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch)
    if isinstance(out, (list, tuple)):
        out = torch.cat([o.reshape(-1, 1) for o in out], dim=1)
    assert out.shape == batch.y.shape, f"forward output {out.shape} != target {batch.y.shape}"

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    tag = f"qm9_gfm_reusehead_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    writer = SummaryWriter(log_dir=f"./logs/{tag}")

    import hydragnn.train

    t0 = time.time()
    hydragnn.train.train_validate_test(
        model, optimizer,
        train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=False,
    )
    print("Done in %.1f sec" % (time.time() - t0))
