#!/usr/bin/env python3
"""
QM9 reuse-head fine-tuning with frozen GFM backbone.

- Loads GFM checkpoint (accepts either {"model_state_dict": ...} or a raw state_dict)
- Freezes backbone parameters (keeps heads_NN trainable)
- Applies global label normalization (JSON with "mean" and "std")
- Runs hydragnn.train.train_validate_test with DDP support
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

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
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
    tag = f"qm9_gfm_reusehead_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    world_size, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/{world_size}")

    cfg = json.load(open(args.config, "r"))
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

    # load checkpoint (support both wrapped ckpt dict and raw state_dict)
    ck = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        state = {k.replace("module.", ""): v for k, v in ck.items()}
    else:
        raise RuntimeError("Unrecognized checkpoint format")

    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    freeze_backbone_keep_head(model)
    model = get_distributed_model(model, cfg["Verbosity"]["level"])

    # quick forward sanity check (informational)
    try:
        with torch.no_grad():
            batch = next(iter(train_loader)).to("cuda")
            out = model(batch)
            if isinstance(out, (list, tuple)):
                out = torch.cat([o.reshape(-1, 1) for o in out], dim=1)
            log(f"[SANITY] forward out shape={tuple(out.shape)} y shape={tuple(batch.y.shape)}")
    except Exception as e:
        log(f"[WARN] quick forward failed: {e}")

    lr = float(cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting reuse-head finetune → tag={tag}")
    t0 = time.time()
    hydragnn.train.train_validate_test(
        model, optimizer, train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True),
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")
