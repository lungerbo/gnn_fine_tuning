#!/usr/bin/env python3
"""t1x: scratch head — freeze backbone, reinit head₀ in-place, train head only."""

import os, json, time, random, argparse, shutil, re
import numpy as np
import torch
import torch.nn as nn
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
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
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


def kaiming_reinit_head(module: nn.Module):
    """Reinitialize Linear layers in a head with Kaiming normal; zero biases (in-place)."""
    reinit_cnt = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            reinit_cnt += 1
    return reinit_cnt


if __name__ == "__main__":
    args = parse_args()
    tag = f"t1x_gfm_scratch_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
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

    # Load pretrained weights (we will reuse the backbone)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
    # safe-ish load (older/newer PyTorch may return tuple or object)
    res = model.load_state_dict(state, strict=False)
    missing = getattr(res, "missing_keys", getattr(res, "missing", []))
    unexpected = getattr(res, "unexpected_keys", getattr(res, "unexpected", []))
    if len(missing) == 0:
        log(" [LOAD] No missing keys.")
    else:
        log(f" [LOAD] Missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if len(unexpected) > 0:
        log(f" [LOAD] Unexpected ckpt keys (ignored): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    # Replace / reinitialize head₀ IN-PLACE (do NOT append a new head)
    assert hasattr(model, "heads_NN") and len(model.heads_NN) >= 1, "Model lacks heads_NN[0]"
    old_head_sd = {k: v.detach().clone() for k, v in model.heads_NN[0].state_dict().items()}
    reinit_layers = kaiming_reinit_head(model.heads_NN[0])  # in-place reinit
    new_head_sd = model.heads_NN[0].state_dict()
    max_delta = max((new_head_sd[k] - old_head_sd[k]).abs().max().item() for k in old_head_sd)
    log(f"[HEAD] head₀ reinitialized in-place: {reinit_layers} Linear layers reset; max |Δ|={max_delta:.3e}")

    # Freeze backbone; train head only
    for name, p in model.named_parameters():
        p.requires_grad = ("heads_NN" in name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[MODEL] Params total={total:,}, trainable={trainable:,} ({100.0*trainable/max(1,total):.1f}%)")

    # DDP wrap and optimizer
    model = get_distributed_model(model, cfg["Verbosity"]["level"])
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found after freezing — check freeze pattern.")
    optimizer = torch.optim.AdamW(params, lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting scratch-head (frozen-backbone) → tag={tag}")
    t0 = time.time()

    hydragnn.train.train_validate_test(
        model, optimizer,
        train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True)
    )
    log(f"[DONE] Training completed in {time.time() - t0:.1f} sec")
