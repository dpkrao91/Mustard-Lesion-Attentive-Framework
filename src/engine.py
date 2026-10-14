"""Training / evaluation loops shared by the CLI scripts."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, use_amp: bool = False) -> float:
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        bs = images.size(0)
        running += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{running / max(n, 1):.4f}")
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_loss, y_true, y_pred, y_prob)."""
    model.eval()
    running, n = 0.0, 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        prob = torch.softmax(logits, dim=1)
        running += loss.item() * images.size(0)
        n += images.size(0)
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(prob.argmax(1).cpu().tolist())
        all_prob.append(prob.cpu().numpy())
    return (running / max(n, 1),
            np.asarray(all_true),
            np.asarray(all_pred),
            np.concatenate(all_prob, axis=0))


def build_optimizer(model, cfg):
    name = cfg.train.get("optimizer", "adamw").lower()
    lr = float(cfg.train.lr)
    wd = float(cfg.train.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    name = cfg.train.get("scheduler", "cosine").lower()
    epochs = int(cfg.train.epochs)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * max(steps_per_epoch, 1))
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'")


@torch.no_grad()
def measure_latency(model, device, image_size: int = 224,
                    warmup: int = 5, iters: int = 30) -> Dict[str, float]:
    """Single-image inference latency (ms). For edge-deployability reporting."""
    import time

    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    return {"latency_ms": elapsed * 1000.0, "throughput_fps": 1.0 / elapsed}
