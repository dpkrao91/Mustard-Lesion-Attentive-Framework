"""Reproducibility, checkpointing and plotting helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism vs. speed trade-off; keep cudnn deterministic for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def save_checkpoint(model, optimizer, epoch: int, best_metric: float,
                    path: str | Path, extra: Dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path: str | Path, map_location="cpu"):
    # These checkpoints are produced by this codebase (a trusted source) and
    # store a plain config dict alongside the weights, so weights_only=False is
    # safe here. Fall back gracefully on older torch that lacks the kwarg.
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt


def read_checkpoint_config(path: str | Path, map_location="cpu"):
    """Return the config dict stored inside a checkpoint (or None)."""
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    return ckpt.get("config") if isinstance(ckpt, dict) else None


def plot_confusion_matrix(cm, class_names, out_path: str | Path,
                          normalize: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    title = "Confusion matrix"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
        title += " (row-normalised)"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_history(history: Dict, out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, history["val_macro_f1"], label="val macro-F1", color="green")
    ax2.set_title("Validation macro-F1"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
