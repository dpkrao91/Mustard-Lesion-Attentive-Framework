#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split.

Usage::

    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint runs/palm/best.pt --output-dir runs/palm/eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.dataset import CLASS_NAMES, build_dataloaders  # noqa: E402
from src.engine import evaluate, measure_latency  # noqa: E402
from src.metrics import compute_metrics, pretty_print  # noqa: E402
from src.models import build_model_for_eval, count_parameters  # noqa: E402
from src.transforms import build_eval_transform, build_train_transform  # noqa: E402
from src.utils import (get_device, load_checkpoint, plot_confusion_matrix,  # noqa: E402
                       read_checkpoint_config, save_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(args.output_dir or Path(args.checkpoint).parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = int(cfg.data.get("image_size", 224))
    _, _, test_loader, _ = build_dataloaders(
        cfg, build_train_transform(img_size), build_eval_transform(img_size))

    ckpt_cfg = read_checkpoint_config(args.checkpoint, device)
    model = build_model_for_eval(cfg, ckpt_cfg, num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)

    criterion = nn.CrossEntropyLoss()
    _, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, list(CLASS_NAMES))
    metrics.update(measure_latency(model, device, img_size))
    metrics["trainable_params"] = count_parameters(model)

    save_json(metrics, out_dir / "test_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], list(CLASS_NAMES),
                          out_dir / "confusion_matrix.png")
    print(pretty_print(metrics, list(CLASS_NAMES)))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
