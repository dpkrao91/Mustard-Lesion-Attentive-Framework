#!/usr/bin/env python3
"""Train a model on the mustard leaf disease dataset.

Examples
--------
Proposed PALM model (EfficientNet-B0 + CBAM + class-balanced focal loss)::

    python scripts/train.py --config configs/default.yaml

Baseline (no attention, plain cross-entropy)::

    python scripts/train.py --model efficientnet_b0 --loss ce --no-cbam \
        --output-dir runs/baseline_effb0

Any timm backbone works (resnet50, mobilenetv3_large_100, convnext_tiny, ...).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import add_common_args, apply_cli_overrides, load_config, to_plain  # noqa: E402
from src.dataset import CLASS_NAMES, build_dataloaders  # noqa: E402
from src.engine import (  # noqa: E402
    build_optimizer, build_scheduler, evaluate, measure_latency, train_one_epoch)
from src.losses import build_loss  # noqa: E402
from src.metrics import compute_metrics, pretty_print  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.transforms import build_eval_transform, build_train_transform  # noqa: E402
from src.utils import (  # noqa: E402
    get_device, plot_history, save_checkpoint, save_json, set_seed)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    output_dir = Path(cfg.get("output_dir", "runs/palm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Output: {output_dir}")
    print(f"Backbone: {cfg.model.backbone} | CBAM: {cfg.model.get('use_cbam', True)} "
          f"| Loss: {cfg.loss.name}")

    img_size = int(cfg.data.get("image_size", 224))
    train_tf = build_train_transform(img_size)
    eval_tf = build_eval_transform(img_size)
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        cfg, train_tf, eval_tf)

    class_counts = train_ds.class_counts().tolist()
    print("Train class counts:", dict(zip(CLASS_NAMES, class_counts)))

    model = build_model(cfg, num_classes=len(CLASS_NAMES)).to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    criterion = build_loss(cfg, class_counts).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    epochs = int(cfg.train.epochs)
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_epoch = -1.0, -1

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                  device, scaler, use_amp)
        val_loss, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, y_prob, list(CLASS_NAMES))
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(metrics["macro_f1"])
        print(f"[{epoch:02d}/{epochs}] train_loss={tr_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macroF1={metrics['macro_f1']:.4f} "
              f"val_balAcc={metrics['balanced_accuracy']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            save_checkpoint(model, optimizer, epoch, best_f1,
                            output_dir / "best.pt",
                            extra={"config": to_plain(cfg), "class_names": list(CLASS_NAMES)})

    save_checkpoint(model, optimizer, epochs, best_f1, output_dir / "last.pt",
                    extra={"config": to_plain(cfg), "class_names": list(CLASS_NAMES)})
    plot_history(history, output_dir / "training_curves.png")
    save_json(history, output_dir / "history.json")

    # Final test evaluation with the best checkpoint.
    from src.utils import load_checkpoint
    load_checkpoint(model, output_dir / "best.pt", map_location=device)
    test_loss, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(y_true, y_pred, y_prob, list(CLASS_NAMES))
    test_metrics["best_val_macro_f1"] = best_f1
    test_metrics["best_epoch"] = best_epoch
    test_metrics.update(measure_latency(model, device, img_size))
    test_metrics["trainable_params"] = count_parameters(model)
    save_json(test_metrics, output_dir / "test_metrics.json")

    from src.utils import plot_confusion_matrix
    plot_confusion_matrix(test_metrics["confusion_matrix"], list(CLASS_NAMES),
                          output_dir / "confusion_matrix.png")

    print("\n===== TEST (best checkpoint) =====")
    print(pretty_print(test_metrics, list(CLASS_NAMES)))
    print(f"\nLatency: {test_metrics['latency_ms']:.2f} ms/img "
          f"({test_metrics['throughput_fps']:.1f} fps) on {device}")
    print(f"Saved everything to {output_dir}")


if __name__ == "__main__":
    main()
