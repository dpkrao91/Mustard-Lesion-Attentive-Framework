#!/usr/bin/env python3
"""Generate Grad-CAM visualisations and compute Leaf-Attention-Fidelity (LAF).

Two outputs:

1. A grid of Grad-CAM overlays for a sample of test images (qualitative).
2. ``laf_report.json`` with mean LAF over the test set, per class
   (quantitative explainability — the proposed metric).

Usage::

    python scripts/explain.py --config configs/default.yaml \
        --checkpoint runs/palm/best.pt --num-vis 12 --max-laf 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.dataset import CLASS_NAMES, MustardLeafDataset  # noqa: E402
from src.gradcam import (GradCAM, compute_leaf_mask_from_tensor,  # noqa: E402
                         leaf_attention_fidelity, overlay_cam)
from src.models import build_model_for_eval  # noqa: E402
from src.transforms import build_eval_transform  # noqa: E402
from src.utils import get_device, load_checkpoint, read_checkpoint_config, save_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-vis", type=int, default=12)
    parser.add_argument("--max-laf", type=int, default=400,
                        help="Max test images used for the LAF average (speed).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(args.output_dir or Path(args.checkpoint).parent / "explain")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = int(cfg.data.get("image_size", 224))
    eval_tf = build_eval_transform(img_size)
    test_ds = MustardLeafDataset(Path(cfg.data.splits_dir) / "test.csv",
                                 cfg.data.root, eval_tf)

    ckpt_cfg = read_checkpoint_config(args.checkpoint, device)
    model = build_model_for_eval(cfg, ckpt_cfg, num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    cam_engine = GradCAM(model)

    # ---- Quantitative: Leaf-Attention-Fidelity over the test set ----
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    n = min(args.max_laf, len(test_ds))
    idxs = rng.choice(len(test_ds), size=n, replace=False)
    laf_per_class = {c: [] for c in CLASS_NAMES}
    for i in idxs:
        img, label = test_ds[int(i)]
        x = img.unsqueeze(0).to(device)
        cam, _ = cam_engine(x, target_class=None)
        mask = compute_leaf_mask_from_tensor(img)
        laf = leaf_attention_fidelity(cam, mask)
        if not np.isnan(laf):
            laf_per_class[CLASS_NAMES[label]].append(laf)

    laf_report = {
        "n_images": int(n),
        "mean_laf_overall": float(np.mean(
            [v for vals in laf_per_class.values() for v in vals])),
        "mean_laf_per_class": {
            c: (float(np.mean(v)) if v else float("nan"))
            for c, v in laf_per_class.items()},
    }
    save_json(laf_report, out_dir / "laf_report.json")
    print("Leaf-Attention-Fidelity (higher = attention on plant tissue):")
    print(f"  overall: {laf_report['mean_laf_overall']:.4f}")
    for c, v in laf_report["mean_laf_per_class"].items():
        print(f"  {c:<18}: {v:.4f}")

    # ---- Qualitative: Grad-CAM grid ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_idxs = rng.choice(len(test_ds), size=min(args.num_vis, len(test_ds)),
                          replace=False)
    cols = 4
    rows = int(np.ceil(len(vis_idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, i in enumerate(vis_idxs):
        img, label = test_ds[int(i)]
        x = img.unsqueeze(0).to(device)
        cam, pred = cam_engine(x, target_class=None)
        overlay = overlay_cam(img, cam)
        ax = axes.ravel()[k]
        ax.imshow(overlay)
        ok = "OK" if pred == label else "X"
        ax.set_title(f"t:{CLASS_NAMES[label][:9]}\np:{CLASS_NAMES[pred][:9]} [{ok}]",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_grid.png", dpi=180)
    plt.close(fig)
    cam_engine.close()
    print(f"\nSaved Grad-CAM grid and LAF report to {out_dir}")


if __name__ == "__main__":
    main()
