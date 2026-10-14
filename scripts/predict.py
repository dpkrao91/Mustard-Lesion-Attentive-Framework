#!/usr/bin/env python3
"""Predict the disease class for a single image (deployment-style inference).

Usage::

    python scripts/predict.py --config configs/default.yaml \
        --checkpoint runs/palm/best.pt --image path/to/leaf.jpg --cam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.dataset import CLASS_NAMES  # noqa: E402
from src.models import build_model_for_eval  # noqa: E402
from src.transforms import build_eval_transform  # noqa: E402
from src.utils import get_device, load_checkpoint, read_checkpoint_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cam", action="store_true", help="Also save a Grad-CAM overlay.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    img_size = int(cfg.data.get("image_size", 224))
    tf = build_eval_transform(img_size)

    ckpt_cfg = read_checkpoint_config(args.checkpoint, device)
    model = build_model_for_eval(cfg, ckpt_cfg, num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    with Image.open(args.image) as im:
        x = tf(im.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    order = np.argsort(prob)[::-1]
    print(f"Prediction for {args.image}:")
    for i in order:
        print(f"  {CLASS_NAMES[i]:<18}: {prob[i] * 100:5.2f}%")

    if args.cam:
        from src.gradcam import GradCAM, overlay_cam
        import cv2
        cam_engine = GradCAM(model)
        cam, pred = cam_engine(x, target_class=int(order[0]))
        overlay = overlay_cam(x[0], cam)
        out = Path(args.image).with_suffix("").as_posix() + "_gradcam.png"
        cv2.imwrite(out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cam_engine.close()
        print(f"  Grad-CAM overlay -> {out}")


if __name__ == "__main__":
    main()
