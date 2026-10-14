"""Automatic leaf-foreground segmentation (no manual masks required).

Used to compute the Leaf-Attention-Fidelity (LAF) metric: the fraction of a
Grad-CAM saliency map that falls on plant tissue rather than on the soil / weed /
background clutter that characterises field-captured mustard imagery.

The segmentation is deliberately simple and label-free so the whole pipeline is
reproducible without extra annotation:

1. Convert to HSV and ExG (Excess-Green) colour space.
2. Threshold for vegetation. Crucially we keep BOTH healthy green tissue and the
   chlorotic / necrotic diseased tissue (Alternaria's brown target lesions and
   White Rust's chalky-white pustules) by unioning a green mask with a
   "non-soil, high-value" mask.
3. Morphological opening/closing + largest-component selection to clean up.

This is a proxy, not a ground-truth organ mask; the paper reports LAF as a
relative diagnostic across models, which is robust to the proxy's noise.
"""
from __future__ import annotations

import cv2
import numpy as np


def excess_green(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    total = r + g + b + 1e-6
    r_n, g_n, b_n = r / total, g / total, b / total
    return 2.0 * g_n - r_n - b_n


def segment_leaf(rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 {0,1} leaf-foreground mask of shape (H, W)."""
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Green / yellow-green healthy & chlorotic tissue.
    green = ((h >= 20) & (h <= 95) & (s >= 40) & (v >= 35)).astype(np.uint8)

    # Excess-green vegetation index (catches desaturated diseased leaf too).
    exg = excess_green(rgb)
    veg = (exg > 0.02).astype(np.uint8)

    # Bright chalky-white pustules (White Rust) -> high V, low-ish S.
    chalky = ((v >= 180) & (s <= 60)).astype(np.uint8)

    mask = ((green | veg | chalky) > 0).astype(np.uint8)

    # Remove obvious dark soil shadows.
    mask[v < 25] = 0

    # Morphological cleanup.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep the largest connected component (the focal leaf) plus sizeable others.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size:
            keep = np.where(areas >= max(0.05 * mask.size / 1.0, areas.max() * 0.15))[0] + 1
            cleaned = np.isin(labels, keep).astype(np.uint8)
            if cleaned.sum() > 0:
                mask = cleaned

    # Fallback: if segmentation collapses, treat whole frame as foreground so
    # LAF degrades gracefully to "uninformative" rather than crashing.
    if mask.sum() < 0.01 * mask.size:
        mask = np.ones_like(mask)
    return mask
