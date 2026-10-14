"""Grad-CAM and the Leaf-Attention-Fidelity (LAF) explainability metric.

Grad-CAM (Selvaraju et al., 2017) produces a class-discriminative saliency map.
We hook the post-CBAM feature map exposed by ``AttentiveClassifier``.

LAF (this work) quantifies whether that saliency lands on plant tissue. Given a
normalised CAM ``M in [0,1]^{HxW}`` and a leaf mask ``L in {0,1}^{HxW}``::

    LAF = sum(M * L) / sum(M)

i.e. the proportion of total activation mass on foliage. A model exploiting
background shortcuts scores low even at high accuracy; a pathology-grounded
model scores high. We report mean LAF over the test set per class.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .segmentation import segment_leaf
from .transforms import denormalize


class GradCAM:
    """Grad-CAM on the cached post-attention feature map of AttentiveClassifier."""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd = self.model.cbam.register_forward_hook(self._save_act) \
            if not isinstance(self.model.cbam, torch.nn.Identity) else None
        # If CBAM is Identity (baseline), hook the backbone feature output instead
        # by wrapping forward_features via a tensor hook at call time.

    def _save_act(self, module, inp, out):
        self._activations = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._gradients = grad

    def __call__(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Return (cam HxW in [0,1], predicted/target class index)."""
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        # Fallback path: if no CBAM hook, use the cached last_feature_map.
        if self._activations is None:
            feat = self.model.last_feature_map
            feat.retain_grad()
            self._activations = feat
            self._gradients = None

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        grads = self._gradients
        if grads is None and self.model.last_feature_map is not None:
            grads = self.model.last_feature_map.grad
        acts = self._activations

        weights = grads.mean(dim=(2, 3), keepdim=True)        # GAP over spatial
        cam = (weights * acts).sum(dim=1, keepdim=True)        # weighted combo
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Reset for next call.
        self._activations = None
        self._gradients = None
        return cam, target_class

    def close(self):
        if self._fwd is not None:
            self._fwd.remove()


def leaf_attention_fidelity(cam: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Fraction of CAM activation mass falling on the leaf foreground."""
    h, w = leaf_mask.shape
    cam_r = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    total = float(cam_r.sum())
    if total <= 1e-8:
        return float("nan")
    on_leaf = float((cam_r * leaf_mask).sum())
    return on_leaf / total


def overlay_cam(image_tensor: torch.Tensor, cam: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image with the CAM heat-map overlaid."""
    img = denormalize(image_tensor)  # HxWx3 in [0,1]
    h, w = img.shape[:2]
    cam_r = cv2.resize(cam.astype(np.float32), (w, h))
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = 0.55 * img + 0.45 * heat
    return (out.clip(0, 1) * 255).astype(np.uint8)


def compute_leaf_mask_from_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    img = (denormalize(image_tensor) * 255).astype(np.uint8)
    return segment_leaf(img)
