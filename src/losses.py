"""Loss functions for class-imbalanced classification.

Implements four options selectable from the config:

* ``ce``          -- plain cross-entropy (baseline).
* ``weighted_ce`` -- inverse-frequency weighted cross-entropy.
* ``focal``       -- focal loss (Lin et al., 2017).
* ``cb_focal``    -- class-balanced focal loss (Cui et al., CVPR 2019),
                     the loss used by the proposed PALM framework.

The class-balanced weighting uses the *effective number of samples*
``E_n = (1 - beta^n) / (1 - beta)`` so the per-class weight is
``(1 - beta) / (1 - beta^{n_c})``. With the documented counts
(Healthy=2000, Alternaria=783, White Rust=3000) this up-weights the
Alternaria minority without the instability of raw inverse frequency.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    counts[counts == 0] = 1.0
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)  # normalise to mean 1
    return torch.as_tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        ce = F.nll_loss(log_prob, target, weight=self.weight, reduction="none")
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(cfg, class_counts: Sequence[int]) -> nn.Module:
    name = cfg.loss.name
    gamma = float(cfg.loss.get("gamma", 2.0))
    beta = float(cfg.loss.get("beta", 0.999))

    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        counts = np.asarray(class_counts, dtype=np.float64)
        counts[counts == 0] = 1.0
        w = torch.as_tensor((counts.sum() / (len(counts) * counts)), dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=None)
    if name == "cb_focal":
        w = effective_number_weights(class_counts, beta=beta)
        return FocalLoss(gamma=gamma, weight=w)
    raise ValueError(f"Unknown loss '{name}'")
