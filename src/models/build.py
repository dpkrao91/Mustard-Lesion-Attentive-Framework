"""Model factory.

``build_model`` returns a backbone (any timm identifier) with an optional CBAM
attention block inserted between the final convolutional feature map and the
classifier. This single switch (``model.use_cbam``) is what separates a plain
baseline from the proposed PALM model, enabling a clean ablation.

The forward pass exposes ``.last_feature_map`` (a registered attribute) so that
Grad-CAM and the leaf-attention-fidelity metric can hook the correct layer
regardless of which backbone is chosen.
"""
from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from .cbam import CBAM


class AttentiveClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 use_cbam: bool = True, pretrained: bool = True,
                 drop_rate: float = 0.2):
        super().__init__()
        # features_only=False but num_classes=0 -> backbone returns pooled feat.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.feature_dim = self.backbone.num_features
        self.use_cbam = use_cbam
        self.cbam = CBAM(self.feature_dim) if use_cbam else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.last_feature_map: Optional[torch.Tensor] = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        # Some timm models already return pooled vectors; guard for 4-D maps.
        if feat.dim() == 2:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        feat = self.cbam(feat)
        self.last_feature_map = feat  # cached for Grad-CAM
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = self.pool(feat).flatten(1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_model(cfg, num_classes: int) -> AttentiveClassifier:
    return AttentiveClassifier(
        backbone_name=cfg.model.backbone,
        num_classes=num_classes,
        use_cbam=bool(cfg.model.get("use_cbam", True)),
        pretrained=bool(cfg.model.get("pretrained", True)),
        drop_rate=float(cfg.model.get("drop_rate", 0.2)),
    )


def build_model_for_eval(cfg, checkpoint_config, num_classes: int) -> AttentiveClassifier:
    """Build the architecture that matches a checkpoint.

    Eval/inference scripts must reconstruct the *exact* architecture the weights
    were trained with — otherwise a checkpoint trained with, say, ``--no-cbam``
    or a different backbone will not load. We therefore prefer the model
    settings stored inside the checkpoint over whatever is in the YAML, and we
    never re-download ImageNet weights (they are about to be overwritten).
    """
    model_cfg = {}
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_cfg = dict(checkpoint_config["model"])
    return AttentiveClassifier(
        backbone_name=model_cfg.get("backbone", cfg.model.backbone),
        num_classes=num_classes,
        use_cbam=bool(model_cfg.get("use_cbam", cfg.model.get("use_cbam", True))),
        pretrained=False,
        drop_rate=float(model_cfg.get("drop_rate", cfg.model.get("drop_rate", 0.2))),
    )


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
