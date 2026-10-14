#!/usr/bin/env bash
# =============================================================================
# Reproduce every experiment reported in the paper.
# Each run writes metrics to runs/<name>/test_metrics.json
# =============================================================================
set -euo pipefail

CONFIG="configs/default.yaml"

echo "==> [0/8] Preparing stratified splits"
python scripts/prepare_splits.py --config "$CONFIG"

echo "==> [1/8] Baseline: ResNet50 + CE"
python scripts/train.py --config "$CONFIG" --model resnet50 --loss ce --no-cbam \
  --output-dir runs/resnet50_ce

echo "==> [2/8] Baseline: MobileNetV3-Large + CE"
python scripts/train.py --config "$CONFIG" --model mobilenetv3_large_100 --loss ce --no-cbam \
  --output-dir runs/mobilenetv3_ce

echo "==> [3/8] Baseline: EfficientNet-B0 + CE"
python scripts/train.py --config "$CONFIG" --model efficientnet_b0 --loss ce --no-cbam \
  --output-dir runs/effb0_ce

echo "==> [4/8] EfficientNet-B0 + weighted CE  (imbalance ablation)"
python scripts/train.py --config "$CONFIG" --model efficientnet_b0 --loss weighted_ce --no-cbam \
  --output-dir runs/effb0_wce

echo "==> [5/8] EfficientNet-B0 + CB-Focal     (loss ablation, no attention)"
python scripts/train.py --config "$CONFIG" --model efficientnet_b0 --loss cb_focal --no-cbam \
  --output-dir runs/effb0_cbfocal

echo "==> [6/8] EfficientNet-B0 + CBAM + CE     (attention ablation, plain loss)"
python scripts/train.py --config "$CONFIG" --model efficientnet_b0 --loss ce --use-cbam \
  --output-dir runs/effb0_cbam_ce

echo "==> [7/8] PALM (proposed): EfficientNet-B0 + CBAM + CB-Focal"
python scripts/train.py --config "$CONFIG" --model efficientnet_b0 --loss cb_focal --use-cbam \
  --output-dir runs/palm

echo "==> [8/8] Explainability: Grad-CAM + Leaf-Attention-Fidelity for PALM"
python scripts/explain.py --config "$CONFIG" --checkpoint runs/palm/best.pt

echo "All runs complete. Collect metrics with:"
echo "  python scripts/collect_results.py"
