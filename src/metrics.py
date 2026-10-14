"""Classification metrics with an emphasis on imbalance-aware reporting.

Accuracy alone is misleading on this corpus (predicting only White Rust already
yields ~52%). We therefore report macro-F1, balanced accuracy and full
per-class precision/recall/F1, plus the confusion matrix and one-vs-rest AUC.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, class_names: List[str]) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }

    # One-vs-rest macro AUC (guard against single-class edge cases).
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["macro_auc"] = float("nan")

    return metrics


def pretty_print(metrics: Dict, class_names: List[str]) -> str:
    lines = [
        f"accuracy           : {metrics['accuracy']:.4f}",
        f"balanced accuracy  : {metrics['balanced_accuracy']:.4f}",
        f"macro F1           : {metrics['macro_f1']:.4f}",
        f"weighted F1        : {metrics['weighted_f1']:.4f}",
        f"macro AUC (ovr)    : {metrics['macro_auc']:.4f}",
        f"Cohen's kappa      : {metrics['cohen_kappa']:.4f}",
        "",
        "per-class:",
    ]
    for name in class_names:
        pc = metrics["per_class"][name]
        lines.append(
            f"  {name:<18} P={pc['precision']:.3f} "
            f"R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")
    return "\n".join(lines)
