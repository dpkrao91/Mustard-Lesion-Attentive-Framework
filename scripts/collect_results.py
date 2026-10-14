#!/usr/bin/env python3
"""Aggregate ``test_metrics.json`` from every run into one comparison table.

Outputs ``runs/summary.csv`` and prints a Markdown table you can paste straight
into the paper's results section.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUNS_DIR = Path("runs")


def main():
    rows = []
    for mfile in sorted(RUNS_DIR.glob("*/test_metrics.json")):
        run = mfile.parent.name
        m = json.loads(mfile.read_text())
        pc = m.get("per_class", {})
        rows.append({
            "run": run,
            "accuracy": round(m.get("accuracy", float("nan")), 4),
            "balanced_acc": round(m.get("balanced_accuracy", float("nan")), 4),
            "macro_f1": round(m.get("macro_f1", float("nan")), 4),
            "macro_auc": round(m.get("macro_auc", float("nan")), 4),
            "alternaria_recall": round(
                pc.get("Alternaria_Blight", {}).get("recall", float("nan")), 4),
            "params_M": round(m.get("trainable_params", 0) / 1e6, 2),
            "latency_ms": round(m.get("latency_ms", float("nan")), 2),
        })
    if not rows:
        print("No runs found under runs/. Train models first (see run_all.sh).")
        return
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    RUNS_DIR.mkdir(exist_ok=True)
    df.to_csv(RUNS_DIR / "summary.csv", index=False)
    print(df.to_markdown(index=False))
    print(f"\nSaved -> {RUNS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
