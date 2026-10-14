#!/usr/bin/env python3
"""Create reproducible stratified train/val/test splits.

Scans ``data.root`` for the three class sub-folders and writes
``train.csv``, ``val.csv``, ``test.csv`` (columns: path,label,class_name) into
``data.splits_dir``. Stratification preserves the per-class ratio in every
split, which matters given the documented imbalance.

Usage::

    python scripts/prepare_splits.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.dataset import CLASS_NAMES, CLASS_TO_IDX  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect(root: Path):
    rows = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder '{class_dir}'. See data/README.md for "
                f"the required layout.")
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({
                    "path": str(p.relative_to(root)),
                    "label": CLASS_TO_IDX[class_name],
                    "class_name": class_name,
                })
    if not rows:
        raise RuntimeError(f"No images found under {root}.")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.data.root)
    splits_dir = Path(cfg.data.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))

    df = collect(root)
    print(f"Found {len(df)} images.")
    print(df["class_name"].value_counts().to_string())

    train_df, temp_df = train_test_split(
        df, test_size=args.val_frac + args.test_frac,
        stratify=df["label"], random_state=seed)
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["label"], random_state=seed)

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = splits_dir / f"{name}.csv"
        part.to_csv(out, index=False)
        print(f"  {name:<5}: {len(part):>5} -> {out}")

    print("\nDone. Splits are deterministic for seed =", seed)


if __name__ == "__main__":
    main()
