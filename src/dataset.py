"""Dataset and dataloader construction for the mustard leaf disease corpus.

Expected on-disk layout (ImageFolder style)::

    data/raw/
        Healthy/            *.jpg
        Alternaria_Blight/  *.jpg
        White_Rust/         *.jpg

``scripts/prepare_splits.py`` consumes this layout and emits stratified
train/val/test CSV files under ``data/splits/``. This module then reads those
CSVs so that every experiment uses *exactly* the same partition (reproducibility).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Canonical class ordering. Keep this fixed so checkpoints stay compatible.
CLASS_NAMES: Tuple[str, ...] = ("Healthy", "Alternaria_Blight", "White_Rust")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


class MustardLeafDataset(Dataset):
    """Reads (image_path, label) pairs from a split CSV."""

    def __init__(self, csv_path: str | Path, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        df = pd.read_csv(csv_path)
        if not {"path", "label"}.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain 'path' and 'label' columns")
        self.samples: List[Tuple[str, int]] = list(zip(df["path"], df["label"]))
        self.labels = np.asarray(df["label"].tolist(), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        path = self.root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for c in self.labels:
            counts[c] += 1
        return counts


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampler — a second, complementary imbalance remedy to
    the class-balanced loss. Each minority sample is drawn more often per epoch."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weight = 1.0 / counts
    sample_weight = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def build_dataloaders(
    cfg,
    train_tf,
    eval_tf,
    splits_dir: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, MustardLeafDataset]:
    """Construct train / val / test dataloaders from the prepared splits."""
    splits_dir = Path(splits_dir or cfg.data.splits_dir)
    root = Path(cfg.data.root)

    train_ds = MustardLeafDataset(splits_dir / "train.csv", root, train_tf)
    val_ds = MustardLeafDataset(splits_dir / "val.csv", root, eval_tf)
    test_ds = MustardLeafDataset(splits_dir / "test.csv", root, eval_tf)

    use_sampler = bool(cfg.data.get("use_weighted_sampler", False))
    if use_sampler:
        sampler = make_weighted_sampler(train_ds.labels)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds
