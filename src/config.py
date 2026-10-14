"""Lightweight YAML configuration loader with dotted-attribute access."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict that also supports attribute access (cfg.train.epochs)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a :class:`Config`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}
    return Config(raw)


def to_plain(obj: Any) -> Any:
    """Recursively convert Config/dicts into plain JSON-serialisable dicts."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def merge_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Override leaf keys using dotted paths, e.g. {'train.epochs': 5}."""
    cfg = Config(copy.deepcopy(dict(cfg)))
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node: Dict[str, Any] = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared CLI arguments used across the training / evaluation scripts."""
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the backbone name (timm identifier).")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["ce", "weighted_ce", "focal", "cb_focal"],
                        help="Override the loss function.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--use-cbam", dest="use_cbam", action="store_true", default=None,
                        help="Force-enable the CBAM attention head.")
    parser.add_argument("--no-cbam", dest="use_cbam", action="store_false", default=None,
                        help="Force-disable the CBAM attention head.")
    return parser


def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Map parsed CLI flags onto the loaded configuration."""
    mapping = {
        "model": "model.backbone",
        "loss": "loss.name",
        "epochs": "train.epochs",
        "batch_size": "train.batch_size",
        "lr": "train.lr",
        "seed": "seed",
        "output_dir": "output_dir",
        "use_cbam": "model.use_cbam",
    }
    overrides = {}
    for cli_key, dotted in mapping.items():
        value = getattr(args, cli_key, None)
        if value is not None:
            overrides[dotted] = value
    return merge_overrides(cfg, overrides)
