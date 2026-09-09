"""Versioned experiment configuration.

One YAML file drives preprocessing, model, training, inference and export so a
run can be reproduced from outputs/<run_id>/config.yaml alone.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "project": "liver_3d_unet",
    "seed": 42,
    "data": {
        "root": "data/synthetic",
        "dataset": "TO_CONFIRM",
        "split_manifest": "data_manifests/split_v1.csv",
        "cache_dir": "data/cache",
        "orientation": "RAS",
        "spacing_mm": [1.5, 1.5, 2.0],
        "intensity_window_hu": [-200.0, 250.0],
        "patch_size": [96, 96, 96],
        "merge_labels_above_zero": True,
        "spacing_valid_range_mm": [0.4, 6.0],
        "min_liver_volume_ml": 100.0,
        "max_liver_volume_ml": 4000.0,
    },
    "augment": {
        "flip_prob": 0.5,
        "intensity_shift": 0.1,
        "intensity_scale": 0.1,
        "noise_std": 0.02,
    },
    "sampling": {
        "pos_ratio": 0.7,
        "patches_per_volume": 2,
    },
    "model": {
        "architecture": "unet3d",
        "in_channels": 1,
        "channels": [32, 64, 128, 256],
        "out_channels": 1,
        "norm": "instance",
        "activation": "leakyrelu",
        "dropout": 0.0,
    },
    "training": {
        "loss": "dice_bce",
        "dice_weight": 1.0,
        "bce_weight": 1.0,
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "batch_size": 2,
        "max_epochs": 200,
        "iters_per_epoch": 100,
        "early_stopping_patience": 30,
        "amp": True,
        "num_workers": 0,
        "val_interval": 1,
        "checkpoint_metric": "dice",
    },
    "inference": {
        "roi_size": [96, 96, 96],
        "overlap": 0.5,
        "blend": "gaussian",
        "sw_batch_size": 2,
        "threshold": 0.5,
        "largest_component": True,
        "fill_holes": True,
        "min_component_ml": 10.0,
    },
    "outputs": {
        "dir": "outputs",
        "save_probability_map": True,
        "restore_original_geometry": True,
        "save_overlays": True,
    },
}


class Config(dict):
    """Dict with attribute access, deep-merged over DEFAULTS."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None, overrides: Dict[str, Any] | None = None) -> Config:
    """Load YAML config merged over DEFAULTS. Missing path means defaults only."""
    raw: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    merged = _deep_merge(DEFAULTS, raw)
    merged = _deep_merge(merged, overrides or {})
    merged["_source_path"] = os.path.abspath(path) if path else None
    return Config(merged)


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def config_hash(cfg: Config, keys=("data",)) -> str:
    """Stable hash over config sections that invalidate cached artifacts."""
    subset = {k: cfg.get(k) for k in keys}
    blob = json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
