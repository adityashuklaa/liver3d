"""Shared fixtures: tiny synthetic dataset, tiny config, tiny checkpoint."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config  # noqa: E402
from src.data.split import make_split, write_manifest  # noqa: E402
from src.data.synth import write_dataset  # noqa: E402


TINY_OVERRIDES = {
    "seed": 0,
    "data": {
        "spacing_mm": [4.0, 4.0, 4.0],
        "patch_size": [16, 16, 16],
        "min_liver_volume_ml": 5.0,
        "max_liver_volume_ml": 5000.0,
        "spacing_valid_range_mm": [0.4, 10.0],
    },
    "sampling": {"pos_ratio": 0.9, "patches_per_volume": 2},
    "model": {"channels": [4, 8]},
    "training": {
        "batch_size": 2,
        "max_epochs": 2,
        "iters_per_epoch": 3,
        "amp": False,
        "num_workers": 0,
        "learning_rate": 0.005,
        "early_stopping_patience": 5,
    },
    "inference": {"roi_size": [16, 16, 16], "sw_batch_size": 2, "min_component_ml": 0.5},
    "outputs": {"save_overlays": False},
}


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    write_dataset(str(root), n_cases=4, shape=(48, 48, 40), spacing=(3.0, 3.0, 3.0), seed=1)
    return str(root)


@pytest.fixture(scope="session")
def manifest_path(dataset_dir, tmp_path_factory):
    from src.data.split import discover_cases

    cases = discover_cases(dataset_dir)
    rows = make_split(cases, ratios=(0.5, 0.25, 0.25), seed=0)
    path = str(tmp_path_factory.mktemp("manifest") / "split.csv")
    write_manifest(rows, path)
    return path


@pytest.fixture(scope="session")
def tiny_cfg(manifest_path, tmp_path_factory):
    overrides = dict(TINY_OVERRIDES)
    cfg = load_config(None, overrides=overrides)
    cfg["data"]["split_manifest"] = manifest_path
    cfg["data"]["cache_dir"] = str(tmp_path_factory.mktemp("cache"))
    cfg["outputs"]["dir"] = str(tmp_path_factory.mktemp("outputs"))
    return cfg


@pytest.fixture(scope="session")
def phantom():
    """One in-memory phantom: (volume, mask, affine)."""
    from src.data.synth import make_phantom

    return make_phantom(0, shape=(48, 48, 40), spacing=(3.0, 3.0, 3.0), seed=3)


@pytest.fixture(scope="session")
def checkpoint_path(tiny_cfg, tmp_path_factory):
    """An untrained but structurally valid checkpoint for API/inference tests."""
    from src.checkpoint import save_checkpoint
    from src.models.unet3d import build_model

    model = build_model(tiny_cfg)
    path = str(tmp_path_factory.mktemp("ckpt") / "best.pt")
    save_checkpoint(path, model, tiny_cfg, meta={"run_id": "test", "epoch": 0})
    return path
