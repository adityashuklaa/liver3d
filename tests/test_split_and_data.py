"""Split validation (TC 03) and dataset/patch sampling tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import PatchDataset, VolumeDataset, load_cached, preprocess_and_cache, random_patch
from src.data.split import (
    SplitError,
    discover_cases,
    make_split,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def test_discovery_finds_pairs(dataset_dir):
    cases = discover_cases(dataset_dir)
    assert len(cases) == 4
    assert all(case["image"] != case["label"] for case in cases)


def test_split_is_patient_level_and_complete(dataset_dir):
    cases = discover_cases(dataset_dir)
    rows = make_split(cases, ratios=(0.5, 0.25, 0.25), seed=0)
    counts = validate_manifest(rows)
    assert sum(counts.values()) == len(cases)
    assert counts["train"] >= 1 and counts["val"] >= 1


def test_leakage_is_detected(dataset_dir, tmp_path):
    cases = discover_cases(dataset_dir)
    rows = make_split(cases, ratios=(0.5, 0.25, 0.25), seed=0)
    leaked = [dict(r) for r in rows]
    leaked[0]["patient_id"] = leaked[-1]["patient_id"]
    leaked[0]["split"] = "train"
    leaked[-1]["split"] = "test"
    with pytest.raises(SplitError, match="leakage"):
        validate_manifest(leaked)


def test_manifest_roundtrip(dataset_dir, tmp_path):
    cases = discover_cases(dataset_dir)
    rows = make_split(cases, seed=1)
    path = tmp_path / "split.csv"
    write_manifest(rows, str(path))
    reloaded = read_manifest(str(path))
    assert len(reloaded) == len(rows)
    assert read_manifest(str(path), split="train")


def test_missing_file_is_detected(dataset_dir):
    cases = discover_cases(dataset_dir)
    rows = make_split(cases, seed=1)
    rows[0]["image"] = rows[0]["image"] + ".missing"
    with pytest.raises(SplitError, match="missing"):
        validate_manifest(rows)


def test_cache_and_patch_sampling(manifest_path, tiny_cfg):
    rows = read_manifest(manifest_path, split="train")
    path = preprocess_and_cache(rows[0], tiny_cfg)
    image, label, fg_idx, geom = load_cached(path)
    assert image.shape == label.shape
    assert len(fg_idx) > 0

    rng = np.random.RandomState(0)
    patch = tuple(tiny_cfg["data"]["patch_size"])
    hits = 0
    for _ in range(20):
        img_patch, lbl_patch = random_patch(image, label, fg_idx, patch, 1.0, rng)
        assert img_patch.shape == patch and lbl_patch.shape == patch
        hits += int(lbl_patch.sum() > 0)
    assert hits >= 18  # foreground-biased sampling actually finds the liver


def test_patch_dataset_batches(manifest_path, tiny_cfg):
    rows = read_manifest(manifest_path, split="train")
    dataset = PatchDataset(rows, tiny_cfg, train=True, seed=0)
    assert len(dataset) == len(rows) * tiny_cfg["sampling"]["patches_per_volume"]
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (1, *tiny_cfg["data"]["patch_size"])
    assert float(sample["image"].min()) >= 0.0 and float(sample["image"].max()) <= 1.0
    assert set(np.unique(sample["label"].numpy())).issubset({0.0, 1.0})


def test_volume_dataset_returns_geometry(manifest_path, tiny_cfg):
    rows = read_manifest(manifest_path, split="val")
    dataset = VolumeDataset(rows, tiny_cfg)
    item = dataset[0]
    assert item["image"].dim() == 4
    assert item["geom"].orig_shape
    assert item["label"] is not None
