"""End-to-end smoke tests: train -> evaluate -> infer (TC 01, TC 05, TC 06)."""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
import pytest

from src.config import save_config
from src.data.split import read_manifest
from src.evaluate import main as evaluate_main
from src.infer import LiverSegmenter
from src.infer import main as infer_main
from src.train import main as train_main


@pytest.fixture(scope="module")
def trained_run(tiny_cfg, tmp_path_factory):
    config_path = str(tmp_path_factory.mktemp("cfg") / "tiny.yaml")
    save_config(tiny_cfg, config_path)
    summary = train_main([
        "--config", config_path,
        "--device", "cpu",
        "--run-id", "pytest_run",
        "--max-epochs", "2",
        "--iters-per-epoch", "2",
    ])
    return {"config": config_path, **summary}


def test_training_writes_reproducibility_artifacts(trained_run):
    run_dir = trained_run["run_dir"]
    for name in ("config.yaml", "env.json", "split_manifest.csv", "train_log.jsonl",
                 "metrics.csv", "best.pt", "last.pt", "run_summary.json"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    assert trained_run["best_checkpoint_sha256"]
    assert trained_run["model_version"].startswith("liver-unet-")
    with open(os.path.join(run_dir, "train_log.jsonl"), encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records and "train_loss" in records[0]


def test_evaluation_produces_case_metrics(trained_run):
    payload = evaluate_main([
        "--checkpoint", trained_run["best_checkpoint"],
        "--split", "test",
        "--device", "cpu",
        "--no-overlays",
        "--no-surface",
    ])
    assert payload["n_cases"] >= 1
    assert "dice" in payload["metrics"]
    out_dir = os.path.dirname(payload["checkpoint"])
    csv_path = os.path.join(out_dir, "eval_test", "case_metrics.csv")
    assert os.path.exists(csv_path)
    assert payload["manifest_sha256"] and payload["checkpoint_sha256"]


def test_inference_restores_source_geometry(trained_run, tiny_cfg, tmp_path):
    rows = read_manifest(tiny_cfg["data"]["split_manifest"], split="test")
    case = rows[0]
    audit = infer_main([
        "--checkpoint", trained_run["best_checkpoint"],
        "--input", case["image"],
        "--out-dir", str(tmp_path / "inference"),
        "--device", "cpu",
        "--no-overlays",
    ])
    assert audit["status"] == "completed"
    assert audit["review_required"] is True
    assert audit["model_version"].startswith("liver-unet-")

    source = nib.load(case["image"])
    mask = nib.load(audit["outputs"]["mask"])
    assert mask.shape == source.shape
    assert np.allclose(mask.affine, source.affine, atol=1e-4)
    assert set(np.unique(np.asanyarray(mask.dataobj))).issubset({0, 1})
    assert os.path.exists(audit["audit_path"])


def test_segmenter_rejects_invalid_input(trained_run, tmp_path):
    from src.data.io import InputValidationError, save_nifti

    segmenter = LiverSegmenter(trained_run["best_checkpoint"], device="cpu")
    tiny = np.zeros((4, 4, 4), dtype=np.float32)
    path = save_nifti(tiny, np.eye(4), str(tmp_path / "too_small.nii.gz"))
    with pytest.raises(InputValidationError):
        segmenter.predict_file(path, str(tmp_path / "out"))


def test_checkpoint_bundle_carries_config(trained_run):
    from src.checkpoint import load_checkpoint

    model, cfg, meta = load_checkpoint(trained_run["best_checkpoint"], device="cpu")
    assert cfg["data"]["patch_size"] == [16, 16, 16]
    assert meta["checkpoint_sha256"]
    assert meta["run_id"] == "pytest_run"
