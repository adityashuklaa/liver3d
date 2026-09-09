"""Locked evaluation entry point.

    python -m src.evaluate --checkpoint outputs/<run_id>/best.pt --split test

Metrics are computed in the source voxel grid of each case (after inverting
preprocessing), so the numbers describe the exported mask a clinician would
open, not an internal resampled grid. Every eligible case is reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .data.dataset import VolumeDataset, build_cache
from .data.io import load_volume, save_nifti, spacing_from_affine
from .data.preprocess import restore_to_source
from .data.split import read_manifest, validate_manifest
from .inference import sliding_window_inference
from .metrics import aggregate, case_metrics
from .postprocess import postprocess
from .train import resolve_device
from .utils.checksum import file_sha256
from .utils.env import write_environment
from .utils.logging_utils import get_logger
from .visualize import case_report

LOG = get_logger("liver3d.evaluate")

CSV_FIELDS = [
    "case_id", "split", "dice", "iou", "precision", "recall", "hd95_mm", "assd_mm",
    "pred_volume_ml", "ref_volume_ml", "volume_error_ml", "n_components",
    "inference_seconds", "warnings",
]


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args.device)
    override_cfg = load_config(args.config) if args.config else None
    model, cfg, meta = load_checkpoint(args.checkpoint, device=str(device), cfg=override_cfg)
    if override_cfg is not None:
        LOG.warning("config override in use; preprocessing may differ from the trained release")

    manifest_path = args.manifest or cfg["data"]["split_manifest"]
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no cases in split {args.split!r} of {manifest_path}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, "predictions")
    overlay_dir = os.path.join(out_dir, "overlays")
    write_environment(os.path.join(out_dir, "env.json"))

    build_cache(rows, cfg)
    dataset = VolumeDataset(rows, cfg)
    inference_cfg = cfg["inference"]

    results: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        case_id = item["case_id"]
        row = item["row"]
        started = time.time()
        probs = sliding_window_inference(
            item["image"],
            model,
            roi_size=inference_cfg.get("roi_size", cfg["data"]["patch_size"]),
            overlap=float(inference_cfg.get("overlap", 0.5)),
            sw_batch_size=int(inference_cfg.get("sw_batch_size", 2)),
            blend=str(inference_cfg.get("blend", "gaussian")),
            device=device,
            amp=bool(cfg["training"].get("amp", False)),
        )
        elapsed = time.time() - started
        prob_np = probs.squeeze(0).cpu().numpy()

        geom = item["geom"]
        mask_proc, report = postprocess(prob_np, cfg, geom.dst_spacing_mm)
        mask_src, src_affine = restore_to_source(mask_proc.astype(np.float32), geom, order=0)
        mask_src = (mask_src >= 0.5).astype(np.uint8)

        reference, ref_affine = load_volume(row["label"])
        reference = (reference > 0).astype(np.uint8)
        spacing = spacing_from_affine(ref_affine)
        metrics = case_metrics(mask_src, reference, spacing, with_surface=not args.no_surface)

        warnings: List[str] = []
        if report["empty_prediction"]:
            warnings.append("empty prediction - mandatory review")
        if mask_src.shape != reference.shape:
            warnings.append(f"restored shape {mask_src.shape} != reference {reference.shape}")

        record = {
            "case_id": case_id,
            "split": args.split,
            "inference_seconds": round(elapsed, 3),
            "warnings": "; ".join(warnings),
            **metrics,
        }
        results.append(record)
        LOG.info(
            "%s dice %.4f hd95 %s (%.1fs)",
            case_id,
            metrics["dice"],
            f"{metrics.get('hd95_mm'):.2f}" if metrics.get("hd95_mm") is not None else "n/a",
            elapsed,
        )

        if args.save_predictions:
            save_nifti(mask_src, src_affine, os.path.join(pred_dir, f"{case_id}_mask.nii.gz"), dtype=np.uint8)
            if cfg["outputs"].get("save_probability_map", False):
                prob_src, _ = restore_to_source(prob_np, geom, order=1)
                save_nifti(prob_src.astype(np.float32), src_affine,
                           os.path.join(pred_dir, f"{case_id}_prob.nii.gz"), dtype=np.float32)

    results.sort(key=lambda r: r["dice"])
    csv_path = os.path.join(out_dir, "case_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in results:
            writer.writerow({k: record.get(k, "") for k in CSV_FIELDS})

    summary = aggregate([{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in results])
    payload = {
        "split": args.split,
        "n_cases": len(results),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "model_version": meta.get("model_version"),
        "manifest": os.path.abspath(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "threshold": float(inference_cfg.get("threshold", 0.5)),
        "postprocessing": {
            "largest_component": bool(inference_cfg.get("largest_component", False)),
            "fill_holes": bool(inference_cfg.get("fill_holes", False)),
            "min_component_ml": float(inference_cfg.get("min_component_ml", 0.0)),
        },
        "metrics": summary,
        "worst_cases": [r["case_id"] for r in results[:3]],
        "best_cases": [r["case_id"] for r in results[-3:]],
        "cases_with_warnings": [r["case_id"] for r in results if r["warnings"]],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if args.overlays and results:
        picks = {results[0]["case_id"]: "worst", results[len(results) // 2]["case_id"]: "median",
                 results[-1]["case_id"]: "best"}
        for index in range(len(dataset)):
            item = dataset[index]
            if item["case_id"] not in picks:
                continue
            image = item["image"].squeeze(0).numpy()
            label = item["label"].squeeze(0).numpy() if item["label"] is not None else None
            probs = sliding_window_inference(
                item["image"], model,
                roi_size=inference_cfg.get("roi_size", cfg["data"]["patch_size"]),
                overlap=float(inference_cfg.get("overlap", 0.5)),
                sw_batch_size=int(inference_cfg.get("sw_batch_size", 2)),
                blend=str(inference_cfg.get("blend", "gaussian")), device=device,
            ).squeeze(0).cpu().numpy()
            mask_proc, _ = postprocess(probs, cfg, item["geom"].dst_spacing_mm)
            case_report(
                image, mask_proc, overlay_dir,
                f"{picks[item['case_id']]}_{item['case_id']}",
                item["geom"].dst_spacing_mm, reference=label, make_mesh=True,
            )

    print(json.dumps({"n_cases": payload["n_cases"],
                      "dice_mean": summary.get("dice", {}).get("mean"),
                      "dice_ci95": [summary.get("dice", {}).get("ci95_low"), summary.get("dice", {}).get("ci95_high")],
                      "output": os.path.abspath(out_dir)}, indent=2))
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a locked split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="override the config stored in the checkpoint")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.add_argument("--overlays", action="store_true", default=True)
    parser.add_argument("--no-overlays", dest="overlays", action="store_false")
    parser.add_argument("--no-surface", action="store_true", help="skip HD95/ASSD (faster)")
    return parser.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    return evaluate(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
