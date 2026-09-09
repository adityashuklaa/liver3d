"""Inference on a new volume, with inverse transforms and an audit record.

    python -m src.infer --checkpoint outputs/<run_id>/best.pt --input scan.nii.gz

The exported mask is written in the source geometry so it overlays the original
scan in any viewer. Output is a proposed segmentation and always carries
review_required=True.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .data.io import InputValidationError, load_volume, save_nifti, spacing_from_affine
from .data.preprocess import preprocess_case, restore_to_source
from .data.validate import check_input_volume
from .inference import sliding_window_inference
from .postprocess import postprocess
from .train import resolve_device
from .utils.checksum import file_sha256
from .utils.logging_utils import get_logger
from .visualize import case_report

LOG = get_logger("liver3d.infer")


class LiverSegmenter:
    """Loaded model + config, reusable across requests (used by the API)."""

    def __init__(self, checkpoint: str, device: str = "auto", config: Optional[str] = None):
        self.device = resolve_device(device)
        override = load_config(config) if config else None
        self.model, self.cfg, self.meta = load_checkpoint(checkpoint, device=str(self.device), cfg=override)
        self.checkpoint_path = os.path.abspath(checkpoint)
        self.model_version = self.meta.get("model_version", "unknown")
        self.preprocessing_version = "prep-" + str(self.cfg["data"]["spacing_mm"]) + str(
            self.cfg["data"]["intensity_window_hu"]
        )

    def predict_array(self, image: np.ndarray, affine: np.ndarray) -> Dict[str, Any]:
        """Run the full chain on an in-memory volume. Returns arrays + report."""
        warnings: List[str] = list(check_input_volume(image, affine, self.cfg))
        started = time.time()

        proc, _, geom = preprocess_case(image, affine, self.cfg, label=None)
        tensor = torch.from_numpy(proc).float().unsqueeze(0)
        inference_cfg = self.cfg["inference"]
        probs = sliding_window_inference(
            tensor,
            self.model,
            roi_size=inference_cfg.get("roi_size", self.cfg["data"]["patch_size"]),
            overlap=float(inference_cfg.get("overlap", 0.5)),
            sw_batch_size=int(inference_cfg.get("sw_batch_size", 2)),
            blend=str(inference_cfg.get("blend", "gaussian")),
            device=self.device,
            amp=bool(self.cfg["training"].get("amp", False)),
        ).squeeze(0).cpu().numpy()

        mask_proc, report = postprocess(probs, self.cfg, geom.dst_spacing_mm)
        mask_src, src_affine = restore_to_source(mask_proc.astype(np.float32), geom, order=0)
        mask_src = (mask_src >= 0.5).astype(np.uint8)
        prob_src = None
        if self.cfg["outputs"].get("save_probability_map", True):
            prob_src, _ = restore_to_source(probs, geom, order=1)

        if report["empty_prediction"]:
            warnings.append("empty prediction - flagged for mandatory review")
        elif report["volume_ml"] < float(self.cfg["data"].get("min_liver_volume_ml", 100.0)):
            warnings.append(
                f"predicted liver volume {report['volume_ml']:.0f} ml is below the plausible range"
            )
        if mask_src.shape != tuple(image.shape):
            warnings.append(f"restored mask shape {mask_src.shape} != source {tuple(image.shape)}")

        return {
            "mask": mask_src,
            "probability": prob_src,
            "mask_processed": mask_proc,
            "image_processed": proc,
            "affine": src_affine,
            "geometry": geom,
            "postprocess_report": report,
            "warnings": warnings,
            "runtime_seconds": round(time.time() - started, 3),
        }

    def predict_file(
        self,
        input_path: str,
        out_dir: str,
        job_id: Optional[str] = None,
        make_overlays: bool = True,
        requester: str = "cli",
    ) -> Dict[str, Any]:
        job_id = job_id or uuid.uuid4().hex[:16]
        os.makedirs(out_dir, exist_ok=True)
        image, affine = load_volume(input_path)
        result = self.predict_array(image, affine)

        mask_path = save_nifti(result["mask"], result["affine"],
                               os.path.join(out_dir, f"{job_id}_mask.nii.gz"), dtype=np.uint8)
        prob_path = None
        if result["probability"] is not None:
            prob_path = save_nifti(result["probability"].astype(np.float32), result["affine"],
                                   os.path.join(out_dir, f"{job_id}_prob.nii.gz"), dtype=np.float32)
        artifacts = {}
        if make_overlays and self.cfg["outputs"].get("save_overlays", True):
            artifacts = case_report(
                result["image_processed"], result["mask_processed"], out_dir, job_id,
                result["geometry"].dst_spacing_mm, reference=None, make_mesh=True,
            )

        audit = {
            "job_id": job_id,
            "status": "completed",
            "requester": requester,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_checksum_sha256": file_sha256(input_path) if os.path.isfile(input_path) else None,
            "input_shape": list(image.shape),
            "input_spacing_mm": [float(v) for v in spacing_from_affine(affine)],
            "model_version": self.model_version,
            "checkpoint_sha256": self.meta.get("checkpoint_sha256"),
            "preprocessing": {
                "orientation": self.cfg["data"]["orientation"],
                "spacing_mm": self.cfg["data"]["spacing_mm"],
                "intensity_window_hu": self.cfg["data"]["intensity_window_hu"],
            },
            "inference": {
                "roi_size": self.cfg["inference"]["roi_size"],
                "overlap": self.cfg["inference"]["overlap"],
                "threshold": self.cfg["inference"]["threshold"],
                "device": str(self.device),
            },
            "postprocess_report": result["postprocess_report"],
            "runtime_seconds": result["runtime_seconds"],
            "warnings": result["warnings"],
            "mask_format": "NIfTI",
            "outputs": {"mask": mask_path, "probability": prob_path, **artifacts},
            "review_required": True,
            "intended_use": "research prototype; not a cleared medical device",
        }
        audit_path = os.path.join(out_dir, f"{job_id}_audit.json")
        with open(audit_path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2)
        audit["audit_path"] = audit_path
        return audit


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment the liver in one CT volume")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NIfTI file or DICOM series directory")
    parser.add_argument("--out-dir", default="outputs/inference")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--no-overlays", dest="overlays", action="store_false", default=True)
    return parser.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    args = parse_args(argv)
    segmenter = LiverSegmenter(args.checkpoint, device=args.device, config=args.config)
    try:
        audit = segmenter.predict_file(
            args.input, args.out_dir, job_id=args.job_id, make_overlays=args.overlays
        )
    except InputValidationError as exc:
        LOG.error("input rejected: %s", exc)
        raise SystemExit(2) from exc
    print(json.dumps({k: v for k, v in audit.items() if k != "postprocess_report"}, indent=2))
    return audit


if __name__ == "__main__":  # pragma: no cover
    main()
