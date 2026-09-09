"""Export a browser-ready study bundle: slice sprites, masks, metrics, mesh.

    python scripts/export_viewer_data.py --checkpoint outputs/demo/best.pt --out outputs/viewer

For each case it runs the real inference path, then writes:
  <case>_ct.png    grayscale CT sprite sheet (all slices, HU-windowed)
  <case>_pred.png  predicted mask sprite sheet (white on black)
  <case>_ref.png   reference mask sprite sheet
  bundle.json      geometry, per-case metrics, per-slice Dice, mesh, run provenance

The viewer draws from these, so what a reviewer sees on screen is the exported
mask itself, not a re-rendered approximation of it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import ndimage as ndi  # noqa: E402

from src.data.io import load_volume, spacing_from_affine  # noqa: E402
from src.data.split import read_manifest  # noqa: E402
from src.infer import LiverSegmenter  # noqa: E402
from src.metrics import case_metrics, dice  # noqa: E402
from src.utils.checksum import file_sha256  # noqa: E402


def sprite(volume_2d_stack: np.ndarray, cols: int) -> np.ndarray:
    """Tile (H, W, S) into one (rows*H, cols*W) sheet, slice order left-to-right."""
    h, w, s = volume_2d_stack.shape
    rows = int(np.ceil(s / cols))
    sheet = np.zeros((rows * h, cols * w), dtype=volume_2d_stack.dtype)
    for index in range(s):
        r, c = divmod(index, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = volume_2d_stack[:, :, index]
    return sheet


def save_png(array_u8: np.ndarray, path: str) -> str:
    from PIL import Image

    Image.fromarray(array_u8).save(path, optimize=True)
    return path


def window_to_u8(volume: np.ndarray, lo: float, hi: float) -> np.ndarray:
    clipped = np.clip(volume, lo, hi)
    return np.rint((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a study bundle for the web viewer")
    parser.add_argument("--checkpoint", default="outputs/demo/best.pt")
    parser.add_argument("--manifest", default="data_manifests/synthetic_split.csv")
    parser.add_argument("--splits", nargs="+", default=["test", "val"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", default="outputs/viewer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cols", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    segmenter = LiverSegmenter(args.checkpoint, device=args.device)

    rows = []
    for split in args.splits:
        rows += [dict(r) for r in read_manifest(args.manifest, split=split)]
    rows = rows[: args.limit]

    window = segmenter.cfg["data"]["intensity_window_hu"]
    cases = []
    for row in rows:
        case_id = row["case_id"]
        image, affine = load_volume(row["image"])
        reference = (load_volume(row["label"])[0] > 0).astype(np.uint8)
        spacing = spacing_from_affine(affine)

        started = time.time()
        result = segmenter.predict_array(image, affine)
        elapsed = time.time() - started
        pred = result["mask"].astype(np.uint8)

        metrics = case_metrics(pred, reference, spacing)
        per_slice = []
        for index in range(image.shape[2]):
            p, r = pred[:, :, index], reference[:, :, index]
            if p.sum() == 0 and r.sum() == 0:
                per_slice.append(None)
            else:
                per_slice.append(round(float(dice(p, r)), 4))

        # sprite sheets: CT once, masks as separate single-channel sheets
        ct_u8 = window_to_u8(image, float(window[0]), float(window[1]))
        save_png(sprite(ct_u8, args.cols), os.path.join(args.out, f"{case_id}_ct.png"))
        save_png(sprite(pred * 255, args.cols), os.path.join(args.out, f"{case_id}_pred.png"))
        save_png(sprite(reference * 255, args.cols), os.path.join(args.out, f"{case_id}_ref.png"))

        # decimated surface for the 3D tab
        small = ndi.zoom(pred.astype(np.float32), 0.5, order=1)
        mesh = None
        if small.sum() > 8:
            from skimage import measure

            verts, faces, _, _ = measure.marching_cubes(small, level=0.5, spacing=tuple(spacing * 2))
            verts = verts - verts.mean(0)
            mesh = {
                "x": [round(float(v), 2) for v in verts[:, 0]],
                "y": [round(float(v), 2) for v in verts[:, 1]],
                "z": [round(float(v), 2) for v in verts[:, 2]],
                "i": [int(v) for v in faces[:, 0]],
                "j": [int(v) for v in faces[:, 1]],
                "k": [int(v) for v in faces[:, 2]],
            }

        cases.append({
            "case_id": case_id,
            "split": row["split"],
            "shape": [int(v) for v in image.shape],
            "cols": args.cols,
            "spacing_mm": [round(float(v), 3) for v in spacing],
            "window_hu": [float(window[0]), float(window[1])],
            "hu_range": [float(image.min()), float(image.max())],
            "metrics": {k: (None if v is None else round(float(v), 4)) for k, v in metrics.items()},
            "per_slice_dice": per_slice,
            "runtime_seconds": round(elapsed, 3),
            "warnings": result["warnings"],
            "postprocess": result["postprocess_report"],
            "mesh": mesh,
        })
        print(f"{case_id:>12}  dice {metrics['dice']:.4f}  hd95 {metrics['hd95_mm']:.2f} mm  {elapsed:.2f}s")

    bundle = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": segmenter.model_version,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(segmenter.device),
        "config": {
            "spacing_mm": segmenter.cfg["data"]["spacing_mm"],
            "patch_size": segmenter.cfg["data"]["patch_size"],
            "window_hu": segmenter.cfg["data"]["intensity_window_hu"],
            "threshold": segmenter.cfg["inference"]["threshold"],
            "overlap": segmenter.cfg["inference"]["overlap"],
            "channels": segmenter.cfg["model"]["channels"],
        },
        "cases": cases,
    }
    with open(os.path.join(args.out, "bundle.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh)
    print("bundle ->", os.path.abspath(os.path.join(args.out, "bundle.json")))


if __name__ == "__main__":
    main()
