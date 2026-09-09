"""Postprocessing rules applied to a probability map.

Order: threshold -> drop implausibly small components -> optionally keep the
largest component -> optionally fill enclosed holes. Every step is switchable
from the config because none of them is safe on every dataset (fragmented
pathology, partial fields of view).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi


def threshold(probability: np.ndarray, value: float = 0.5) -> np.ndarray:
    return (np.asarray(probability) >= float(value)).astype(np.uint8)


def remove_small_components(mask: np.ndarray, spacing_mm, min_ml: float) -> Tuple[np.ndarray, int]:
    if min_ml <= 0 or not mask.any():
        return mask, 0
    voxel_ml = float(np.prod(np.asarray(spacing_mm, dtype=float))) / 1000.0
    labelled, count = ndi.label(mask)
    if count == 0:
        return mask, 0
    sizes = ndi.sum(mask, labelled, index=np.arange(1, count + 1))
    keep = (sizes * voxel_ml) >= float(min_ml)
    if keep.all():
        return mask, 0
    out = np.isin(labelled, np.nonzero(keep)[0] + 1).astype(np.uint8)
    return out, int((~keep).sum())


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    labelled, count = ndi.label(mask)
    if count <= 1:
        return mask
    sizes = ndi.sum(mask, labelled, index=np.arange(1, count + 1))
    return (labelled == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    # 2D slice-wise plus 3D fill: a 3D fill alone leaves vessel channels that
    # reach the volume border unfilled.
    out = ndi.binary_fill_holes(mask.astype(bool))
    for index in range(out.shape[2]):
        out[:, :, index] = ndi.binary_fill_holes(out[:, :, index])
    return out.astype(np.uint8)


def postprocess(probability: np.ndarray, cfg, spacing_mm) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the configured rules. Returns (mask uint8, report)."""
    inference_cfg = cfg["inference"]
    report: Dict[str, object] = {}

    mask = threshold(probability, float(inference_cfg.get("threshold", 0.5)))
    report["voxels_after_threshold"] = int(mask.sum())

    mask, dropped = remove_small_components(
        mask, spacing_mm, float(inference_cfg.get("min_component_ml", 0.0))
    )
    report["small_components_removed"] = dropped

    if bool(inference_cfg.get("largest_component", False)):
        before = int(ndi.label(mask)[1])
        mask = largest_component(mask)
        report["components_before_largest"] = before

    if bool(inference_cfg.get("fill_holes", False)):
        mask = fill_holes(mask)

    report["voxels_final"] = int(mask.sum())
    voxel_ml = float(np.prod(np.asarray(spacing_mm, dtype=float))) / 1000.0
    report["volume_ml"] = float(mask.sum() * voxel_ml)
    report["empty_prediction"] = bool(mask.sum() == 0)
    return mask.astype(np.uint8), report
