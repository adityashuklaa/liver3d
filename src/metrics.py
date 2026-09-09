"""Case-level segmentation metrics in physical units.

Overlap metrics (Dice, IoU, precision, recall) plus surface metrics (HD95,
average symmetric surface distance) computed with voxel spacing in millimetres.
Aggregates report mean, median, sd, IQR and a bootstrap 95% CI.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi

EPS = 1e-8


def _binary(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(bool)


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = _binary(pred), _binary(ref)
    denom = pred.sum() + ref.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, ref).sum() / denom)


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = _binary(pred), _binary(ref)
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, ref).sum() / union)


def precision(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = _binary(pred), _binary(ref)
    if pred.sum() == 0:
        return 1.0 if ref.sum() == 0 else 0.0
    return float(np.logical_and(pred, ref).sum() / pred.sum())


def recall(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = _binary(pred), _binary(ref)
    if ref.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(np.logical_and(pred, ref).sum() / ref.sum())


def volume_ml(mask: np.ndarray, spacing_mm) -> float:
    voxel_ml = float(np.prod(np.asarray(spacing_mm, dtype=float))) / 1000.0
    return float(_binary(mask).sum() * voxel_ml)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = _binary(mask)
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    return np.logical_xor(mask, eroded)


def surface_distances(pred: np.ndarray, ref: np.ndarray, spacing_mm) -> np.ndarray:
    """Symmetric set of surface-to-surface distances in millimetres."""
    pred_surface, ref_surface = _surface(pred), _surface(ref)
    if not pred_surface.any() or not ref_surface.any():
        return np.array([np.nan])
    spacing = np.asarray(spacing_mm, dtype=float)
    dt_ref = ndi.distance_transform_edt(~ref_surface, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    return np.concatenate([dt_ref[pred_surface], dt_pred[ref_surface]])


def hd95(pred: np.ndarray, ref: np.ndarray, spacing_mm) -> float:
    distances = surface_distances(pred, ref, spacing_mm)
    if np.all(np.isnan(distances)):
        return float("nan")
    return float(np.percentile(distances, 95))


def assd(pred: np.ndarray, ref: np.ndarray, spacing_mm) -> float:
    distances = surface_distances(pred, ref, spacing_mm)
    if np.all(np.isnan(distances)):
        return float("nan")
    return float(np.mean(distances))


def case_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing_mm,
    with_surface: bool = True,
) -> Dict[str, float]:
    pred, ref = _binary(pred), _binary(ref)
    out = {
        "dice": dice(pred, ref),
        "iou": iou(pred, ref),
        "precision": precision(pred, ref),
        "recall": recall(pred, ref),
        "pred_volume_ml": volume_ml(pred, spacing_mm),
        "ref_volume_ml": volume_ml(ref, spacing_mm),
        "n_components": int(ndi.label(pred)[1]),
    }
    out["volume_error_ml"] = out["pred_volume_ml"] - out["ref_volume_ml"]
    if with_surface:
        out["hd95_mm"] = hd95(pred, ref, spacing_mm)
        out["assd_mm"] = assd(pred, ref, spacing_mm)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return (float("nan"), float("nan"))
    if clean.size == 1:
        return (float(clean[0]), float(clean[0]))
    rng = np.random.RandomState(seed)
    means = np.array(
        [rng.choice(clean, size=clean.size, replace=True).mean() for _ in range(n_boot)]
    )
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def aggregate(rows: List[Dict[str, float]], n_boot: int = 2000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Mean / median / sd / IQR / bootstrap CI per metric over all cases."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray(
            [r[key] for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))],
            dtype=float,
        )
        if values.size == 0:
            continue
        lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
        summary[key] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "min": float(values.min()),
            "max": float(values.max()),
            "ci95_low": lo,
            "ci95_high": hi,
        }
    return summary
