"""Data quality gates and input eligibility checks.

Hard failures raise InputValidationError; softer findings are returned as
warnings so a case can be flagged instead of silently accepted.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import InputValidationError, spacing_from_affine


def check_geometry(image_affine: np.ndarray, label_affine: np.ndarray, tol: float = 1e-3) -> None:
    if not np.allclose(np.asarray(image_affine), np.asarray(label_affine), atol=tol):
        raise InputValidationError(
            "image and label do not occupy the same physical space (affine mismatch)"
        )


def check_shape(image: np.ndarray, label: np.ndarray) -> None:
    if image.shape != label.shape:
        raise InputValidationError(
            f"image shape {image.shape} does not match label shape {label.shape}"
        )


def check_labels(label: np.ndarray, allowed=(0, 1, 2), merge_above_zero: bool = True) -> List[str]:
    values = np.unique(label)
    warnings: List[str] = []
    illegal = [float(v) for v in values if float(v) not in [float(a) for a in allowed]]
    if illegal and not merge_above_zero:
        raise InputValidationError(f"label contains unexpected values: {illegal}")
    if illegal:
        warnings.append(f"labels above the documented classes merged to liver: {illegal}")
    if float(values.max()) <= 0:
        raise InputValidationError("label mask is empty")
    return warnings


def liver_volume_ml(mask: np.ndarray, spacing_mm) -> float:
    voxel_ml = float(np.prod(np.asarray(spacing_mm, dtype=float))) / 1000.0
    return float(mask.astype(bool).sum()) * voxel_ml


def check_volume_plausible(mask: np.ndarray, spacing_mm, cfg) -> List[str]:
    vol = liver_volume_ml(mask, spacing_mm)
    lo = float(cfg["data"].get("min_liver_volume_ml", 100.0))
    hi = float(cfg["data"].get("max_liver_volume_ml", 4000.0))
    if vol <= 0:
        raise InputValidationError("liver mask is empty after preprocessing")
    if vol < lo or vol > hi:
        return [f"liver volume {vol:.0f} ml is outside the plausible range [{lo:.0f}, {hi:.0f}] ml"]
    return []


def check_input_volume(image: np.ndarray, affine: np.ndarray, cfg) -> List[str]:
    """Eligibility gate for inference inputs. Raises on unusable volumes."""
    warnings: List[str] = []
    if image.ndim != 3:
        raise InputValidationError(f"expected a 3D volume, got {image.ndim}D")
    if min(image.shape) < 8:
        raise InputValidationError(f"volume too small for 3D inference: {image.shape}")
    if not np.all(np.isfinite(image)):
        raise InputValidationError("volume contains non-finite voxels (corrupt slices)")
    spacing = spacing_from_affine(affine)
    if not np.all(np.isfinite(spacing)) or float(spacing.min()) <= 0:
        raise InputValidationError(f"invalid voxel spacing derived from affine: {spacing}")
    lo, hi = [float(v) for v in cfg["data"].get("spacing_valid_range_mm", [0.4, 6.0])]
    if float(spacing.min()) < lo or float(spacing.max()) > hi:
        warnings.append(
            f"voxel spacing {np.round(spacing, 3).tolist()} mm is outside the validated "
            f"range [{lo}, {hi}] mm; result requires review"
        )
    lo_hu, hi_hu = float(image.min()), float(image.max())
    if hi_hu <= lo_hu:
        raise InputValidationError("volume has no intensity variation")
    if hi_hu < 100 or lo_hu > -100:
        warnings.append(
            f"intensity range [{lo_hu:.0f}, {hi_hu:.0f}] does not look like Hounsfield units"
        )
    return warnings


def validate_training_case(
    image: np.ndarray,
    image_affine: np.ndarray,
    label: np.ndarray,
    label_affine: np.ndarray,
    cfg,
) -> Dict[str, object]:
    """Full pre-training gate for one image/label pair."""
    warnings: List[str] = []
    check_shape(image, label)
    check_geometry(image_affine, label_affine)
    warnings += check_input_volume(image, image_affine, cfg)
    warnings += check_labels(label, merge_above_zero=cfg["data"].get("merge_labels_above_zero", True))
    warnings += check_volume_plausible(label > 0, spacing_from_affine(label_affine), cfg)
    return {"passed": True, "warnings": warnings}


def check_no_identifier(text: str) -> Optional[str]:
    """Cheap guard against obvious identifiers in filenames or log fields."""
    lowered = str(text).lower()
    for token in ("patientname", "mrn", "dob", "ssn", "birth"):
        if token in lowered:
            return f"possible identifier token '{token}' in '{text}'"
    return None


def summarize(warnings: List[str]) -> Tuple[bool, List[str]]:
    return (len(warnings) == 0), warnings
