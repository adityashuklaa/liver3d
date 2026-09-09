"""Deterministic, invertible preprocessing.

Order (fixed, see docs/validation_report_template.md):
  load -> validate -> orient to RAS -> resample to target spacing ->
  clip HU window and scale to [0, 1]

Every step that changes geometry writes into a GeometryRecord, so a predicted
mask can be pushed back onto the exact source voxel grid.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .io import spacing_from_affine

RAS_ORNT = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])


@dataclass
class GeometryRecord:
    """Everything needed to invert preprocessing back to source geometry."""

    orig_shape: List[int]
    orig_affine: List[List[float]]
    ornt: List[List[float]]
    ras_shape: List[int]
    ras_affine: List[List[float]]
    src_spacing_mm: List[float]
    dst_spacing_mm: List[float]
    proc_shape: List[int]
    proc_affine: List[List[float]]
    intensity_window_hu: List[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "GeometryRecord":
        return GeometryRecord(**payload)

    @staticmethod
    def from_json(path: str) -> "GeometryRecord":
        with open(path, "r", encoding="utf-8") as fh:
            return GeometryRecord.from_dict(json.load(fh))


def to_ras(data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorient an array to canonical RAS. Returns (data, affine, ornt)."""
    ornt = nib.orientations.io_orientation(affine)
    out = nib.orientations.apply_orientation(data, ornt)
    new_affine = np.asarray(affine).dot(nib.orientations.inv_ornt_aff(ornt, data.shape))
    return np.ascontiguousarray(out), new_affine, ornt


def from_ras(data_ras: np.ndarray, ornt: np.ndarray) -> np.ndarray:
    """Inverse of to_ras: RAS array back to the source array orientation."""
    inv = nib.orientations.ornt_transform(RAS_ORNT, np.asarray(ornt, dtype=float))
    return np.ascontiguousarray(nib.orientations.apply_orientation(data_ras, inv))


def _zoom(data: np.ndarray, factors: np.ndarray, order: int) -> np.ndarray:
    if np.allclose(factors, 1.0):
        return data
    mode = "nearest"
    return ndi.zoom(data, zoom=factors, order=order, mode=mode, prefilter=order > 1)


def _fit_shape(data: np.ndarray, target_shape) -> np.ndarray:
    """Crop or edge-pad so the array matches target_shape exactly."""
    target = tuple(int(v) for v in target_shape)
    if data.shape == target:
        return data
    slices, pads = [], []
    for cur, tgt in zip(data.shape, target):
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
            pads.append((0, 0))
        else:
            slices.append(slice(0, cur))
            before = (tgt - cur) // 2
            pads.append((before, tgt - cur - before))
    out = data[tuple(slices)]
    if any(p != (0, 0) for p in pads):
        out = np.pad(out, pads, mode="edge")
    return out


def resample_to_shape(data: np.ndarray, target_shape, order: int) -> np.ndarray:
    factors = np.array(target_shape, dtype=float) / np.array(data.shape, dtype=float)
    return _fit_shape(_zoom(data, factors, order), target_shape)


def resample_to_spacing(
    data: np.ndarray,
    affine: np.ndarray,
    src_spacing,
    dst_spacing,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to a target voxel spacing and return (data, updated affine)."""
    src = np.asarray(src_spacing, dtype=float)
    dst = np.asarray(dst_spacing, dtype=float)
    factors = src / dst
    out = _zoom(data, factors, order)
    achieved = np.array(out.shape, dtype=float) / np.array(data.shape, dtype=float)
    new_affine = np.array(affine, dtype=float).copy()
    scale = 1.0 / achieved
    new_affine[:3, :3] = np.asarray(affine)[:3, :3] @ np.diag(scale)
    # corner-align the resampled grid with the source grid
    shift = np.asarray(affine)[:3, :3] @ (0.5 * scale - 0.5)
    new_affine[:3, 3] = np.asarray(affine)[:3, 3] + shift
    return out, new_affine


def window_scale(data: np.ndarray, window) -> np.ndarray:
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        raise ValueError(f"invalid intensity window: {window}")
    out = np.clip(data, lo, hi)
    return ((out - lo) / (hi - lo)).astype(np.float32)


def preprocess_case(
    image: np.ndarray,
    affine: np.ndarray,
    cfg,
    label: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], GeometryRecord]:
    """Run the fixed preprocessing chain on one case.

    Returns (image [0,1] float32, binary label uint8 or None, geometry record).
    """
    data_cfg = cfg["data"]
    dst_spacing = np.asarray(data_cfg["spacing_mm"], dtype=float)
    window = data_cfg["intensity_window_hu"]

    orig_shape = list(image.shape)
    orig_affine = np.asarray(affine, dtype=float)

    img_ras, ras_affine, ornt = to_ras(image, orig_affine)
    lbl_ras = to_ras(label, orig_affine)[0] if label is not None else None
    src_spacing = spacing_from_affine(ras_affine)

    img_rs, proc_affine = resample_to_spacing(img_ras, ras_affine, src_spacing, dst_spacing, order=1)
    if lbl_ras is not None:
        lbl_rs = resample_to_shape(lbl_ras, img_rs.shape, order=0)
        if data_cfg.get("merge_labels_above_zero", True):
            lbl_rs = (lbl_rs > 0.5).astype(np.uint8)
        else:
            lbl_rs = np.rint(lbl_rs).astype(np.uint8)
    else:
        lbl_rs = None

    img_out = window_scale(img_rs, window)

    geom = GeometryRecord(
        orig_shape=orig_shape,
        orig_affine=orig_affine.tolist(),
        ornt=np.asarray(ornt, dtype=float).tolist(),
        ras_shape=list(img_ras.shape),
        ras_affine=np.asarray(ras_affine, dtype=float).tolist(),
        src_spacing_mm=[float(v) for v in src_spacing],
        dst_spacing_mm=[float(v) for v in dst_spacing],
        proc_shape=list(img_out.shape),
        proc_affine=np.asarray(proc_affine, dtype=float).tolist(),
        intensity_window_hu=[float(window[0]), float(window[1])],
    )
    return img_out, lbl_rs, geom


def restore_to_source(
    array_proc: np.ndarray,
    geom: GeometryRecord,
    order: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert resampling and orientation. Returns (array, source affine)."""
    ras = resample_to_shape(array_proc.astype(np.float32), geom.ras_shape, order=order)
    src = from_ras(ras, np.asarray(geom.ornt, dtype=float))
    src = _fit_shape(src, geom.orig_shape)
    return src, np.asarray(geom.orig_affine, dtype=float)
