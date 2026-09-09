"""Volume input/output for NIfTI and (optionally) DICOM series.

Everything downstream works on a plain float32 array plus a 4x4 affine, so the
source format never leaks into the model code.
"""
from __future__ import annotations

import os
from typing import Tuple

import nibabel as nib
import numpy as np


class InputValidationError(ValueError):
    """Raised for unsupported, corrupt or spatially invalid input."""


def load_nifti(path) -> Tuple[np.ndarray, np.ndarray, "nib.Nifti1Header"]:
    try:
        img = nib.load(str(path))
    except Exception as exc:  # pragma: no cover - corrupt file path
        raise InputValidationError(f"cannot read NIfTI volume: {exc}") from exc
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise InputValidationError(f"expected a 3D volume, got shape {data.shape}")
    return data.astype(np.float32), img.affine.astype(np.float64), img.header


def load_dicom_series(directory) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a single-series DICOM directory into a volume plus affine.

    Requires pydicom. Slices are sorted by ImagePositionPatient projected on the
    slice normal so that acquisition order does not affect geometry.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise InputValidationError(
            "DICOM input requires pydicom (pip install pydicom)"
        ) from exc

    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if not f.startswith(".")
    ]
    slices = []
    for path in files:
        try:
            ds = pydicom.dcmread(path)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        slices.append(ds)
    if not slices:
        raise InputValidationError("no readable DICOM slices found")

    series_uids = {getattr(s, "SeriesInstanceUID", "") for s in slices}
    if len(series_uids) > 1:
        raise InputValidationError("directory contains more than one DICOM series")

    orient = np.array(getattr(slices[0], "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    row, col = orient[:3], orient[3:]
    normal = np.cross(row, col)
    slices.sort(key=lambda s: float(np.dot(np.array(s.ImagePositionPatient, float), normal)))

    array = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
    slope = float(getattr(slices[0], "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0) or 0.0)
    array = array * slope + intercept  # Hounsfield units

    py, px = [float(v) for v in getattr(slices[0], "PixelSpacing", [1.0, 1.0])]
    if len(slices) > 1:
        p0 = np.array(slices[0].ImagePositionPatient, float)
        p1 = np.array(slices[1].ImagePositionPatient, float)
        pz = float(np.linalg.norm(p1 - p0)) or float(getattr(slices[0], "SliceThickness", 1.0))
    else:
        pz = float(getattr(slices[0], "SliceThickness", 1.0))

    affine = np.eye(4)
    # DICOM patient coordinates are LPS; NIfTI is RAS, so negate x and y.
    lps_to_ras = np.array([-1.0, -1.0, 1.0])
    affine[:3, 0] = row * px * lps_to_ras
    affine[:3, 1] = col * py * lps_to_ras
    affine[:3, 2] = normal * pz * lps_to_ras
    affine[:3, 3] = np.array(slices[0].ImagePositionPatient, float) * lps_to_ras

    # Array axes are (rows, cols, slices) = (col-direction, row-direction, normal);
    # transpose to (x, y, z) matching the affine column order.
    array = np.transpose(array, (1, 0, 2))
    meta = {
        "n_slices": len(slices),
        "modality": str(getattr(slices[0], "Modality", "")),
        "rows": int(slices[0].Rows),
        "columns": int(slices[0].Columns),
    }
    return array.astype(np.float32), affine, meta


def load_volume(path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file or DICOM directory. Returns (array, affine)."""
    path = str(path)
    if os.path.isdir(path):
        data, affine, _ = load_dicom_series(path)
        return data, affine
    if path.endswith((".nii", ".nii.gz", ".hdr", ".img")):
        data, affine, _ = load_nifti(path)
        return data, affine
    raise InputValidationError(f"unsupported input format: {path}")


def save_nifti(array: np.ndarray, affine: np.ndarray, path, dtype=None) -> str:
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if dtype is not None:
        array = array.astype(dtype)
    img = nib.Nifti1Image(array, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, path)
    return path


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
