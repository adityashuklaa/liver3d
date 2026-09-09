"""Unit tests for the invertible preprocessing chain (TC 01, TC 02, TC 06)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.io import InputValidationError, spacing_from_affine
from src.data.preprocess import (
    from_ras,
    preprocess_case,
    resample_to_shape,
    restore_to_source,
    to_ras,
    window_scale,
)
from src.data.validate import check_geometry, check_input_volume, validate_training_case


def test_orientation_roundtrip(phantom):
    volume, mask, affine = phantom
    ras, ras_affine, ornt = to_ras(volume, affine)
    back = from_ras(ras, ornt)
    assert back.shape == volume.shape
    assert np.array_equal(back, volume)


def test_orientation_makes_axes_ras(phantom):
    volume, _, affine = phantom
    _, ras_affine, _ = to_ras(volume, affine)
    import nibabel as nib

    assert nib.orientations.aff2axcodes(ras_affine) == ("R", "A", "S")


def test_window_scale_bounds():
    data = np.array([-1000.0, -200.0, 25.0, 250.0, 3000.0], dtype=np.float32)
    scaled = window_scale(data, [-200.0, 250.0])
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert scaled[2] == pytest.approx((25.0 + 200.0) / 450.0)
    with pytest.raises(ValueError):
        window_scale(data, [250.0, -200.0])


def test_resample_to_shape_exact():
    data = np.random.RandomState(0).rand(20, 18, 16).astype(np.float32)
    out = resample_to_shape(data, (10, 9, 8), order=1)
    assert out.shape == (10, 9, 8)


def test_preprocess_then_restore_keeps_geometry(phantom, tiny_cfg):
    volume, mask, affine = phantom
    image, label, geom = preprocess_case(volume, affine, tiny_cfg, label=mask)

    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert label is not None and set(np.unique(label)).issubset({0, 1})
    assert np.allclose(spacing_from_affine(np.array(geom.proc_affine)), tiny_cfg["data"]["spacing_mm"], atol=1e-6)

    restored, restored_affine = restore_to_source(label.astype(np.float32), geom, order=0)
    assert restored.shape == volume.shape
    assert np.allclose(restored_affine, affine)

    # the label survives the round trip well enough to stay the same organ
    restored_bin = restored >= 0.5
    intersection = np.logical_and(restored_bin, mask > 0).sum()
    dice = 2 * intersection / (restored_bin.sum() + (mask > 0).sum())
    assert dice > 0.85


def test_geometry_mismatch_is_rejected(phantom, tiny_cfg):
    volume, mask, affine = phantom
    shifted = affine.copy()
    shifted[0, 3] += 25.0
    with pytest.raises(InputValidationError):
        check_geometry(affine, shifted)
    with pytest.raises(InputValidationError):
        validate_training_case(volume, affine, mask, shifted, tiny_cfg)


def test_shape_mismatch_is_rejected(phantom, tiny_cfg):
    volume, mask, affine = phantom
    with pytest.raises(InputValidationError):
        validate_training_case(volume, affine, mask[:-2], affine, tiny_cfg)


def test_input_eligibility_warns_outside_spacing_range(phantom, tiny_cfg):
    volume, _, affine = phantom
    coarse = affine.copy()
    coarse[:3, :3] *= 5.0
    warnings = check_input_volume(volume, coarse, tiny_cfg)
    assert any("spacing" in w for w in warnings)


def test_corrupt_volume_is_rejected(phantom, tiny_cfg):
    volume, _, affine = phantom
    broken = volume.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(InputValidationError):
        check_input_volume(broken, affine, tiny_cfg)
