"""Synthetic abdominal CT phantoms.

Not a substitute for LiTS or CHAOS. They exist so the whole pipeline - cache,
training, sliding-window inference, geometry inversion, API - can be run and
tested end to end on any machine, and so unit tests never need patient data.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from .io import save_nifti

HU_AIR = -1000.0
HU_FAT = -90.0
HU_SOFT = 45.0
HU_LIVER = 110.0
HU_BONE = 400.0


def _ellipsoid(shape, centre, radii, rng=None, wobble: float = 0.0) -> np.ndarray:
    grids = np.ogrid[: shape[0], : shape[1], : shape[2]]
    value = sum(
        ((grid - c) / max(r, 1e-3)) ** 2 for grid, c, r in zip(grids, centre, radii)
    )
    mask = value <= 1.0
    if wobble and rng is not None:
        from scipy import ndimage as ndi

        noise = ndi.gaussian_filter(rng.normal(0, 1, size=shape), sigma=4)
        mask = (value + wobble * noise) <= 1.0
    return mask


def make_phantom(
    case_index: int,
    shape: Tuple[int, int, int] = (128, 128, 96),
    spacing=(1.6, 1.6, 2.0),
    seed: int = 0,
    orientation_flip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic CT volume plus its liver mask. Returns (ct, mask, affine)."""
    rng = np.random.RandomState(seed + case_index)
    shape = tuple(int(s) for s in shape)

    volume = np.full(shape, HU_AIR, dtype=np.float32)
    centre = np.array(shape) / 2.0

    body = _ellipsoid(shape, centre, (shape[0] * 0.42, shape[1] * 0.34, shape[2] * 0.48))
    volume[body] = HU_FAT
    organs = _ellipsoid(shape, centre, (shape[0] * 0.36, shape[1] * 0.28, shape[2] * 0.42))
    volume[organs] = HU_SOFT

    liver_centre = centre + np.array([
        -shape[0] * 0.12 + rng.uniform(-3, 3),
        rng.uniform(-3, 3),
        shape[2] * 0.10 + rng.uniform(-3, 3),
    ])
    liver_radii = np.array([shape[0] * 0.22, shape[1] * 0.17, shape[2] * 0.20]) * rng.uniform(0.85, 1.15, size=3)
    liver = _ellipsoid(shape, liver_centre, liver_radii, rng=rng, wobble=0.08)
    # a wedge cut gives the liver a non-convex lobe boundary
    wedge = _ellipsoid(shape, liver_centre + np.array([liver_radii[0] * 0.7, 0, -liver_radii[2] * 0.7]),
                       liver_radii * 0.55)
    liver = np.logical_and(liver, ~wedge)
    volume[liver] = HU_LIVER + rng.uniform(-8, 8)

    # neighbouring structures with liver-like intensity to make the task non-trivial
    spleen = _ellipsoid(shape, centre + np.array([shape[0] * 0.22, shape[1] * 0.05, shape[2] * 0.12]),
                        (shape[0] * 0.10, shape[1] * 0.08, shape[2] * 0.11))
    volume[spleen] = HU_LIVER - 6
    spine = _ellipsoid(shape, centre + np.array([0, shape[1] * 0.26, 0]),
                       (shape[0] * 0.07, shape[1] * 0.07, shape[2] * 0.46))
    volume[spine] = HU_BONE

    # lesions inside the liver: still liver by label definition
    for _ in range(rng.randint(0, 3)):
        lesion_centre = liver_centre + rng.uniform(-1, 1, size=3) * liver_radii * 0.5
        lesion = _ellipsoid(shape, lesion_centre, np.repeat(rng.uniform(3, 7), 3))
        volume[np.logical_and(lesion, liver)] = HU_LIVER - rng.uniform(25, 45)

    volume += rng.normal(0, 12, size=shape).astype(np.float32)  # acquisition noise

    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    affine[:3, 3] = [-shape[0] * spacing[0] / 2, -shape[1] * spacing[1] / 2, -shape[2] * spacing[2] / 2]
    if orientation_flip:
        # LAS-style volume: exercises the orientation handling and its inverse
        affine[0, 0] *= -1
        affine[1, 1] *= -1
        volume = volume[::-1, ::-1].copy()
        liver = liver[::-1, ::-1].copy()

    return volume.astype(np.float32), liver.astype(np.uint8), affine


def write_dataset(
    out_dir: str,
    n_cases: int = 8,
    shape=(128, 128, 96),
    spacing=(1.6, 1.6, 2.0),
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Write imagesTr/labelsTr NIfTI pairs. Returns the case records."""
    images_dir = os.path.join(out_dir, "imagesTr")
    labels_dir = os.path.join(out_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cases = []
    for index in range(n_cases):
        case_id = f"synth_{index:03d}"
        spacing_case = tuple(float(s) * float(np.random.RandomState(seed + index).uniform(0.9, 1.1)) for s in spacing)
        volume, mask, affine = make_phantom(
            index, shape=shape, spacing=spacing_case, seed=seed,
            orientation_flip=(index % 3 == 0),
        )
        image_path = save_nifti(volume, affine, os.path.join(images_dir, f"{case_id}.nii.gz"), dtype=np.float32)
        label_path = save_nifti(mask, affine, os.path.join(labels_dir, f"{case_id}.nii.gz"), dtype=np.uint8)
        cases.append({"case_id": case_id, "patient_id": case_id, "image": image_path, "label": label_path})
    return cases
