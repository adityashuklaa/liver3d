"""Caching dataset and foreground-aware patch sampling.

Preprocessing is deterministic, so each case is preprocessed once and cached as
an .npz next to a geometry record. Training draws patches from the cache;
validation and evaluation use whole volumes with sliding-window inference.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import config_hash
from .io import load_volume
from .preprocess import GeometryRecord, preprocess_case
from .validate import validate_training_case

MAX_CACHED_FG_VOXELS = 20000


def _cache_path(cfg, case_id: str) -> str:
    root = cfg["data"].get("cache_dir", "data/cache")
    return os.path.join(root, config_hash(cfg), f"{case_id}.npz")


def preprocess_and_cache(
    row: Dict[str, str],
    cfg,
    force: bool = False,
    validate: bool = True,
) -> str:
    """Preprocess one manifest row into the cache. Returns the cache path."""
    path = _cache_path(cfg, row["case_id"])
    if os.path.exists(path) and not force:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image, affine = load_volume(row["image"])
    label = None
    if row.get("label"):
        label, label_affine = load_volume(row["label"])
        if validate:
            validate_training_case(image, affine, label, label_affine, cfg)

    img, lbl, geom = preprocess_case(image, affine, cfg, label=label)

    fg_idx = np.zeros((0, 3), dtype=np.int32)
    if lbl is not None and lbl.any():
        coords = np.argwhere(lbl > 0).astype(np.int32)
        if len(coords) > MAX_CACHED_FG_VOXELS:
            sel = np.random.RandomState(0).choice(
                len(coords), MAX_CACHED_FG_VOXELS, replace=False
            )
            coords = coords[sel]
        fg_idx = coords

    np.savez_compressed(
        path,
        image=img.astype(np.float32),
        label=(lbl if lbl is not None else np.zeros((0,), np.uint8)).astype(np.uint8),
        fg_idx=fg_idx,
        geom=np.frombuffer(json.dumps(geom.to_dict()).encode("utf-8"), dtype=np.uint8),
    )
    return path


def load_cached(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, GeometryRecord]:
    with np.load(path, allow_pickle=False) as data:
        image = data["image"]
        label = data["label"]
        fg_idx = data["fg_idx"]
        geom_json = data["geom"].tobytes().decode("utf-8")
    geom = GeometryRecord.from_dict(json.loads(geom_json))
    label = label if label.ndim == 3 else None
    return image, label, fg_idx, geom


def build_cache(rows: Sequence[Dict[str, str]], cfg, force: bool = False) -> List[str]:
    return [preprocess_and_cache(row, cfg, force=force) for row in rows]


def _pad_to_patch(image: np.ndarray, label: Optional[np.ndarray], patch):
    """Zero-pad a volume smaller than the patch. Returns (image, label, pad_before)."""
    pads = []
    for cur, want in zip(image.shape, patch):
        deficit = max(0, want - cur)
        pads.append((deficit // 2, deficit - deficit // 2))
    pad_before = np.array([p[0] for p in pads], dtype=int)
    if all(p == (0, 0) for p in pads):
        return image, label, pad_before
    image = np.pad(image, pads, mode="constant", constant_values=0.0)
    if label is not None:
        label = np.pad(label, pads, mode="constant", constant_values=0)
    return image, label, pad_before


def random_patch(
    image: np.ndarray,
    label: Optional[np.ndarray],
    fg_idx: np.ndarray,
    patch,
    pos_ratio: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one patch, biased towards liver voxels with probability pos_ratio."""
    image, label, pad_before = _pad_to_patch(image, label, patch)
    shape = np.array(image.shape)
    patch = np.array(patch, dtype=int)

    use_fg = len(fg_idx) > 0 and rng.rand() < pos_ratio
    if use_fg:
        # foreground coordinates are in unpadded space; shift onto the padded grid
        centre = fg_idx[rng.randint(len(fg_idx))].astype(int) + pad_before
        jitter = rng.randint(-patch // 4, patch // 4 + 1, size=3)
        start = centre - patch // 2 + jitter
    else:
        start = np.array([rng.randint(0, max(1, s - p + 1)) for s, p in zip(shape, patch)])
    start = np.clip(start, 0, np.maximum(shape - patch, 0))

    slices = tuple(slice(int(s), int(s + p)) for s, p in zip(start, patch))
    img_patch = image[slices]
    lbl_patch = label[slices] if label is not None else np.zeros(patch, dtype=np.uint8)
    return img_patch.astype(np.float32), lbl_patch.astype(np.float32)


def augment(image: np.ndarray, label: np.ndarray, cfg, rng: np.random.RandomState):
    aug = cfg["augment"]
    for axis in range(3):
        if rng.rand() < float(aug.get("flip_prob", 0.0)):
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    image = np.ascontiguousarray(image)
    label = np.ascontiguousarray(label)

    shift = float(aug.get("intensity_shift", 0.0))
    scale = float(aug.get("intensity_scale", 0.0))
    noise = float(aug.get("noise_std", 0.0))
    if scale > 0:
        image = image * (1.0 + rng.uniform(-scale, scale))
    if shift > 0:
        image = image + rng.uniform(-shift, shift)
    if noise > 0:
        image = image + rng.normal(0.0, noise, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32), label.astype(np.float32)


class PatchDataset(Dataset):
    """Random foreground-aware patches from cached training volumes."""

    def __init__(self, rows: Sequence[Dict[str, str]], cfg, train: bool = True, seed: int = 0):
        self.rows = list(rows)
        self.cfg = cfg
        self.train = train
        self.patch = tuple(int(v) for v in cfg["data"]["patch_size"])
        self.pos_ratio = float(cfg["sampling"].get("pos_ratio", 0.7))
        self.per_volume = int(cfg["sampling"].get("patches_per_volume", 1))
        self.seed = seed
        self.paths = [_cache_path(cfg, r["case_id"]) for r in self.rows]
        missing = [p for p in self.paths if not os.path.exists(p)]
        if missing:
            build_cache(self.rows, cfg)

    def __len__(self) -> int:
        return len(self.rows) * self.per_volume

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_index = index % len(self.rows)
        rng = np.random.RandomState((self.seed + index * 7919) % (2**31 - 1))
        image, label, fg_idx, _ = load_cached(self.paths[case_index])
        img, lbl = random_patch(image, label, fg_idx, self.patch, self.pos_ratio, rng)
        if self.train:
            img, lbl = augment(img, lbl, self.cfg, rng)
        return {
            "image": torch.from_numpy(img).unsqueeze(0),
            "label": torch.from_numpy(lbl).unsqueeze(0),
            "case_id": self.rows[case_index]["case_id"],
        }


class VolumeDataset(Dataset):
    """Whole preprocessed volumes for validation, evaluation and export."""

    def __init__(self, rows: Sequence[Dict[str, str]], cfg):
        self.rows = list(rows)
        self.cfg = cfg
        self.paths = [_cache_path(cfg, r["case_id"]) for r in self.rows]
        if any(not os.path.exists(p) for p in self.paths):
            build_cache(self.rows, cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image, label, _, geom = load_cached(self.paths[index])
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": None if label is None else torch.from_numpy(label.astype(np.float32)).unsqueeze(0),
            "geom": geom,
            "case_id": self.rows[index]["case_id"],
            "row": self.rows[index],
        }
