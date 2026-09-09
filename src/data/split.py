"""Case discovery, patient-level splitting and split validation.

The split is patient-level and locked in a CSV manifest before any patch is
extracted. Patch-level splitting would leak neighbouring regions of the same
patient into validation and inflate the score.
"""
from __future__ import annotations

import csv
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence

MANIFEST_FIELDS = ["case_id", "patient_id", "split", "image", "label"]

_NII_SUFFIXES = (".nii.gz", ".nii")


class SplitError(ValueError):
    """Raised when a split manifest is inconsistent."""


def _stem(path: str) -> str:
    name = os.path.basename(path)
    for suffix in _NII_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _list_volumes(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        if name.endswith(_NII_SUFFIXES):
            out.append(os.path.join(directory, name))
    return out


def default_patient_id(case_id: str, pattern: Optional[str] = None) -> str:
    """Derive a patient id from a case id.

    With no pattern, the case id is the patient id (one study per patient). A
    dataset with several studies per patient must pass a regex whose first
    group is the patient key, e.g. r'(patient\\d+)_study\\d+'.
    """
    if not pattern:
        return case_id
    match = re.search(pattern, case_id)
    if not match:
        raise SplitError(f"patient pattern {pattern!r} did not match case id {case_id!r}")
    return match.group(1)


def discover_cases(root: str, patient_pattern: Optional[str] = None) -> List[Dict[str, str]]:
    """Find image/label pairs in the supported dataset layouts."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise SplitError(f"dataset root does not exist: {root}")

    pairs: Dict[str, Dict[str, str]] = {}

    # Layout A: imagesTr/ + labelsTr/ (MSD, nnU-Net) or images/ + labels/
    for img_dir, lbl_dir in (("imagesTr", "labelsTr"), ("images", "labels")):
        images = _list_volumes(os.path.join(root, img_dir))
        if not images:
            continue
        labels = {_stem(p): p for p in _list_volumes(os.path.join(root, lbl_dir))}
        for image in images:
            case_id = _stem(image)
            label = labels.get(case_id) or labels.get(case_id.replace("_0000", ""))
            if label:
                pairs[case_id] = {"case_id": case_id, "image": image, "label": label}

    # Layout B: flat LiTS style volume-N.nii + segmentation-N.nii
    if not pairs:
        images = _list_volumes(root)
        labels = {}
        for path in images:
            stem = _stem(path)
            if stem.startswith(("segmentation", "label", "mask")):
                key = re.sub(r"^(segmentation|label|mask)[-_]?", "", stem)
                labels[key] = path
        for path in images:
            stem = _stem(path)
            if stem.startswith(("segmentation", "label", "mask")):
                continue
            key = re.sub(r"^(volume|image|ct)[-_]?", "", stem)
            label = labels.get(key)
            if label:
                pairs[stem] = {"case_id": stem, "image": path, "label": label}

    cases = []
    for case in sorted(pairs.values(), key=lambda c: c["case_id"]):
        case["patient_id"] = default_patient_id(case["case_id"], patient_pattern)
        cases.append(case)
    if not cases:
        raise SplitError(
            f"no image/label pairs found under {root}. Expected imagesTr/labelsTr, "
            "images/labels, or flat volume-*/segmentation-* files."
        )
    return cases


def make_split(
    cases: Sequence[Dict[str, str]],
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Assign every case to train/val/test, grouped by patient id."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SplitError(f"split ratios must sum to 1.0, got {ratios}")
    patients = sorted({c["patient_id"] for c in cases})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = int(round(ratios[1] * n))
    if n >= 3:
        n_val = max(1, n_val)
        n_train = min(n_train, n - n_val - 1)
    assignment: Dict[str, str] = {}
    for idx, patient in enumerate(patients):
        if idx < n_train:
            assignment[patient] = "train"
        elif idx < n_train + n_val:
            assignment[patient] = "val"
        else:
            assignment[patient] = "test"

    rows = []
    for case in cases:
        row = dict(case)
        row["split"] = assignment[case["patient_id"]]
        rows.append(row)
    return rows


def write_manifest(rows: Iterable[Dict[str, str]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    return path


def read_manifest(path: str, split: Optional[str] = None) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise SplitError(f"split manifest not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    missing = [f for f in MANIFEST_FIELDS if rows and f not in rows[0]]
    if missing:
        raise SplitError(f"manifest {path} is missing columns: {missing}")
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def validate_manifest(rows: Sequence[Dict[str, str]], check_files: bool = True) -> Dict[str, int]:
    """Fail on leakage, duplicates, unknown splits or missing files."""
    if not rows:
        raise SplitError("split manifest is empty")
    seen_patient_split: Dict[str, str] = {}
    seen_cases = set()
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        case_id, patient_id, split = row["case_id"], row["patient_id"], row["split"]
        if split not in counts:
            raise SplitError(f"unknown split {split!r} for case {case_id}")
        if case_id in seen_cases:
            raise SplitError(f"duplicate case id in manifest: {case_id}")
        seen_cases.add(case_id)
        previous = seen_patient_split.get(patient_id)
        if previous and previous != split:
            raise SplitError(
                f"patient {patient_id} appears in both {previous} and {split} (leakage)"
            )
        seen_patient_split[patient_id] = split
        counts[split] += 1
        if check_files:
            for key in ("image", "label"):
                path = row.get(key, "")
                if path and not os.path.exists(path):
                    raise SplitError(f"{key} file missing for case {case_id}: {path}")
    if counts["train"] == 0:
        raise SplitError("split manifest has no training cases")
    return counts
