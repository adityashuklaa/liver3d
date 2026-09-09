"""Discover cases and lock a patient-level split manifest.

    python scripts/build_manifest.py --root data/synthetic \
        --out data_manifests/split_v1.csv --ratios 0.7 0.15 0.15

Re-running with the same seed and dataset reproduces the same split. The
manifest is the record of the split; never re-split silently before a test-set
evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import (  # noqa: E402
    discover_cases,
    make_split,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from src.utils.checksum import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate a split manifest")
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--out", default="data_manifests/split_v1.csv")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patient-pattern",
        default=None,
        help=r"regex whose first group is the patient key, e.g. '(patient\d+)_study\d+'",
    )
    parser.add_argument("--author", default=os.environ.get("USERNAME") or os.environ.get("USER") or "unknown")
    args = parser.parse_args()

    cases = discover_cases(args.root, patient_pattern=args.patient_pattern)
    rows = make_split(cases, ratios=tuple(args.ratios), seed=args.seed)
    write_manifest(rows, args.out)
    counts = validate_manifest(read_manifest(args.out))

    record = {
        "manifest": os.path.abspath(args.out),
        "manifest_sha256": file_sha256(args.out),
        "dataset_root": os.path.abspath(args.root),
        "n_cases": len(rows),
        "n_patients": len({r["patient_id"] for r in rows}),
        "counts": counts,
        "ratios": args.ratios,
        "seed": args.seed,
        "author": args.author,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    side_car = os.path.splitext(args.out)[0] + "_meta.json"
    with open(side_car, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
