"""Create a synthetic CT dataset so the pipeline can run without patient data.

    python scripts/make_synthetic_dataset.py --out data/synthetic --cases 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synth import write_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic abdominal CT phantoms")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--shape", type=int, nargs=3, default=[128, 128, 96])
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.6, 1.6, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = write_dataset(
        args.out, n_cases=args.cases, shape=tuple(args.shape),
        spacing=tuple(args.spacing), seed=args.seed,
    )
    print(f"wrote {len(cases)} synthetic cases to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
