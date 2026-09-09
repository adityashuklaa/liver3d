"""End-to-end demo: synthetic data -> manifest -> train -> evaluate -> infer.

    python scripts/quickstart.py

Runs on CPU in a few minutes. It proves the pipeline is wired correctly; it is
not a performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config  # noqa: E402
from src.data.split import read_manifest  # noqa: E402


def run(cmd) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole pipeline on synthetic data")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--config", default="configs/synthetic_demo.yaml")
    args = parser.parse_args()

    python = sys.executable
    cfg = load_config(os.path.join(ROOT, args.config))

    run([python, "scripts/make_synthetic_dataset.py", "--out", cfg["data"]["root"], "--cases", str(args.cases)])
    run([python, "scripts/build_manifest.py", "--root", cfg["data"]["root"],
         "--out", cfg["data"]["split_manifest"], "--ratios", "0.6", "0.2", "0.2"])
    run([python, "-m", "src.train", "--config", args.config, "--device", "cpu",
         "--run-id", "demo", "--max-epochs", str(args.epochs)])

    checkpoint = os.path.join(cfg["outputs"]["dir"], "demo", "best.pt")
    run([python, "-m", "src.evaluate", "--checkpoint", checkpoint, "--split", "test", "--device", "cpu"])

    test_rows = [r for r in read_manifest(os.path.join(ROOT, cfg["data"]["split_manifest"])) if r["split"] == "test"]
    if test_rows:
        run([python, "-m", "src.infer", "--checkpoint", checkpoint, "--input", test_rows[0]["image"],
             "--out-dir", "outputs/demo/inference", "--device", "cpu"])

    summary_path = os.path.join(ROOT, cfg["outputs"]["dir"], "demo", "eval_test", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        dice = summary["metrics"].get("dice", {})
        print("\n=== demo test-set Dice ===")
        print(json.dumps({"n_cases": summary["n_cases"], "mean": dice.get("mean"),
                          "median": dice.get("median"),
                          "ci95": [dice.get("ci95_low"), dice.get("ci95_high")]}, indent=2))
    print("\nArtifacts under outputs/demo/. Serve the API with:")
    print(f"  set LIVER3D_CHECKPOINT={checkpoint}")
    print("  uvicorn src.api.service:app --port 8000")


if __name__ == "__main__":
    main()
