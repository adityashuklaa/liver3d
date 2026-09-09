"""Training entry point.

    python -m src.train --config configs/baseline.yaml

Writes everything needed to reproduce and audit the run into
outputs/<run_id>/: config.yaml, env.json, split copy, train_log.jsonl,
metrics.csv, curves.png, last.pt and best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import Config, load_config, save_config
from .data.dataset import PatchDataset, VolumeDataset, build_cache
from .data.split import read_manifest, validate_manifest
from .inference import sliding_window_inference
from .losses import build_loss
from .metrics import dice as dice_metric
from .models.unet3d import build_model, count_parameters
from .utils.checksum import file_sha256, short
from .utils.env import write_environment
from .utils.logging_utils import JsonlLogger, get_logger
from .utils.seed import set_seed

LOG = get_logger("liver3d.train")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(model, dataset: VolumeDataset, cfg: Config, device) -> Dict[str, Any]:
    """Whole-volume sliding-window validation. Returns mean Dice and per case."""
    model.eval()
    inference_cfg = cfg["inference"]
    threshold = float(inference_cfg.get("threshold", 0.5))
    per_case: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if item["label"] is None:
            continue
        probs = sliding_window_inference(
            item["image"],
            model,
            roi_size=inference_cfg.get("roi_size", cfg["data"]["patch_size"]),
            overlap=float(inference_cfg.get("overlap", 0.5)),
            sw_batch_size=int(inference_cfg.get("sw_batch_size", 2)),
            blend=str(inference_cfg.get("blend", "gaussian")),
            device=device,
            amp=bool(cfg["training"].get("amp", False)),
        )
        pred = (probs.squeeze(0).cpu().numpy() >= threshold)
        ref = item["label"].squeeze(0).numpy() > 0.5
        per_case.append({"case_id": item["case_id"], "dice": dice_metric(pred, ref)})
    mean_dice = float(np.mean([c["dice"] for c in per_case])) if per_case else float("nan")
    return {"dice": mean_dice, "cases": per_case}


def train(cfg: Config, args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device = resolve_device(args.device)

    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(cfg["outputs"].get("dir", "outputs"), run_id)
    os.makedirs(run_dir, exist_ok=True)
    jsonl = JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))

    manifest_path = args.manifest or cfg["data"]["split_manifest"]
    rows = read_manifest(manifest_path)
    counts = validate_manifest(rows)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.limit_cases:
        train_rows = train_rows[: args.limit_cases]
        val_rows = val_rows[: max(1, args.limit_cases // 2)]
    if not val_rows:
        LOG.warning("no validation cases in manifest; checkpoint selection falls back to loss")

    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    write_environment(os.path.join(run_dir, "env.json"))
    shutil.copyfile(manifest_path, os.path.join(run_dir, "split_manifest.csv"))
    manifest_sha = file_sha256(manifest_path)

    LOG.info("run %s | device %s | split %s", run_id, device, counts)
    build_cache(train_rows + val_rows, cfg)

    train_ds = PatchDataset(train_rows, cfg, train=True, seed=int(cfg["seed"]))
    val_ds = VolumeDataset(val_rows, cfg) if val_rows else None
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(3, int(cfg["training"]["early_stopping_patience"]) // 3)
    )
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    LOG.info("model parameters: %s", f"{count_parameters(model):,}")
    max_epochs = int(args.max_epochs or cfg["training"]["max_epochs"])
    iters = int(args.iters_per_epoch or cfg["training"]["iters_per_epoch"])
    patience = int(cfg["training"].get("early_stopping_patience", 30))
    val_interval = int(cfg["training"].get("val_interval", 1))

    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    stream = _cycle(loader)
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for _ in range(iters):
            batch = next(stream)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        epoch_loss /= max(1, iters)

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_start, 2),
        }

        if val_ds is not None and epoch % val_interval == 0:
            val = validate(model, val_ds, cfg, device)
            record["val_dice"] = val["dice"]
            record["val_cases"] = val["cases"]
            score = val["dice"]
            scheduler.step(score)
        else:
            score = -epoch_loss

        history.append(record)
        jsonl.log(**record)
        LOG.info(
            "epoch %d/%d loss %.4f val_dice %s (%.1fs)",
            epoch,
            max_epochs,
            epoch_loss,
            f"{record.get('val_dice'):.4f}" if record.get("val_dice") is not None else "n/a",
            record["seconds"],
        )

        save_checkpoint(
            os.path.join(run_dir, "last.pt"),
            model,
            cfg,
            meta={"run_id": run_id, "epoch": epoch, "split_manifest_sha256": manifest_sha},
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(
                os.path.join(run_dir, "best.pt"),
                model,
                cfg,
                meta={
                    "run_id": run_id,
                    "epoch": epoch,
                    "selection_metric": cfg["training"].get("checkpoint_metric", "dice"),
                    "selection_value": score,
                    "split_manifest_sha256": manifest_sha,
                    "trained_on": [r["case_id"] for r in train_rows],
                },
            )
        elif epoch - best_epoch >= patience:
            LOG.info("early stopping at epoch %d (best epoch %d, score %.4f)", epoch, best_epoch, best_score)
            break

    write_metrics_csv(history, os.path.join(run_dir, "metrics.csv"))
    plot_curves(history, os.path.join(run_dir, "curves.png"))

    best_path = os.path.join(run_dir, "best.pt")
    summary = {
        "run_id": run_id,
        "run_dir": os.path.abspath(run_dir),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_run": len(history),
        "minutes": round((time.time() - started) / 60.0, 2),
        "best_checkpoint": os.path.abspath(best_path),
        "best_checkpoint_sha256": file_sha256(best_path) if os.path.exists(best_path) else None,
        "model_version": "liver-unet-" + short(file_sha256(best_path)) if os.path.exists(best_path) else None,
        "split_counts": counts,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    LOG.info("done: %s", json.dumps(summary, indent=2))
    return summary


def write_metrics_csv(history: List[Dict[str, Any]], path: str) -> str:
    fields = ["epoch", "train_loss", "val_dice", "lr", "seconds"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow({k: record.get(k, "") for k in fields})
    return path


def plot_curves(history: List[Dict[str, Any]], path: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting optional
        return None
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    val = [(h["epoch"], h["val_dice"]) for h in history if h.get("val_dice") is not None]
    if val:
        axes[1].plot([v[0] for v in val], [v[1] for v in val], color="tab:green", label="val Dice")
        best = max(val, key=lambda v: v[1])
        axes[1].scatter([best[0]], [best[1]], color="crimson", zorder=5, label=f"selected epoch {best[0]}")
        axes[1].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3D U-Net liver segmentation model")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--manifest", default=None, help="override data.split_manifest")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--limit-cases", type=int, default=None, help="smoke-test on N cases")
    return parser.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    args = parse_args(argv)
    cfg = load_config(args.config)
    return train(cfg, args)


if __name__ == "__main__":  # pragma: no cover
    main()
