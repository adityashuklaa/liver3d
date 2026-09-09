"""Checkpoint bundling: weights + config + provenance in one file.

A release binds one checkpoint to one preprocessing configuration. Swapping
weights without the matching config invalidates the reported performance, so
both travel together and the loader refuses a mismatch it cannot resolve.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from .config import Config, load_config
from .models.unet3d import build_model
from .utils.checksum import file_sha256, short
from .utils.env import environment_record


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    cfg: Config,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
        "meta": meta or {},
        "environment": environment_record(),
        "format_version": 1,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    device: str = "cpu",
    cfg: Optional[Config] = None,
) -> Tuple[torch.nn.Module, Config, Dict[str, Any]]:
    """Load a checkpoint bundle. The stored config wins unless one is passed."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(f"{path} is not a liver3d checkpoint bundle")
    stored = payload.get("config") or {}
    resolved = cfg if cfg is not None else load_config(None, overrides=stored)
    model = build_model(resolved)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    meta = dict(payload.get("meta") or {})
    meta["environment"] = payload.get("environment")
    meta["checkpoint_sha256"] = file_sha256(path)
    meta["model_version"] = model_version(path, meta)
    return model, resolved, meta


def model_version(path: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """Stable release identifier derived from the checkpoint checksum."""
    if meta and meta.get("model_version"):
        return str(meta["model_version"])
    return "liver-unet-" + short(file_sha256(path))
