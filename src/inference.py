"""Sliding-window inference for whole volumes.

A full CT volume rarely fits in GPU memory, so the volume is covered with
overlapping windows whose probabilities are blended with a Gaussian (or
constant) weight map. Window size, overlap, blending and batch size must match
the configuration used during validation.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _starts(size: int, roi: int, step: int) -> List[int]:
    if size <= roi:
        return [0]
    positions = list(range(0, size - roi + 1, step))
    if positions[-1] != size - roi:
        positions.append(size - roi)
    return positions


def gaussian_weight(roi: Sequence[int], sigma_scale: float = 0.125, device=None) -> torch.Tensor:
    grids = []
    for size in roi:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        sigma = max(sigma_scale * size, 1e-3)
        grids.append(torch.exp(-(coords**2) / (2 * sigma**2)))
    weight = grids[0][:, None, None] * grids[1][None, :, None] * grids[2][None, None, :]
    weight = weight / weight.max()
    weight = weight.clamp_min(1e-4)
    return weight.to(device) if device is not None else weight


@torch.no_grad()
def sliding_window_inference(
    image: torch.Tensor,
    model: torch.nn.Module,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 2,
    blend: str = "gaussian",
    device=None,
    amp: bool = False,
    progress: bool = False,
) -> torch.Tensor:
    """Return sigmoid probabilities with the same spatial shape as `image`.

    image: tensor of shape (C, D, H, W) or (1, C, D, H, W).
    """
    squeeze_batch = image.dim() == 4
    if squeeze_batch:
        image = image.unsqueeze(0)
    if image.dim() != 5:
        raise ValueError(f"expected a 4D or 5D tensor, got shape {tuple(image.shape)}")

    device = device or next(model.parameters()).device
    model.eval()

    spatial = list(image.shape[2:])
    roi = [min(int(r), int(s)) for r, s in zip(roi_size, spatial)]
    # keep windows divisible by the network's pooling factor
    divisor = getattr(model, "size_divisor", 1)
    roi = [max(divisor, int(np.ceil(r / divisor) * divisor)) for r in roi]
    pads = []
    for size, window in zip(spatial, roi):
        deficit = max(0, window - size)
        pads.append((deficit // 2, deficit - deficit // 2))
    if any(p != (0, 0) for p in pads):
        pad_arg = []
        for before, after in reversed(pads):  # F.pad wants reversed spatial order
            pad_arg.extend([before, after])
        image = F.pad(image, pad_arg, mode="constant", value=0.0)
        spatial = list(image.shape[2:])

    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    steps = [max(1, int(round(r * (1.0 - overlap)))) for r in roi]
    positions = [
        (z, y, x)
        for z in _starts(spatial[0], roi[0], steps[0])
        for y in _starts(spatial[1], roi[1], steps[1])
        for x in _starts(spatial[2], roi[2], steps[2])
    ]

    weight = (
        gaussian_weight(roi, device=device)
        if blend == "gaussian"
        else torch.ones(roi, device=device)
    )

    out = torch.zeros((1, 1, *spatial), dtype=torch.float32, device=device)
    norm = torch.zeros((1, 1, *spatial), dtype=torch.float32, device=device)

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    for index in range(0, len(positions), sw_batch_size):
        batch_positions = positions[index : index + sw_batch_size]
        patches = torch.cat(
            [
                image[:, :, z : z + roi[0], y : y + roi[1], x : x + roi[2]]
                for (z, y, x) in batch_positions
            ],
            dim=0,
        ).to(device)
        with torch.autocast(device_type=autocast_device, enabled=bool(amp and autocast_device == "cuda")):
            logits = model(patches)
        probs = torch.sigmoid(logits.float())
        for slot, (z, y, x) in enumerate(batch_positions):
            out[:, :, z : z + roi[0], y : y + roi[1], x : x + roi[2]] += probs[slot : slot + 1] * weight
            norm[:, :, z : z + roi[0], y : y + roi[1], x : x + roi[2]] += weight
        if progress:
            print(f"sliding window {min(index + sw_batch_size, len(positions))}/{len(positions)}")

    out = out / norm.clamp_min(1e-6)

    if any(p != (0, 0) for p in pads):
        z0, y0, x0 = [p[0] for p in pads]
        z1 = out.shape[2] - pads[0][1]
        y1 = out.shape[3] - pads[1][1]
        x1 = out.shape[4] - pads[2][1]
        out = out[:, :, z0:z1, y0:y1, x0:x1]

    return out[0] if squeeze_batch else out


def predict_volume(image_np: np.ndarray, model: torch.nn.Module, cfg, device=None) -> np.ndarray:
    """Convenience wrapper: preprocessed numpy volume -> probability volume."""
    inference_cfg = cfg["inference"]
    tensor = torch.from_numpy(np.ascontiguousarray(image_np)).float().unsqueeze(0)
    probs = sliding_window_inference(
        tensor,
        model,
        roi_size=inference_cfg.get("roi_size", cfg["data"]["patch_size"]),
        overlap=float(inference_cfg.get("overlap", 0.5)),
        sw_batch_size=int(inference_cfg.get("sw_batch_size", 2)),
        blend=str(inference_cfg.get("blend", "gaussian")),
        device=device,
        amp=bool(cfg["training"].get("amp", False)),
    )
    return probs.squeeze(0).cpu().numpy()
