"""Segmentation losses.

Dice pushes overlap directly, which matters because liver voxels are a small
fraction of an abdominal volume. BCE adds voxel-wise calibration. The smoothing
constant and reduction are explicit because they change the reported number.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOOTH = 1.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = SMOOTH) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.dim()))
    intersection = (probs * target).sum(dims)
    denominator = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """dice_weight * DiceLoss + bce_weight * BCEWithLogits (mean reduction)."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = SMOOTH):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.dice_weight:
            loss = loss + self.dice_weight * dice_loss(logits, target, self.smooth)
        if self.bce_weight:
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(logits, target)
        return loss


def build_loss(cfg) -> nn.Module:
    name = str(cfg["training"].get("loss", "dice_bce")).lower()
    dice_w = float(cfg["training"].get("dice_weight", 1.0))
    bce_w = float(cfg["training"].get("bce_weight", 1.0))
    if name == "dice":
        return DiceBCELoss(dice_weight=dice_w, bce_weight=0.0)
    if name == "bce":
        return DiceBCELoss(dice_weight=0.0, bce_weight=bce_w)
    if name == "dice_bce":
        return DiceBCELoss(dice_weight=dice_w, bce_weight=bce_w)
    raise ValueError(f"unknown loss: {name}")
