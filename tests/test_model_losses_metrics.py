"""Model, loss, metric and postprocessing unit tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference import sliding_window_inference
from src.losses import DiceBCELoss, build_loss, dice_loss
from src.metrics import aggregate, assd, case_metrics, dice, hd95, iou, precision, recall
from src.models.unet3d import UNet3D, build_model, count_parameters
from src.postprocess import fill_holes, largest_component, postprocess, remove_small_components


def test_model_forward_shape():
    model = UNet3D(channels=(4, 8, 16))
    x = torch.randn(2, 1, 32, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32, 32)
    assert model.size_divisor == 4
    assert count_parameters(model) > 0


def test_model_handles_non_divisible_input():
    model = UNet3D(channels=(4, 8))
    out = model(torch.randn(1, 1, 18, 20, 22))
    assert out.shape == (1, 1, 18, 20, 22)


def test_build_model_rejects_unknown_architecture(tiny_cfg):
    cfg = dict(tiny_cfg)
    cfg["model"] = {**tiny_cfg["model"], "architecture": "transformer"}
    with pytest.raises(ValueError):
        build_model(cfg)


def test_dice_loss_rewards_overlap():
    target = torch.zeros(1, 1, 8, 8, 8)
    target[..., 2:6, 2:6, 2:6] = 1.0
    good = torch.where(target > 0, torch.tensor(6.0), torch.tensor(-6.0))
    bad = -good
    assert float(dice_loss(good, target)) < 0.05
    assert float(dice_loss(bad, target)) > 0.9


def test_dice_bce_weights(tiny_cfg):
    loss = build_loss(tiny_cfg)
    assert isinstance(loss, DiceBCELoss)
    target = torch.zeros(1, 1, 8, 8, 8)
    target[..., :4, :4, :4] = 1.0
    logits = torch.zeros_like(target)
    assert float(loss(logits, target)) > 0


def test_overlap_metrics_known_values():
    ref = np.zeros((10, 10, 10), dtype=np.uint8)
    ref[2:6, 2:6, 2:6] = 1
    pred = np.zeros_like(ref)
    pred[2:6, 2:6, 2:4] = 1  # half of the reference
    assert dice(pred, ref) == pytest.approx(2 * 32 / (32 + 64))
    assert iou(pred, ref) == pytest.approx(32 / 64)
    assert precision(pred, ref) == pytest.approx(1.0)
    assert recall(pred, ref) == pytest.approx(0.5)
    assert dice(ref, ref) == 1.0


def test_empty_masks_are_handled():
    empty = np.zeros((5, 5, 5), dtype=np.uint8)
    assert dice(empty, empty) == 1.0
    assert np.isnan(hd95(empty, empty, (1, 1, 1)))


def test_surface_metrics_scale_with_spacing():
    ref = np.zeros((20, 20, 20), dtype=np.uint8)
    ref[5:15, 5:15, 5:15] = 1
    pred = np.zeros_like(ref)
    pred[5:14, 5:15, 5:15] = 1  # one voxel shorter on one face
    d_1mm = hd95(pred, ref, (1.0, 1.0, 1.0))
    d_2mm = hd95(pred, ref, (2.0, 1.0, 1.0))
    assert d_2mm > d_1mm
    assert assd(pred, ref, (1.0, 1.0, 1.0)) >= 0


def test_case_metrics_and_aggregate():
    ref = np.zeros((16, 16, 16), dtype=np.uint8)
    ref[4:12, 4:12, 4:12] = 1
    pred = ref.copy()
    metrics = case_metrics(pred, ref, (2.0, 2.0, 2.0))
    assert metrics["dice"] == 1.0
    assert metrics["pred_volume_ml"] == pytest.approx(8**3 * 8 / 1000.0)
    assert metrics["n_components"] == 1

    summary = aggregate([metrics, {**metrics, "dice": 0.8}])
    assert summary["dice"]["n"] == 2
    assert summary["dice"]["mean"] == pytest.approx(0.9)
    assert summary["dice"]["ci95_low"] <= summary["dice"]["mean"] <= summary["dice"]["ci95_high"]


def test_postprocess_rules(tiny_cfg):
    prob = np.zeros((20, 20, 20), dtype=np.float32)
    prob[3:13, 3:13, 3:13] = 0.9        # liver-sized blob
    prob[17:19, 17:19, 17:19] = 0.8     # spurious speck
    prob[6:8, 6:8, 6:8] = 0.1           # hole inside the blob

    mask, report = postprocess(prob, tiny_cfg, (2.0, 2.0, 2.0))
    assert report["voxels_final"] > 0
    assert mask[17, 17, 17] == 0        # speck removed by largest-component rule
    assert mask[7, 7, 7] == 1           # hole filled
    assert report["empty_prediction"] is False

    cleaned, dropped = remove_small_components(
        (prob > 0.5).astype(np.uint8), (2.0, 2.0, 2.0), min_ml=1.0
    )
    assert dropped >= 1
    assert largest_component(np.zeros((4, 4, 4), np.uint8)).sum() == 0
    assert fill_holes(np.zeros((4, 4, 4), np.uint8)).sum() == 0


def test_empty_prediction_is_flagged(tiny_cfg):
    mask, report = postprocess(np.zeros((10, 10, 10), np.float32), tiny_cfg, (1.0, 1.0, 1.0))
    assert report["empty_prediction"] is True
    assert mask.sum() == 0


class _ConstantModel(torch.nn.Module):
    """Returns a fixed logit everywhere; used to check window blending."""

    size_divisor = 1

    def __init__(self, value: float = 2.0):
        super().__init__()
        self.value = value
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.full_like(x, self.value) + 0 * self.dummy


def test_sliding_window_blending_is_normalised():
    model = _ConstantModel(2.0)
    image = torch.zeros(1, 24, 26, 20)
    probs = sliding_window_inference(image, model, roi_size=(16, 16, 16), overlap=0.5, sw_batch_size=2)
    expected = float(torch.sigmoid(torch.tensor(2.0)))
    assert probs.shape == (1, 24, 26, 20)
    assert torch.allclose(probs, torch.full_like(probs, expected), atol=1e-5)


def test_sliding_window_pads_small_volumes():
    model = _ConstantModel(0.0)
    probs = sliding_window_inference(torch.zeros(1, 8, 9, 10), model, roi_size=(16, 16, 16))
    assert probs.shape == (1, 8, 9, 10)


def test_sliding_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        sliding_window_inference(torch.zeros(1, 16, 16, 16), _ConstantModel(), overlap=1.0)
