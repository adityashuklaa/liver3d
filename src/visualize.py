"""Review artifacts: slice overlays and an interactive 3D surface.

Outputs are for human review, never for automatic acceptance. Slices are taken
through the mask bounding box so a reviewer sees the region that matters.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


def _normalise(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = float(np.percentile(image, 1)), float(np.percentile(image, 99))
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max() or 1.0)
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def _slice_indices(mask: np.ndarray, count: int, axis: int = 2) -> Sequence[int]:
    size = mask.shape[axis]
    if mask.any():
        coords = np.nonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        lo, hi = int(coords.min()), int(coords.max())
    else:
        lo, hi = 0, size - 1
    return np.unique(np.linspace(lo, hi, num=min(count, max(1, hi - lo + 1))).astype(int))


def overlay_slices(
    image: np.ndarray,
    mask: np.ndarray,
    path: str,
    reference: Optional[np.ndarray] = None,
    count: int = 6,
    title: str = "",
) -> Optional[str]:
    """Axial slice grid: prediction in red, reference contour in green."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting optional
        return None

    image = _normalise(image)
    mask = np.asarray(mask).astype(bool)
    indices = _slice_indices(mask if mask.any() else np.ones_like(mask), count)
    cols = min(3, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for position, index in enumerate(indices):
        ax = axes[position // cols][position % cols]
        ax.imshow(np.rot90(image[:, :, index]), cmap="gray", vmin=0, vmax=1)
        overlay = np.rot90(mask[:, :, index])
        rgba = np.zeros((*overlay.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = overlay * 0.35
        ax.imshow(rgba)
        if reference is not None:
            ax.contour(np.rot90(np.asarray(reference).astype(bool)[:, :, index]),
                       levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(f"slice {index}", fontsize=9)
        ax.axis("off")
    for position in range(len(indices), rows * cols):
        axes[position // cols][position % cols].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def mesh_html(mask: np.ndarray, spacing_mm, path: str, title: str = "Liver surface") -> Optional[str]:
    """Marching-cubes surface of the mask as a standalone interactive HTML page."""
    try:
        from skimage import measure
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - optional dependency
        return None

    mask = np.asarray(mask).astype(np.float32)
    if mask.sum() < 10:
        return None
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=tuple(float(s) for s in spacing_mm))
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#b5443a",
                opacity=0.55,
                name="liver",
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def case_report(
    image: np.ndarray,
    mask: np.ndarray,
    out_dir: str,
    case_id: str,
    spacing_mm,
    reference: Optional[np.ndarray] = None,
    make_mesh: bool = True,
) -> dict:
    """Write the overlay grid and (optionally) the 3D surface for one case."""
    os.makedirs(out_dir, exist_ok=True)
    artifacts = {}
    png = overlay_slices(
        image, mask, os.path.join(out_dir, f"{case_id}_overlay.png"),
        reference=reference, title=f"{case_id} - prediction (red) vs reference (green)",
    )
    if png:
        artifacts["overlay_png"] = png
    if make_mesh:
        html = mesh_html(mask, spacing_mm, os.path.join(out_dir, f"{case_id}_surface.html"),
                         title=f"{case_id} liver surface")
        if html:
            artifacts["surface_html"] = html
    return artifacts
