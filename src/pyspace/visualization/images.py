"""R-compatible image encoding and focused spatial plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _slice_4d(image: np.ndarray, selection: Mapping[str, int]) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError("img must use the four-dimensional X,Y,Z,C layout")
    if len(selection) != 1:
        raise ValueError("slice must contain exactly one of X, Y, or Z")
    name, one_based_index = next(iter(selection.items()))
    if name not in {"X", "Y", "Z"}:
        raise ValueError("slice must be named X, Y, or Z")
    axis = {"X": 0, "Y": 1, "Z": 2}[name]
    index = int(one_based_index) - 1
    if index < 0 or index >= image.shape[axis]:
        raise IndexError(f"{name} slice {one_based_index} is outside the image")
    return np.take(image, index, axis=axis)


def _object_rgb(labels: np.ndarray, palette: Sequence[str], objects: Sequence[int] | None) -> np.ndarray:
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 2:
        raise ValueError("The selected object slice must be two-dimensional")
    selected = sorted({int(value) for value in (objects or np.unique(labels))})
    selected = [value for value in selected if value > 0]
    if not selected:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    if any(value > len(palette) for value in selected):
        raise ValueError("objects contains an ID outside col_pal")
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for object_id in selected:
        rgb[labels == object_id] = np.rint(255 * np.asarray(mcolors.to_rgb(palette[object_id - 1]))).astype(np.uint8)
    unselected = (labels > 0) & ~np.isin(labels, selected)
    rgb[unselected] = 128
    return rgb


def _scalar_rgb(
    values: np.ndarray,
    palette: Sequence[str],
    scalars: Sequence[int] | None,
    enhancement: float,
) -> np.ndarray:
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError("The selected scalar slice must have a final channel axis")
    selected = [int(value) for value in (scalars or range(1, values.shape[-1] + 1))]
    if any(value < 1 or value > values.shape[-1] or value > len(palette) for value in selected):
        raise ValueError("scalars contains a channel outside img or col_pal")
    if not 0 <= enhancement < 1:
        raise ValueError("enh_cnt must be in [0, 1)")
    rgb = np.zeros((*values.shape[:2], 3), dtype=float)
    for channel_id in selected:
        channel = values[..., channel_id - 1].astype(float)
        scale = float(np.quantile(channel, 1 - enhancement))
        if scale <= 0:
            continue
        color = 255 * np.asarray(mcolors.to_rgb(palette[channel_id - 1]))
        rgb += channel[..., None] / scale * color
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def plot_image(
    img: np.ndarray,
    img_type: str,
    col_pal: Sequence[str],
    slice: Mapping[str, int] | None = None,
    objects: Sequence[int] | None = None,
    scalars: Sequence[int] | None = None,
    enh_cnt: float = 0,
) -> tuple[np.ndarray, Figure]:
    """Encode and render one SPACE image slice without implicit file writes."""
    if img_type not in {"O", "S"}:
        raise ValueError("img_type must be 'O' or 'S'")
    selected = _slice_4d(np.asarray(img), slice or {"Z": 1})
    rgb = (
        _object_rgb(selected, col_pal, objects) if img_type == "O" else _scalar_rgb(selected, col_pal, scalars, enh_cnt)
    )
    figure, axes = plt.subplots()
    axes.imshow(rgb)
    axes.set_axis_off()
    figure.tight_layout(pad=0)
    return rgb, figure


def plot_spatial_map(coordinates: np.ndarray, values: np.ndarray, *, cmap: str = "viridis") -> Figure:
    """Plot numeric values at validated 2D coordinates."""
    points = np.asarray(coordinates, dtype=float)
    measurements = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or measurements.shape != (len(points),):
        raise ValueError("coordinates must be (n, 2) and values must be (n,)")
    figure, axes = plt.subplots()
    scatter = axes.scatter(points[:, 0], points[:, 1], c=measurements, cmap=cmap)
    axes.set(xlabel="X", ylabel="Y")
    figure.colorbar(scatter, ax=axes, label="Value")
    figure.tight_layout()
    return figure


__all__ = ["plot_image", "plot_spatial_map"]
