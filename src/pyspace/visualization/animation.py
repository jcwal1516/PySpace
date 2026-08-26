"""Spatial slice animations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist


def animate_spatial_slices(
    volume: np.ndarray,
    *,
    axis: int = 2,
    interval: int = 200,
    cmap: str = "viridis",
) -> FuncAnimation:
    """Return a ``FuncAnimation`` traversing slices in ascending order."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("volume must be three-dimensional")
    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2")
    if interval <= 0:
        raise ValueError("interval must be positive")
    figure, axes = plt.subplots()
    first = np.take(array, 0, axis=axis)
    image = axes.imshow(first, cmap=cmap)
    title = axes.set_title(f"Slice 1/{array.shape[axis]}")

    def update(frame: int) -> tuple[Artist, Artist]:
        image.set_data(np.take(array, frame, axis=axis))
        title.set_text(f"Slice {frame + 1}/{array.shape[axis]}")
        return image, title

    return FuncAnimation(figure, update, frames=range(array.shape[axis]), interval=interval, blit=False)


__all__ = ["animate_spatial_slices"]
