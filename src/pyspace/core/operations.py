"""Small mathematical primitives ported from SPACE ``utils.R``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage


def _three_values(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain X, Y, and Z values")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def calc_vol(L: Sequence[float] | np.ndarray, dims: Sequence[int] | np.ndarray) -> int:
    """Return the discrete ellipsoid volume calculated by SPACE ``calc_vol``."""
    radii = _three_values(L, "L")
    dimensions = _three_values(dims, "dims")
    diameters = np.minimum(2 * radii + 1, dimensions)
    axes = [np.arange(-np.floor((diameter - 1) / 2), np.ceil((diameter - 1) / 2) + 1) for diameter in diameters]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    scaled = coordinates / radii
    return int(np.count_nonzero(np.sqrt(np.sum(scaled**2, axis=1)) <= 1))


def calc_vols(
    L: Sequence[Sequence[float] | np.ndarray],
    dims: Sequence[int] | np.ndarray,
) -> list[int]:
    """Apply :func:`calc_vol` to each radius vector, preserving order."""
    return [calc_vol(radius, dims) for radius in L]


def patch_3D(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Label 26-connected components and rank them by decreasing size.

    The returned ``index`` and ``size`` arrays correspond to the two named
    arrays returned by the pinned R implementation. Equal-sized components
    retain connected-component label order.
    """
    array = np.asarray(mask)
    if array.ndim != 3 or array.dtype != np.bool_:
        raise ValueError("mask must be a 3D boolean array")

    labels, component_count = ndimage.label(array, structure=np.ones((3, 3, 3), dtype=np.uint8))
    index_volume = np.zeros(array.shape, dtype=np.int64)
    size_volume = np.zeros(array.shape, dtype=np.int64)
    if component_count == 0:
        return {"index": index_volume, "size": size_volume}

    component_labels = np.arange(1, component_count + 1)
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)[1:]
    order = np.lexsort((component_labels, -sizes))
    for rank, offset in enumerate(order, start=1):
        component_label = int(component_labels[offset])
        component = labels == component_label
        index_volume[component] = rank
        size_volume[component] = int(sizes[offset])
    return {"index": index_volume, "size": size_volume}


__all__ = ["calc_vol", "calc_vols", "patch_3D"]
