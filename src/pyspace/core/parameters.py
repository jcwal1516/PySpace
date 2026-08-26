"""Parameter calculations ported from pinned SPACE ``suggest_parameters.R``."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from .operations import calc_vols


def _collapse_space_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return np.sum(array, axis=3)
    raise ValueError(f"SPACE images must have 2, 3, or 4 dimensions; got {array.ndim}")


def _combined_space_image(images: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray]) -> np.ndarray:
    if isinstance(images, Mapping):
        names = list(images)
        group_ids = [name[1:9] for name in names]
        retained = [
            np.asarray(images[name])
            for name, group_id in zip(names, group_ids, strict=True)
            if name.startswith("O") or group_ids.count(group_id) == 1
        ]
    elif isinstance(images, Sequence) and not isinstance(images, np.ndarray):
        retained = [np.asarray(image) for image in images]
    else:
        return _collapse_space_image(np.asarray(images))
    if not retained:
        raise ValueError("images contains no independent scalar or object maps")
    collapsed = [_collapse_space_image(image) for image in retained]
    if any(image.shape != collapsed[0].shape for image in collapsed[1:]):
        raise ValueError("All image maps must have the same spatial shape")
    return np.sum(np.stack(collapsed), axis=0)


def suggest_number(
    coverage: float,
    radii: Mapping[float, Sequence[float]] | Sequence[Sequence[float]],
    images: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
) -> list[int]:
    """Match ``SPACE::suggest_number`` using foreground pixels and discrete volumes."""
    coverage_value = float(coverage)
    if not np.isfinite(coverage_value) or coverage_value <= 0:
        raise ValueError("coverage must be a finite positive number")
    if coverage_value > 5:
        warnings.warn(
            "Coverage exceeding 5x will yield invalid inference due to pseudo-replication",
            RuntimeWarning,
            stacklevel=2,
        )
    radius_vectors = list(radii.values()) if isinstance(radii, Mapping) else list(radii)
    if not radius_vectors:
        raise ValueError("radii cannot be empty")
    combined = _combined_space_image(images)
    foreground_pixels = int(np.count_nonzero(combined > 0))
    volumes = calc_vols(radius_vectors, combined.shape)
    return [int(math.ceil(coverage_value * foreground_pixels / volume)) for volume in volumes]


def suggest_radii(
    target: Sequence[float],
    pixel_resolution: Mapping[str, float] | Sequence[float],
) -> dict[float, list[int]]:
    """Match ``SPACE::suggest_radii`` for X/Y/Z micron-to-pixel conversion."""
    resolution = (
        [pixel_resolution[key] for key in ("x", "y", "z")]
        if isinstance(pixel_resolution, Mapping)
        else list(pixel_resolution)
    )
    resolution_array = np.asarray(resolution, dtype=float)
    if resolution_array.shape != (3,) or np.any(~np.isfinite(resolution_array)) or np.any(resolution_array <= 0):
        raise ValueError("pixel_resolution must contain positive X, Y, and Z values")
    result: dict[float, list[int]] = {}
    for requested in target:
        radius = float(requested)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("target radii must be finite and non-negative")
        result[radius] = np.maximum(1, np.rint(radius / resolution_array).astype(int)).tolist()
    return result


__all__ = ["suggest_number", "suggest_radii"]
