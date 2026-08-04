"""SPACE-compatible back-mapping of learned patterns into object images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _radius_key(mapping: Mapping[str | float, Any], radius: float) -> Any:
    matches = [value for key, value in mapping.items() if float(key) == radius]
    if len(matches) != 1:
        raise ValueError(f"Exactly one value must match radius {radius}")
    return matches[0]


def _regions(bounds: Sequence[Sequence[float]]) -> tuple[list[tuple[float, float]], list[int]]:
    original = [(float(pair[0]), float(pair[1])) for pair in bounds]
    if not original or any(len(pair) != 2 for pair in bounds):
        raise ValueError("region_bounds must contain two-entry bounds")
    if any(not (0 <= lower < upper <= 100) for lower, upper in original):
        raise ValueError("region bounds must be increasing values within [0, 100]")
    ordered = sorted(enumerate(original, start=1), key=lambda item: item[1][0])
    if any(left[1][1] > right[1][0] for left, right in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("region bounds cannot overlap")
    segments: list[tuple[float, float]] = []
    region_ids: list[int] = []
    cursor = 0.0
    catchall = len(original) + 1
    for region_id, (lower, upper) in ordered:
        if lower > cursor:
            segments.append((cursor, lower))
            region_ids.append(catchall)
        segments.append((lower, upper))
        region_ids.append(region_id)
        cursor = upper
    if cursor < 100:
        segments.append((cursor, 100.0))
        region_ids.append(catchall)
    return segments, region_ids


def _combined_images(images: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray]) -> np.ndarray:
    if isinstance(images, Mapping):
        arrays = list(images.values())
    elif isinstance(images, Sequence) and not isinstance(images, np.ndarray):
        arrays = list(images)
    else:
        arrays = [images]
    normalized = [np.asarray(array) for array in arrays]
    normalized = [array[..., None] if array.ndim == 3 else array for array in normalized]
    if any(array.ndim != 4 or array.shape[:3] != normalized[0].shape[:3] for array in normalized):
        raise ValueError("All images must share a three-dimensional spatial shape")
    return np.concatenate(normalized, axis=3)


def map_pattern(  # noqa: PLR0913, PLR0917
    covar_data: pd.DataFrame,
    region_bounds: Sequence[Sequence[float]],
    img: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
    census: pd.DataFrame,
    radius: float,
    radii: Mapping[str | float, Sequence[float]],
    col_pal: Sequence[str],
    orig_bkgd: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Vote census neighborhoods back into a SPACE-compatible object image."""
    required = {"X", "Y", "V"}
    if not required.issubset(covar_data):
        raise ValueError("covar_data requires X, Y, and V columns")
    variables = list(dict.fromkeys(covar_data["V"].astype(str)))
    missing = [variable for variable in variables if variable not in census]
    if missing:
        raise ValueError(f"census is missing covariation variables: {missing}")
    narrowed = census.loc[census["Radius"].astype(float) == float(radius), [*variables, "X", "Y", "Z"]]
    if narrowed.empty:
        raise ValueError(f"census does not contain radius {radius}")
    path = covar_data.pivot(index="X", columns="V", values="Y").sort_index()
    path = path.reindex(columns=variables)
    if path.isna().any().any():
        raise ValueError("covar_data must contain one value per X position and variable")
    distances = np.linalg.norm(
        narrowed[variables].to_numpy(dtype=float)[:, None, :] - path.to_numpy(dtype=float)[None, :, :],
        axis=2,
    )
    nearest = np.argmin(distances, axis=1)
    minimum_distance = distances[np.arange(len(narrowed)), nearest]
    maximum_distance = float(minimum_distance.max())
    confidence = np.ones(len(narrowed)) if maximum_distance == 0 else 1 - minimum_distance / maximum_distance - 0.0001
    positions = path.index.to_numpy(dtype=float)[nearest]
    segments, region_ids = _regions(region_bounds)
    lower_bounds = np.asarray([segment[0] for segment in segments])
    segment_indices = np.searchsorted(lower_bounds, positions, side="right") - 1
    neighborhood_regions = np.asarray(region_ids, dtype=int)[segment_indices]

    image = _combined_images(img)
    pixel_radius = np.asarray(_radius_key(radii, float(radius)), dtype=int)
    if pixel_radius.shape != (3,) or np.any(pixel_radius <= 0):
        raise ValueError("radii entries must contain positive X/Y/Z pixel radii")
    max_region = max(region_ids)
    votes = np.zeros((*image.shape[:3], max_region + 1), dtype=float)
    for row_offset, (_, row) in enumerate(narrowed.iterrows()):
        center = row[["X", "Y", "Z"]].to_numpy(dtype=int) - 1
        starts = np.maximum(center - pixel_radius, 0)
        stops = np.minimum(center + pixel_radius, np.asarray(image.shape[:3]) - 1)
        axes = [np.arange(start, stop + 1) for start, stop in zip(starts, stops, strict=True)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        included = np.sqrt(np.sum(((grid - center) / pixel_radius) ** 2, axis=-1)) <= 1
        target = tuple(slice(int(start), int(stop) + 1) for start, stop in zip(starts, stops, strict=True))
        region_votes = votes[..., int(neighborhood_regions[row_offset]) - 1]
        region_votes[target] += included * confidence[row_offset]
    votes[..., -1] = 0.000001
    object_image = np.argmax(votes, axis=3) + 1
    object_image[object_image == max_region + 1] = 0
    if orig_bkgd:
        object_image *= np.sum(image, axis=3) > 0
    object_image = object_image[..., None]

    palette = list(col_pal)
    if len(palette) != len(region_bounds):
        raise ValueError("col_pal must contain one color per requested region")
    catchall = len(region_bounds) + 1
    if np.any(object_image == catchall):
        palette.append("#808080")
    unique_palette: list[str] = []
    remap: dict[int, int] = {}
    for old_id, color in enumerate(palette, start=1):
        if color not in unique_palette:
            unique_palette.append(color)
        remap[old_id] = unique_palette.index(color) + 1
    original_labels = object_image.copy()
    for old_id, new_id in remap.items():
        object_image[original_labels == old_id] = new_id
    return object_image, unique_palette


__all__ = ["map_pattern"]
