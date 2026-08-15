"""Deterministic neighborhood sampling shared by census front ends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..io.image_loader import read_image_array
from ..io.table_loader import read_table
from .census_models import Neighborhood


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    return random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)


def normalize_radii(radii: float | list[float] | tuple[float, ...]) -> list[float]:
    """Validate and normalize isotropic radii."""
    values = [float(radii)] if isinstance(radii, (int, float)) else [float(value) for value in radii]
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("radii must contain finite positive numbers")
    return values


def normalize_sample_sizes(
    sample_size: int | list[int] | tuple[int, ...] | None,
    radius_count: int,
) -> list[int | None]:
    """Return one validated sample size per radius."""
    if sample_size is None:
        return [None] * radius_count
    values: list[int | None] = (
        [int(sample_size)] * radius_count if isinstance(sample_size, int) else [int(v) for v in sample_size]
    )
    if len(values) != radius_count:
        raise ValueError("The number of sample sizes does not match the number of radii.")
    if any(value is None or value <= 0 for value in values):
        raise ValueError("sample_size must contain positive integers")
    return values


def _image_points(array: np.ndarray, background_value: int | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(array)
    if image.ndim == 2:
        indices = np.argwhere(image != background_value)
        points = indices[:, [1, 0]].astype(float)
        return points, image[tuple(indices.T)], image[tuple(indices.T)][:, None]
    if image.ndim == 3 and image.shape[-1] <= 4:
        mask = np.any(image != background_value, axis=-1)
        indices = np.argwhere(mask)
        points = indices[:, [1, 0]].astype(float)
        intensities = image[indices[:, 0], indices[:, 1]]
        return points, np.arange(len(points)), np.asarray(intensities)
    if image.ndim == 3:
        indices = np.argwhere(image != background_value)
        points = indices[:, [2, 1, 0]].astype(float)
        values = image[tuple(indices.T)]
        return points, values, values[:, None]
    raise ValueError("Neighborhood images must be 2D, RGB(A), or 3D scalar arrays")


def _table_points(
    frame: pd.DataFrame,
    object_id_column: str | None,
    pixel_to_micron_factor: float | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    coordinate_columns = [name for name in ("x", "y", "z") if name in frame.columns]
    if len(coordinate_columns) < 2:
        coordinate_columns = [name for name in ("X", "Y", "Z") if name in frame.columns]
    if len(coordinate_columns) < 2:
        raise ValueError("Coordinate table must contain x/y or X/Y columns")
    coordinates = frame[coordinate_columns].to_numpy(dtype=float, copy=True)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Coordinate columns must contain only finite values")
    if pixel_to_micron_factor is not None:
        if pixel_to_micron_factor <= 0:
            raise ValueError("pixel_to_micron_factor must be positive")
        coordinates *= float(pixel_to_micron_factor)
    object_column = object_id_column
    if object_column is None:
        object_column = next((name for name in ("object_id", "Object") if name in frame.columns), None)
    object_ids = frame[object_column].to_numpy(copy=True) if object_column else None
    excluded = {*coordinate_columns}
    if object_column:
        excluded.add(object_column)
    value_columns = [
        name for name in frame.columns if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    ]
    intensities = frame[value_columns].to_numpy(dtype=float, copy=True) if value_columns else None
    return coordinates, object_ids, intensities


def _extract(
    center: np.ndarray,
    radius: float,
    points: np.ndarray,
    object_ids: np.ndarray | None,
    intensities: np.ndarray | None,
    tree: cKDTree,
) -> Neighborhood:
    indices = np.asarray(tree.query_ball_point(center, radius), dtype=int)
    if len(indices):
        distances = np.linalg.norm(points[indices] - center, axis=1)
        indices = indices[np.lexsort((indices, distances))]
    return Neighborhood(
        center=tuple(float(value) for value in center),
        radius=radius,
        points=points[indices],
        object_ids=object_ids[indices] if object_ids is not None else None,
        intensities=intensities[indices] if intensities is not None else None,
        patch_id=f"patch_{'_'.join(format(float(value), 'g') for value in center)}_{radius:g}",
        is_3d=points.shape[1] == 3,
    )


def create_neighborhoods(
    data: np.ndarray | pd.DataFrame | str | Path,
    radii: float | list[float],
    seed_points: np.ndarray | None = None,
    background_value: int | float = 0,
    max_neighborhoods: int | None = None,
    object_id_column: str | None = None,
    pixel_to_micron_factor: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> dict[float, list[Neighborhood]]:
    """Create deterministic Euclidean neighborhoods from images or coordinate tables."""
    radius_values = normalize_radii(radii)
    source: np.ndarray | pd.DataFrame
    if isinstance(data, (str, Path)):
        suffix = Path(data).suffix.lower()
        source = read_image_array(data) if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} else read_table(data)
    else:
        source = data
    if isinstance(source, pd.DataFrame):
        points, object_ids, intensities = _table_points(source, object_id_column, pixel_to_micron_factor)
    elif isinstance(source, np.ndarray):
        points, object_ids, intensities = _image_points(source, background_value)
    else:
        raise TypeError(f"Unsupported census source: {type(source).__name__}")
    if len(points) == 0:
        return {radius: [] for radius in radius_values}
    tree = cKDTree(points)
    if seed_points is not None:
        selected = np.asarray(seed_points, dtype=float)
        if selected.ndim != 2 or selected.shape[1] != points.shape[1]:
            raise ValueError(f"seed_points must have shape (n, {points.shape[1]})")
        if max_neighborhoods is not None:
            selected = selected[:max_neighborhoods]
    else:
        count = len(points) if max_neighborhoods is None else min(int(max_neighborhoods), len(points))
        if count < 0:
            raise ValueError("max_neighborhoods cannot be negative")
        selected = points[_generator(random_state).permutation(len(points))[:count]]
    return {
        radius: [_extract(center, radius, points, object_ids, intensities, tree) for center in selected]
        for radius in radius_values
    }


__all__ = ["create_neighborhoods", "normalize_radii", "normalize_sample_sizes"]
