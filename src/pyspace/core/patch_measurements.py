"""SPACE-compatible image normalization, seed eligibility, and patch measurement."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io.image_loader import read_image_array
from .operations import patch_3D

_IMAGE_NAME = re.compile(r"^(?P<kind>[OS])(?P<group>[1-9][0-9]*)$")


def image_group(name: str) -> tuple[str, int]:
    """Parse a SPACE image name such as ``O1`` or ``S3``."""
    match = _IMAGE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Image name {name!r} must match O<number> or S<number>")
    return match.group("kind"), int(match.group("group"))


def _to_4d(name: str, value: str | Path | np.ndarray) -> np.ndarray:
    kind, _ = image_group(name)
    array = np.asarray(read_image_array(value) if isinstance(value, (str, Path)) else value)
    if array.size == 0 or array.ndim < 2 or array.ndim > 4:
        raise ValueError(f"{name} must be a non-empty 2D, 3D, or 4D array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if kind == "O":
        if array.ndim == 2:
            return array[:, :, None, None]
        if array.ndim == 3:
            return array[:, :, :, None]
        if array.shape[-1] != 1:
            raise ValueError(f"Object image {name} must contain exactly one channel")
        return array
    if array.ndim == 2:
        return array[:, :, None, None]
    if array.ndim == 3:
        return array[:, :, None, :]
    return array


def normalize_images(images: Mapping[str, str | Path | np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load images into SPACE's X/Y/Z/channel layout and validate geometry."""
    if not isinstance(images, Mapping) or not images:
        raise ValueError("images must be a non-empty named mapping")
    arrays = {str(name): _to_4d(str(name), value) for name, value in images.items()}
    shapes = {array.shape[:3] for array in arrays.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={array.shape[:3]}" for name, array in arrays.items())
        raise ValueError(f"All images must have identical X/Y/Z dimensions; got {details}")
    types = {name: "object" if image_group(name)[0] == "O" else "scalar" for name in arrays}
    return arrays, types


def isodata_threshold(image: np.ndarray) -> int:
    """Return the pinned SPACE/ImageJ-style threshold for an 8-bit channel."""
    values = np.asarray(image)
    if np.any(values < 0) or np.any(values > 255) or np.any(values != np.floor(values)):
        raise ValueError("Scalar image values must be integer intensities in [0, 255]")
    histogram = np.bincount(values.astype(np.uint8).ravel(), minlength=256)
    occupied = np.flatnonzero(histogram)
    if len(occupied) == 1:
        return int(occupied[0])
    if len(occupied) == 2:
        return int(occupied[0] + 1)
    nonzero_histogram = histogram[1:]
    threshold = int(np.flatnonzero(nonzero_histogram)[0] + 2)
    while threshold < 255:
        low_counts = nonzero_histogram[: threshold - 1]
        high_counts = nonzero_histogram[threshold:]
        low_total = int(low_counts.sum())
        high_total = int(high_counts.sum())
        low_average = float(np.dot(low_counts, np.arange(1, threshold)) / low_total)
        if high_total == 0:
            return threshold - 1
        high_average = float(np.dot(high_counts, np.arange(threshold + 1, 256)) / high_total)
        if round((low_average + high_average) / 2) == threshold:
            return threshold
        threshold += 1
    return threshold


def scalar_thresholds(images: Mapping[str, np.ndarray], image_types: Mapping[str, str]) -> dict[str, list[int] | None]:
    """Calculate one threshold per independent or linked scalar channel."""
    return {
        name: None
        if image_types[name] == "object"
        else [isodata_threshold(array[:, :, :, channel]) for channel in range(array.shape[3])]
        for name, array in images.items()
    }


def _linked_scalar_names(os_pairs: Mapping[str, Any] | None, images: Mapping[str, np.ndarray]) -> set[str]:
    linked: set[str] = set()
    if not os_pairs:
        return linked
    for object_name, value in os_pairs.items():
        _, group = image_group(str(object_name))
        expected = f"S{group}"
        if expected in images:
            linked.add(expected)
        if isinstance(value, pd.DataFrame):
            linked.update(str(column).split(".")[0] for column in value.columns)
    return linked


def eligible_seed_coordinates(
    loaded_images: Mapping[str, np.ndarray],
    image_types: Mapping[str, str],
    bin_thresholds: Mapping[str, Sequence[int] | None],
    os_pairs: Mapping[str, Any] | None,
    background_value: int | float,
    allowed_seed_values: Mapping[str, Sequence[int] | None] | None = None,
) -> np.ndarray:
    """Return eligible R-array coordinates, excluding linked-only scalar pixels."""
    shape = next(iter(loaded_images.values())).shape[:3]
    eligible = np.zeros(shape, dtype=bool)
    linked = _linked_scalar_names(os_pairs, loaded_images)
    for name, image in loaded_images.items():
        allowed = None if allowed_seed_values is None else allowed_seed_values.get(name)
        if image_types[name] == "object":
            object_map = image[:, :, :, 0]
            layer = object_map != background_value
            if allowed is not None:
                allowed_array = np.asarray(allowed)
                layer &= np.isin(object_map, allowed_array) if allowed_array.size else False
            eligible |= layer
        elif name not in linked:
            thresholds = bin_thresholds[name]
            if thresholds is None:
                continue
            channels = range(image.shape[3]) if allowed is None else [int(value) - 1 for value in allowed]
            for channel in channels:
                if channel < 0 or channel >= image.shape[3]:
                    raise ValueError(f"Seed channel for {name} is outside 1..{image.shape[3]}")
                eligible |= image[:, :, :, channel] >= int(thresholds[channel])
    return np.argwhere(eligible)


def _component_rows(
    mask: np.ndarray,
    value_column: str,
    value: float | int,
    scalar_columns: list[str],
    scalar_data: np.ndarray | None,
) -> list[dict[str, float | int]]:
    components = patch_3D(np.asarray(mask, dtype=bool))["index"]
    rows: list[dict[str, float | int]] = []
    for component_id in range(1, int(components.max()) + 1):
        component = components == component_id
        row: dict[str, float | int] = {"Area": int(component.sum()), value_column: value}
        if scalar_data is not None:
            row.update(
                {
                    column: float(scalar_data[:, :, :, position][component].sum())
                    for position, column in enumerate(scalar_columns)
                }
            )
        rows.append(row)
    return rows


def _ellipsoid(
    shape: tuple[int, int, int], center: np.ndarray, radius: np.ndarray
) -> tuple[tuple[slice, ...], np.ndarray]:
    lower = np.maximum(np.floor(center - radius).astype(int), 0)
    upper = np.minimum(np.ceil(center + radius).astype(int) + 1, np.asarray(shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))
    axes = [np.arange(start, stop) for start, stop in zip(lower, upper, strict=True)]
    grids = np.meshgrid(*axes, indexing="ij")
    distance = sum(
        ((grid - coordinate) / axis_radius) ** 2
        for grid, coordinate, axis_radius in zip(grids, center, radius, strict=True)
    )
    return slices, np.asarray(distance <= 1)


def normalize_os_pairs(
    os_pairs: Mapping[str, Any] | None,
    images: Mapping[str, np.ndarray],
) -> dict[str, pd.DataFrame] | None:
    """Attach semantic row/column labels to NumPy link matrices."""
    if os_pairs is None:
        return None
    normalized: dict[str, pd.DataFrame] = {}
    for object_name, value in os_pairs.items():
        object_key = str(object_name)
        _, group = image_group(object_key)
        scalar_name = f"S{group}"
        if object_key not in images or scalar_name not in images:
            raise ValueError(f"OS_pairs entry {object_key!r} requires images {object_key} and {scalar_name}")
        object_ids = sorted(int(item) for item in np.unique(images[object_key]) if item != 0)
        channel_names = [f"{scalar_name}.{index}" for index in range(1, images[scalar_name].shape[3] + 1)]
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
        else:
            matrix = np.asarray(value)
            if matrix.ndim != 2:
                raise ValueError(f"OS_pairs[{object_key!r}] must be a two-dimensional matrix")
            if matrix.shape != (len(object_ids), len(channel_names)):
                expected = (len(object_ids), len(channel_names))
                raise ValueError(f"OS_pairs[{object_key!r}] has shape {matrix.shape}; expected {expected}")
            frame = pd.DataFrame(
                matrix,
                index=pd.Index([f"{object_key}.{item}" for item in object_ids]),
                columns=pd.Index(channel_names),
            )
        normalized[object_key] = frame
    return normalized


def measure_image_neighborhood(
    images: Mapping[str, np.ndarray],
    center: np.ndarray,
    radius: np.ndarray,
    thresholds: Mapping[str, Sequence[int] | None],
    background_value: int | float,
) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """Measure all connected object/scalar patches in one ellipsoid."""
    shape = next(iter(images.values())).shape[:3]
    slices, included = _ellipsoid(shape, center, radius)
    groups = sorted({image_group(name)[1] for name in images})
    patches: dict[str, pd.DataFrame] = {}
    for group in groups:
        object_name = f"O{group}"
        scalar_name = f"S{group}"
        scalar = images[scalar_name][slices] if scalar_name in images else None
        if object_name in images:
            objects = images[object_name][slices][:, :, :, 0]
            scalar_columns = (
                [] if scalar is None else [f"{scalar_name}.{index}" for index in range(1, scalar.shape[3] + 1)]
            )
            rows: list[dict[str, float | int]] = []
            for object_id in sorted(int(item) for item in np.unique(objects[included]) if item != background_value):
                rows.extend(
                    _component_rows(
                        included & (objects == object_id),
                        object_name,
                        object_id,
                        scalar_columns,
                        scalar,
                    )
                )
            if not rows:
                rows = [{"Area": 0, object_name: 0, **dict.fromkeys(scalar_columns, 0.0)}]
            patches[object_name] = pd.DataFrame(rows)
        elif scalar is not None:
            scalar_threshold = thresholds[scalar_name]
            if scalar_threshold is None:
                raise RuntimeError(f"Missing thresholds for {scalar_name}")
            for channel in range(scalar.shape[3]):
                column = f"{scalar_name}.{channel + 1}"
                rows = []
                for positive in (True, False):
                    selected = scalar[:, :, :, channel] >= scalar_threshold[channel]
                    if not positive:
                        selected = ~selected
                    rows.extend(
                        _component_rows(
                            included & selected,
                            column,
                            0,
                            [column],
                            scalar[:, :, :, [channel]],
                        )
                    )
                patches[column] = pd.DataFrame(rows or [{"Area": 0.0, column: 0.0}])
    return patches, included


__all__ = [
    "eligible_seed_coordinates",
    "image_group",
    "isodata_threshold",
    "measure_image_neighborhood",
    "normalize_images",
    "normalize_os_pairs",
    "scalar_thresholds",
]
