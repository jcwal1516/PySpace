"""Image census implementation matching the pinned SPACE patch semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .census_models import CensusResult, Neighborhood, summarize_neighborhoods
from .patch_measurements import (
    eligible_seed_coordinates,
    measure_image_neighborhood,
    normalize_images,
    normalize_os_pairs,
    scalar_thresholds,
)
from .patch_summary import summarize_patches


def _radius_plan(
    radii: float | Sequence[float] | Mapping[str | float, Sequence[float]],
    depth: int,
) -> list[tuple[float, np.ndarray]]:
    if isinstance(radii, Mapping):
        result = [(float(name), np.asarray(axes, dtype=float)) for name, axes in radii.items()]
    else:
        values = [float(radii)] if isinstance(radii, (int, float)) else [float(value) for value in radii]
        result = [(value, np.asarray([value, value, value if depth > 1 else 1.0])) for value in values]
    if not result:
        raise ValueError("radii cannot be empty")
    for name, axes in result:
        if not np.isfinite(name) or name <= 0 or axes.shape != (3,) or np.any(~np.isfinite(axes)) or np.any(axes <= 0):
            raise ValueError("Each radius must have a positive name and finite positive X/Y/Z axes")
    return result


def _sample_sizes(sample_size: int | Sequence[int], count: int) -> list[int]:
    values = [int(sample_size)] * count if isinstance(sample_size, int) else [int(value) for value in sample_size]
    if len(values) != count:
        raise ValueError("The number of sample sizes does not match the number of radii.")
    if any(value <= 0 for value in values):
        raise ValueError("sample_size must contain positive integers")
    return values


def _apply_object_remap(
    images: dict[str, np.ndarray],
    object_remap: Mapping[str, Mapping[int, int]] | None,
) -> tuple[dict[str, np.ndarray], bool]:
    if not object_remap:
        return images, False
    unknown = sorted(set(object_remap) - set(images))
    if unknown:
        raise ValueError(f"object_remap contains unknown images: {unknown}")
    output = dict(images)
    changed = False
    for name, mapping in object_remap.items():
        if not name.startswith("O"):
            raise ValueError(f"object_remap can only target object images, got {name}")
        source = images[name]
        destination = source.copy()
        for old, new in mapping.items():
            selected = source == int(old)
            destination[selected] = int(new)
            changed |= bool(np.any(selected) and int(old) != int(new))
        output[name] = destination
    return output, changed


def _explicit_coordinates(seed_points: Any, depth: int) -> np.ndarray | None:
    if seed_points is None or isinstance(seed_points, Mapping):
        return None
    coordinates = np.asarray(seed_points)
    if coordinates.ndim != 2 or coordinates.shape[1] not in {2, 3}:
        raise ValueError("Explicit seed_points must have shape (n, 2) or (n, 3)")
    if coordinates.shape[1] == 2:
        if depth > 1:
            raise ValueError("Three-dimensional images require three-coordinate seed points")
        coordinates = np.column_stack([coordinates, np.zeros(len(coordinates), dtype=coordinates.dtype)])
    if np.any(coordinates != np.floor(coordinates)):
        raise ValueError("seed point coordinates must be integers")
    return coordinates.astype(int)


def _coordinates_in_ellipsoid(shape: tuple[int, int, int], center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    lower = np.maximum(np.floor(center - axes).astype(int), 0)
    upper = np.minimum(np.ceil(center + axes).astype(int) + 1, np.asarray(shape))
    candidates = np.stack(
        np.meshgrid(*(np.arange(start, stop) for start, stop in zip(lower, upper, strict=True)), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    distance = np.sum(((candidates - center) / axes) ** 2, axis=1)
    return candidates[distance <= 1]


def _radius_key(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def census_image(
    images: str | Path | Mapping[str, str | Path | np.ndarray],
    radii: float | Sequence[float] | Mapping[str | float, Sequence[float]],
    sample_size: int | Sequence[int],
    os_pairs: Mapping[str, Any] | None = None,
    *,
    seed_points: Any = None,
    seed_indices: Sequence[int] | np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    cores: int | None = None,
    object_remap: Mapping[str, Mapping[int, int]] | None = None,
) -> CensusResult:
    """Census object/scalar images with local randomness and explicit plans.

    Float radii are isotropic. A mapping such as ``{10: [35, 35, 1]}``
    provides the named physical radius and its R-compatible X/Y/Z pixel axes.
    """
    worker_count = 1 if cores is None else int(cores)
    if worker_count <= 0:
        raise ValueError("cores must be a positive integer or None")
    background_value = 0
    named_images = {"O1": images} if isinstance(images, (str, Path)) else images
    loaded, image_types = normalize_images(named_images)
    loaded, remap_applied = _apply_object_remap(loaded, object_remap)
    depth = loaded[next(iter(loaded))].shape[2]
    radius_plan = _radius_plan(radii, depth)
    sample_sizes = _sample_sizes(sample_size, len(radius_plan))
    thresholds = scalar_thresholds(loaded, image_types)
    normalized_pairs = normalize_os_pairs(os_pairs, loaded)
    allowed_values = seed_points if isinstance(seed_points, Mapping) else None
    eligible = eligible_seed_coordinates(
        loaded,
        image_types,
        thresholds,
        normalized_pairs,
        background_value,
        allowed_values,
    )
    explicit = _explicit_coordinates(seed_points, depth)
    if explicit is not None:
        selected = explicit
        sampling_mode = "explicit_coordinates"
    else:
        if len(eligible) == 0:
            raise ValueError("No eligible seed points were found")
        if seed_indices is not None:
            indices = np.asarray(seed_indices, dtype=int)
            if (
                indices.ndim != 1
                or np.any(indices < 0)
                or np.any(indices >= len(eligible))
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError("seed_indices must be unique valid zero-based indices into eligible seeds")
        else:
            generator = (
                random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
            )
            indices = generator.permutation(len(eligible))
        selected = eligible[indices]
        sampling_mode = "explicit_indices" if seed_indices is not None else "local_generator"
    max_requested = max(sample_sizes)
    if max_requested > len(selected):
        raise ValueError("Cannot sample more seed points than eligible pixels without replacement.")
    selected = selected[:max_requested]

    neighborhoods: list[Neighborhood] = []
    combined_patches: dict[str, list[pd.DataFrame]] = {}
    census_frames: list[pd.DataFrame] = []
    for (radius_name, axes), count in zip(radius_plan, sample_sizes, strict=True):
        radius_seeds = selected[:count]
        neighborhood_patches: dict[str, list[pd.DataFrame]] = {}
        radius_neighborhoods: list[Neighborhood] = []

        def measure(center: np.ndarray, radius_axes: np.ndarray = axes) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
            measured, _ = measure_image_neighborhood(loaded, center, radius_axes, thresholds, background_value)
            points = _coordinates_in_ellipsoid(next(iter(loaded.values())).shape[:3], center, radius_axes)
            return measured, points

        if worker_count > 1 and len(radius_seeds) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                measured_neighborhoods = list(executor.map(measure, radius_seeds))
        else:
            measured_neighborhoods = [measure(center) for center in radius_seeds]
        for neighborhood_id, (center, (measured, points)) in enumerate(
            zip(radius_seeds, measured_neighborhoods, strict=True), start=1
        ):
            for group_name, frame in measured.items():
                tagged = frame.copy()
                tagged["Nbhd"] = neighborhood_id
                neighborhood_patches.setdefault(group_name, []).append(tagged)
            radius_neighborhoods.append(
                Neighborhood(
                    center=tuple(float(value) for value in center),
                    radius=radius_name,
                    points=points,
                    patch_id=f"patch_{neighborhood_id}_{_radius_key(radius_name)}",
                    is_3d=depth > 1,
                )
            )
        radius_patch_frames = {
            name: pd.concat(frames, ignore_index=True) for name, frames in neighborhood_patches.items()
        }
        summary = summarize_patches(radius_patch_frames, normalized_pairs)
        measured_variables = list(summary.columns)
        for index, neighborhood in enumerate(radius_neighborhoods):
            neighborhood.variable_measurements = {name: float(summary.iloc[index][name]) for name in measured_variables}
        coordinates = pd.DataFrame(radius_seeds + 1, columns=["X", "Y", "Z"])
        summary = pd.concat([summary.reset_index(drop=True), coordinates], axis=1)
        summary["Radius"] = radius_name
        census_frames.append(summary)
        neighborhoods.extend(radius_neighborhoods)
        for name, frame in radius_patch_frames.items():
            tagged = frame.copy()
            tagged["Radius"] = radius_name
            combined_patches.setdefault(name, []).append(tagged)

    census = pd.concat(census_frames, ignore_index=True) if census_frames else pd.DataFrame()
    variable_names = [name for name in census.columns if name not in {"X", "Y", "Z", "Radius"}]
    patch_list = {name: pd.concat(frames, ignore_index=True) for name, frames in combined_patches.items()}
    metadata = {
        "source_images": {
            name: "array" if isinstance(value, np.ndarray) else str(value) for name, value in named_images.items()
        },
        "image_shapes": {name: list(array.shape) for name, array in loaded.items()},
        "image_types": image_types,
        "radii": [name for name, _ in radius_plan],
        "radius_axes": {str(name): axes.tolist() for name, axes in radius_plan},
        "sample_sizes": [len(frame) for frame in census_frames],
        "total_neighborhoods": len(neighborhoods),
        "cores": worker_count,
        "object_remap_applied_to_labels": remap_applied,
        "seed_sampling_mode": sampling_mode,
        "randomness": "explicit" if sampling_mode.startswith("explicit") else "python_local_generator",
        "r_parity_mode": "strict" if sampling_mode.startswith("explicit") else "algorithmic_with_python_rng",
    }
    return CensusResult(
        neighborhoods=neighborhoods,
        metadata=metadata,
        variables=variable_names,
        summary_stats=summarize_neighborhoods(neighborhoods),
        biomolecule_pairings=normalized_pairs,
        patch_list=patch_list,
        census=census,
    )


__all__ = ["census_image"]
