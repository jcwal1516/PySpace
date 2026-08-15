"""Three-dimensional connectivity, volume weights, and permutation tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from .spatial_primitives import (
    MoransIResult,
    SpatialPermutationResult,
    SpatialWeightMatrix,
    WeightMatrixType,
    validate_coordinates,
    weight_result,
)
from .spatial_stats_2d import calculate_morans_i


def _connectivity_mask(
    coordinates: np.ndarray,
    connectivity_type: int,
    distance_threshold: float,
) -> np.ndarray:
    if connectivity_type not in {6, 18, 26}:
        raise ValueError("connectivity_type must be 6, 18, or 26")
    differences = np.abs(coordinates[:, None, :] - coordinates[None, :, :])
    changed_axes = np.count_nonzero(differences > 1e-12, axis=2)
    allowed_axes = {6: 1, 18: 2, 26: 3}[connectivity_type]
    distances = np.linalg.norm(differences, axis=2)
    return (changed_axes > 0) & (changed_axes <= allowed_axes) & (distances <= distance_threshold)


def _default_threshold(coordinates: np.ndarray, connectivity_type: int) -> float:
    distances = cdist(coordinates, coordinates)
    mask = _connectivity_mask(coordinates, connectivity_type, np.inf)
    candidate = np.where(mask, distances, np.inf).min(axis=1)
    finite = candidate[np.isfinite(candidate)]
    if not len(finite):
        raise ValueError("No coordinate pairs satisfy the requested 3D connectivity")
    return float(finite.max())


def create_3d_volume_weight_matrix(
    coordinates: np.ndarray,
    volumes: np.ndarray | None = None,
    connectivity_type: int = 26,
    distance_threshold: float | None = None,
    volume_weighted: bool = True,
    row_standardize: bool = True,
) -> SpatialWeightMatrix:
    """Construct explicit 6/18/26-connectivity weights for 3D points."""
    points = validate_coordinates(coordinates, (3,))
    threshold = (
        _default_threshold(points, connectivity_type) if distance_threshold is None else float(distance_threshold)
    )
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("distance_threshold must be finite and positive")
    object_volumes = np.ones(len(points)) if volumes is None else np.asarray(volumes, dtype=float)
    if object_volumes.shape != (len(points),) or np.any(~np.isfinite(object_volumes)) or np.any(object_volumes <= 0):
        raise ValueError("volumes must contain one finite positive value per coordinate")
    distances = cdist(points, points)
    selected = _connectivity_mask(points, connectivity_type, threshold)
    weights = np.zeros_like(distances)
    weights[selected] = 1 / distances[selected]
    if volume_weighted:
        weights *= np.sqrt(object_volumes[:, None] * object_volumes[None, :])
    return weight_result(
        weights,
        WeightMatrixType.DISTANCE_BAND,
        {
            "connectivity_type": connectivity_type,
            "distance_threshold": threshold,
            "volume_weighted": volume_weighted,
            "is_3d": True,
        },
        row_standardize,
    )


def calculate_3d_morans_i(
    values: np.ndarray,
    coordinates: np.ndarray,
    volumes: np.ndarray | None = None,
    connectivity_type: int = 26,
    distance_threshold: float | None = None,
    compute_local: bool = True,
    significance_level: float = 0.05,
    alternative: str = "two-sided",
) -> MoransIResult:
    """Calculate Moran's I using validated volume-aware 3D weights."""
    weights = create_3d_volume_weight_matrix(
        coordinates,
        volumes,
        connectivity_type,
        distance_threshold,
        volume_weighted=volumes is not None,
    )
    result = calculate_morans_i(values, weights, compute_local, significance_level, alternative)
    result.interpretation = f"3D ({connectivity_type}-connectivity): {result.interpretation}"
    return result


def calculate_3d_spatial_connectivity(
    coordinates: np.ndarray,
    connectivity_type: int = 26,
    distance_threshold: float | None = None,
    compute_clustering: bool = True,
) -> dict[str, Any]:
    """Summarize the unweighted 3D connectivity graph."""
    points = validate_coordinates(coordinates, (3,))
    threshold = (
        _default_threshold(points, connectivity_type) if distance_threshold is None else float(distance_threshold)
    )
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("distance_threshold must be finite and positive")
    adjacency = _connectivity_mask(points, connectivity_type, threshold)
    graph = nx.from_numpy_array(adjacency.astype(int))
    components = list(nx.connected_components(graph))
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    result: dict[str, Any] = {
        "num_points": len(points),
        "num_edges": graph.number_of_edges(),
        "mean_degree": float(degrees.mean()),
        "density": float(nx.density(graph)),
        "num_components": len(components),
        "component_sizes": sorted((len(component) for component in components), reverse=True),
        "connectivity_type": connectivity_type,
        "distance_threshold": threshold,
        "analysis_type": "3d_spatial_connectivity",
    }
    if compute_clustering:
        result["mean_clustering_coefficient"] = float(nx.average_clustering(graph))
    for axis, name in enumerate(("x", "y", "z")):
        differences = np.abs(points[:, None, axis] - points[None, :, axis])
        selected = differences[adjacency & (differences > 0)]
        result[f"{name}_connectivity_range"] = float(selected.mean()) if len(selected) else 0.0
    return result


def _validate_plan(plan: Sequence[np.ndarray], count: int, size: int) -> list[np.ndarray]:
    if len(plan) != count:
        raise ValueError(f"Expected {count} permutation indices, got {len(plan)}")
    expected = np.arange(size)
    result = [np.asarray(indices, dtype=int) for indices in plan]
    if any(indices.shape != (size,) or not np.array_equal(np.sort(indices), expected) for indices in result):
        raise ValueError("Each permutation must contain every zero-based row index exactly once")
    return result


def _volume_plan(volumes: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    order = np.argsort(volumes, kind="stable")
    block_count = min(10, max(1, len(volumes) // 5))
    result = np.arange(len(volumes))
    for block in np.array_split(order, block_count):
        result[block] = generator.permutation(block)
    return result


def volume_preserving_permutation_test_3d(
    values: np.ndarray,
    coordinates: np.ndarray,
    volumes: np.ndarray | None = None,
    test_statistic: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float] | None = None,
    num_permutations: int = 999,
    connectivity_type: int = 26,
    preserve_volume_structure: bool = True,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    permutation_indices: Sequence[np.ndarray] | None = None,
) -> SpatialPermutationResult:
    """Run a local-RNG 3D permutation test, optionally within volume blocks."""
    observations = np.asarray(values, dtype=float)
    points = validate_coordinates(coordinates, (3,))
    object_volumes = None if volumes is None else np.asarray(volumes, dtype=float)
    if observations.shape != (len(points),) or np.any(~np.isfinite(observations)):
        raise ValueError("values must contain one finite observation per coordinate")
    if object_volumes is not None and (object_volumes.shape != observations.shape or np.any(object_volumes <= 0)):
        raise ValueError("volumes must contain one positive value per coordinate")
    if num_permutations <= 0 or not 0 < confidence_level < 1:
        raise ValueError("num_permutations must be positive and confidence_level must be in (0, 1)")

    def default_statistic(sample: np.ndarray, locations: np.ndarray, sizes: np.ndarray | None) -> float:
        return calculate_3d_morans_i(
            sample,
            locations,
            sizes,
            connectivity_type,
            compute_local=False,
        ).observed_i

    statistic = test_statistic or default_statistic
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    if permutation_indices is not None:
        plans = _validate_plan(permutation_indices, num_permutations, len(observations))
        permutation_method = "explicit"
    elif preserve_volume_structure and object_volumes is not None:
        plans = [_volume_plan(object_volumes, generator) for _ in range(num_permutations)]
        permutation_method = "3d_volume_preserving"
    else:
        plans = [generator.permutation(len(observations)) for _ in range(num_permutations)]
        permutation_method = "unrestricted"
    observed = float(statistic(observations, points, object_volumes))
    null = np.asarray([statistic(observations[indices], points, object_volumes) for indices in plans], dtype=float)
    if np.any(~np.isfinite(null)):
        raise ValueError("test_statistic returned non-finite permutation values")
    standard_deviation = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    alpha = 1 - confidence_level
    return SpatialPermutationResult(
        observed,
        null,
        float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1)),
        float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
        float((observed - null.mean()) / standard_deviation) if standard_deviation else 0.0,
        (float(np.quantile(null, alpha / 2)), float(np.quantile(null, 1 - alpha / 2))),
        len(null),
        permutation_method,
    )


__all__ = [
    "calculate_3d_morans_i",
    "calculate_3d_spatial_connectivity",
    "create_3d_volume_weight_matrix",
    "volume_preserving_permutation_test_3d",
]
