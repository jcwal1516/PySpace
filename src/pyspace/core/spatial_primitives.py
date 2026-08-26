"""Validated result types and small mathematical primitives for spatial statistics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse


class WeightMatrixType(Enum):
    """Supported spatial-weight constructions."""

    QUEEN = "queen"
    ROOK = "rook"
    KNN = "knn"
    DISTANCE_BAND = "distance_band"
    INVERSE_DISTANCE = "inverse_distance"
    KERNEL = "kernel"


@dataclass
class SpatialWeightMatrix:
    """Spatial weights plus the construction facts needed to interpret them."""

    weights: np.ndarray | csr_matrix
    matrix_type: WeightMatrixType
    parameters: dict[str, Any]
    n_neighbors_mean: float
    connectivity_histogram: np.ndarray
    is_symmetric: bool
    is_row_standardized: bool

    def to_dense(self) -> np.ndarray:
        source = cast(Any, self.weights).toarray() if issparse(self.weights) else self.weights
        return np.asarray(source, dtype=float)


@dataclass
class MoransIResult:
    observed_i: float
    expected_i: float
    variance_i: float
    z_score: float
    p_value: float
    p_value_one_sided: float
    interpretation: str
    local_i: np.ndarray | None = None
    local_p_values: np.ndarray | None = None
    local_z_scores: np.ndarray | None = None
    hotspots: np.ndarray | None = None
    coldspots: np.ndarray | None = None


@dataclass
class GearyCResult:
    observed_c: float
    expected_c: float
    variance_c: float
    z_score: float
    p_value: float
    p_value_one_sided: float
    interpretation: str


@dataclass
class LISAResult:
    local_statistics: np.ndarray
    local_p_values: np.ndarray
    local_z_scores: np.ndarray
    spatial_lag: np.ndarray
    cluster_types: np.ndarray
    significance_mask: np.ndarray
    quadrant_labels: list[str]


@dataclass
class GetisOrdResult:
    g_statistics: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    hotspots: np.ndarray
    coldspots: np.ndarray
    expected_g: np.ndarray
    variance_g: np.ndarray


@dataclass
class SpatialPermutationResult:
    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    p_value_one_sided: float
    effect_size: float
    confidence_interval: tuple[float, float]
    num_permutations: int
    permutation_method: str


def validate_coordinates(coordinates: np.ndarray, dimensions: tuple[int, ...] = (2, 3)) -> np.ndarray:
    """Return finite coordinate data with an allowed dimensionality."""
    result = np.asarray(coordinates, dtype=float)
    if result.ndim != 2 or result.shape[1] not in dimensions:
        expected = " or ".join(str(value) for value in dimensions)
        raise ValueError(f"coordinates must have shape (n, {expected})")
    if len(result) == 0 or np.any(~np.isfinite(result)):
        raise ValueError("coordinates must be non-empty and finite")
    return result


def weight_result(
    weights: np.ndarray,
    matrix_type: WeightMatrixType,
    parameters: dict[str, Any],
    row_standardize: bool,
) -> SpatialWeightMatrix:
    """Optionally row-standardize weights and derive connectivity metadata."""
    result = np.asarray(weights, dtype=float).copy()
    original_symmetric = bool(np.allclose(result, result.T))
    if row_standardize:
        row_sums = result.sum(axis=1)
        nonzero = row_sums > 0
        result[nonzero] /= row_sums[nonzero, None]
    counts = np.count_nonzero(result, axis=1)
    histogram = np.bincount(counts, minlength=int(counts.max()) + 1 if len(counts) else 1)
    stored: np.ndarray | csr_matrix = (
        csr_matrix(result) if len(result) > 50 and np.count_nonzero(result) < result.size * 0.3 else result
    )
    return SpatialWeightMatrix(
        weights=stored,
        matrix_type=matrix_type,
        parameters=parameters,
        n_neighbors_mean=float(counts.mean()) if len(counts) else 0.0,
        connectivity_histogram=histogram,
        is_symmetric=original_symmetric and not row_standardize or bool(np.allclose(result, result.T)),
        is_row_standardized=row_standardize,
    )


__all__ = [
    "GearyCResult",
    "GetisOrdResult",
    "LISAResult",
    "MoransIResult",
    "SpatialPermutationResult",
    "SpatialWeightMatrix",
    "WeightMatrixType",
    "validate_coordinates",
    "weight_result",
]
