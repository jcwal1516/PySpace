"""Two-dimensional spatial weights, autocorrelation, and permutation tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from .spatial_primitives import (
    GearyCResult,
    GetisOrdResult,
    LISAResult,
    MoransIResult,
    SpatialPermutationResult,
    SpatialWeightMatrix,
    WeightMatrixType,
    validate_coordinates,
    weight_result,
)


def _default_distance(distances: np.ndarray) -> float:
    masked = np.where(distances > 0, distances, np.inf)
    nearest = masked.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    if not len(finite):
        raise ValueError("At least two distinct coordinate locations are required")
    return float(finite.max())


def create_spatial_weight_matrix(  # noqa: PLR0913 - explicit public weight construction controls
    coordinates: np.ndarray,
    matrix_type: WeightMatrixType = WeightMatrixType.QUEEN,
    k_neighbors: int = 8,
    distance_threshold: float | None = None,
    kernel_bandwidth: float | None = None,
    row_standardize: bool = True,
    include_self: bool = False,
    binary_weights: bool = False,
) -> SpatialWeightMatrix:
    """Construct weights for finite 2D points using one named rule."""
    points = validate_coordinates(coordinates, (2,))
    distances = cdist(points, points)
    threshold = _default_distance(distances) if distance_threshold is None else float(distance_threshold)
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("distance_threshold must be finite and positive")
    weights = np.zeros_like(distances)
    if matrix_type == WeightMatrixType.KNN:
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        count = min(k_neighbors, len(points) - 1)
        for row in range(len(points)):
            neighbors = np.argsort(distances[row], kind="stable")[1 : count + 1]
            weights[row, neighbors] = 1.0
        weights = np.maximum(weights, weights.T)
    elif matrix_type in {WeightMatrixType.ROOK, WeightMatrixType.QUEEN}:
        difference = np.abs(points[:, None, :] - points[None, :, :])
        nonzero_axes = np.count_nonzero(difference > 1e-12, axis=2)
        within = np.max(difference, axis=2) <= threshold
        allowed_axes = 1 if matrix_type == WeightMatrixType.ROOK else 2
        weights[(nonzero_axes > 0) & (nonzero_axes <= allowed_axes) & within] = 1.0
    elif matrix_type == WeightMatrixType.DISTANCE_BAND:
        weights[(distances > 0) & (distances <= threshold)] = 1.0
    elif matrix_type == WeightMatrixType.INVERSE_DISTANCE:
        selected = (distances > 0) & (distances <= threshold)
        weights[selected] = 1 / distances[selected]
    elif matrix_type == WeightMatrixType.KERNEL:
        bandwidth = threshold if kernel_bandwidth is None else float(kernel_bandwidth)
        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError("kernel_bandwidth must be finite and positive")
        selected = distances <= threshold
        weights[selected] = np.exp(-0.5 * (distances[selected] / bandwidth) ** 2)
    else:
        raise ValueError(f"Unsupported 2D weight type: {matrix_type.value}")
    if binary_weights:
        weights = (weights > 0).astype(float)
    np.fill_diagonal(weights, 1.0 if include_self else 0.0)
    return weight_result(
        weights,
        matrix_type,
        {
            "k_neighbors": k_neighbors,
            "distance_threshold": threshold,
            "kernel_bandwidth": kernel_bandwidth,
            "include_self": include_self,
            "binary_weights": binary_weights,
        },
        row_standardize,
    )


def _values_and_weights(values: np.ndarray, weight_matrix: SpatialWeightMatrix) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(values, dtype=float)
    weights = weight_matrix.to_dense()
    if observations.ndim != 1 or len(observations) != len(weights) or weights.shape[0] != weights.shape[1]:
        raise ValueError("values length must match a square weight matrix")
    if np.any(~np.isfinite(observations)):
        raise ValueError("values must contain only finite observations")
    if len(observations) < 3:
        raise ValueError("At least three observations are required")
    if weights.sum() <= 0:
        raise ValueError("Weight matrix contains no spatial connections")
    return observations, weights


def _p_values(z_score: float, alternative: str) -> tuple[float, float]:
    if alternative == "two-sided":
        return float(2 * stats.norm.sf(abs(z_score))), float(stats.norm.sf(z_score))
    if alternative == "greater":
        value = float(stats.norm.sf(z_score))
        return value, value
    if alternative == "less":
        value = float(stats.norm.cdf(z_score))
        return value, value
    raise ValueError("alternative must be two-sided, greater, or less")


def calculate_morans_i(
    values: np.ndarray,
    weight_matrix: SpatialWeightMatrix,
    compute_local: bool = True,
    significance_level: float = 0.05,
    alternative: str = "two-sided",
) -> MoransIResult:
    """Calculate global Moran's I and optional normal-approximation local I."""
    observations, weights = _values_and_weights(values, weight_matrix)
    count = len(observations)
    deviations = observations - observations.mean()
    squared_sum = float(deviations @ deviations)
    if squared_sum == 0:
        raise ValueError("Moran's I is undefined for constant values")
    weight_sum = float(weights.sum())
    observed = float((count / weight_sum) * (deviations @ weights @ deviations) / squared_sum)
    expected = -1 / (count - 1)
    s1 = 0.5 * float(np.sum((weights + weights.T) ** 2))
    s2 = float(np.sum((weights.sum(axis=1) + weights.sum(axis=0)) ** 2))
    b2 = count * float(np.sum(deviations**4)) / squared_sum**2
    denominator = (count - 1) * (count - 2) * (count - 3) * weight_sum**2
    if denominator:
        numerator = count * ((count**2 - 3 * count + 3) * s1 - count * s2 + 3 * weight_sum**2)
        numerator -= b2 * ((count**2 - count) * s1 - 2 * count * s2 + 6 * weight_sum**2)
        variance = max(numerator / denominator - expected**2, np.finfo(float).eps)
    else:
        variance = float("nan")
    z_score = float((observed - expected) / np.sqrt(variance)) if np.isfinite(variance) else float("nan")
    p_value, one_sided = _p_values(z_score, alternative)
    relationship = "positive" if observed > expected else "negative"
    pattern = "clustered" if relationship == "positive" else "dispersed"
    qualifier = "Significant" if p_value < significance_level else "Weak"
    local_i = local_z = local_p = hotspots = coldspots = None
    if compute_local:
        standardized = deviations / np.sqrt(squared_sum / (count - 1))
        lag = weights @ standardized
        local_i = standardized * lag
        local_scale = np.sqrt(np.maximum(np.sum(weights**2, axis=1), np.finfo(float).eps))
        local_z = local_i / local_scale
        local_p = 2 * stats.norm.sf(np.abs(local_z))
        significant = local_p < significance_level
        hotspots = significant & (local_i > 0) & (standardized > 0)
        coldspots = significant & (local_i > 0) & (standardized < 0)
    return MoransIResult(
        observed_i=observed,
        expected_i=expected,
        variance_i=variance,
        z_score=z_score,
        p_value=p_value,
        p_value_one_sided=one_sided,
        interpretation=f"{qualifier} {relationship} spatial autocorrelation ({pattern} pattern)",
        local_i=local_i,
        local_p_values=local_p,
        local_z_scores=local_z,
        hotspots=hotspots,
        coldspots=coldspots,
    )


def calculate_gearys_c(
    values: np.ndarray,
    weight_matrix: SpatialWeightMatrix,
    significance_level: float = 0.05,
    alternative: str = "two-sided",
) -> GearyCResult:
    """Calculate Geary's C with a documented normal approximation."""
    observations, weights = _values_and_weights(values, weight_matrix)
    count = len(observations)
    deviations = observations - observations.mean()
    squared_differences = (observations[:, None] - observations[None, :]) ** 2
    observed = float(
        (count - 1) * np.sum(weights * squared_differences) / (2 * weights.sum() * (deviations @ deviations))
    )
    expected = 1.0
    variance = 1 / max(count - 1, 1)
    z_score = float((observed - expected) / np.sqrt(variance))
    p_value, one_sided = _p_values(z_score, alternative)
    relationship = "positive" if observed < expected else "negative"
    qualifier = "Significant" if p_value < significance_level else "Weak"
    return GearyCResult(
        observed, expected, variance, z_score, p_value, one_sided, f"{qualifier} {relationship} spatial association"
    )


def lisa_analysis(
    values: np.ndarray,
    weight_matrix: SpatialWeightMatrix,
    significance_level: float = 0.05,
) -> LISAResult:
    """Return local Moran statistics and HH/LL/HL/LH quadrant labels."""
    observations, weights = _values_and_weights(values, weight_matrix)
    standardized = (observations - observations.mean()) / observations.std(ddof=1)
    lag = weights @ standardized
    local = standardized * lag
    scale = np.sqrt(np.maximum(np.sum(weights**2, axis=1), np.finfo(float).eps))
    z_scores = local / scale
    p_values = 2 * stats.norm.sf(np.abs(z_scores))
    labels = np.select(
        [
            (standardized >= 0) & (lag >= 0),
            (standardized < 0) & (lag < 0),
            (standardized >= 0) & (lag < 0),
        ],
        ["HH", "LL", "HL"],
        default="LH",
    )
    return LISAResult(local, p_values, z_scores, lag, labels, p_values < significance_level, ["HH", "LL", "HL", "LH"])


def spatial_hotspot_detection(
    values: np.ndarray,
    weight_matrix: SpatialWeightMatrix,
    significance_level: float = 0.05,
) -> GetisOrdResult:
    """Calculate local Getis-Ord G* z scores."""
    observations, weights = _values_and_weights(values, weight_matrix)
    count = len(observations)
    sums = weights.sum(axis=1)
    squared_sums = np.sum(weights**2, axis=1)
    expected = observations.mean() * sums
    variance = observations.std(ddof=1) ** 2 * np.maximum((count * squared_sums - sums**2) / (count - 1), 0)
    numerator = weights @ observations - expected
    z_scores = np.divide(numerator, np.sqrt(variance), out=np.zeros_like(numerator), where=variance > 0)
    p_values = 2 * stats.norm.sf(np.abs(z_scores))
    significant = p_values < significance_level
    return GetisOrdResult(
        weights @ observations,
        z_scores,
        p_values,
        significant & (z_scores > 0),
        significant & (z_scores < 0),
        expected,
        variance,
    )


def _validate_permutation_plan(plan: Sequence[np.ndarray], count: int, size: int) -> list[np.ndarray]:
    if len(plan) != count:
        raise ValueError(f"Expected {count} permutation indices, got {len(plan)}")
    expected = np.arange(size)
    result = [np.asarray(indices, dtype=int) for indices in plan]
    if any(indices.shape != (size,) or not np.array_equal(np.sort(indices), expected) for indices in result):
        raise ValueError("Each permutation index must contain every zero-based row index exactly once")
    return result


def spatial_permutation_test(  # noqa: PLR0913 - explicit statistical controls
    values: np.ndarray,
    coordinates: np.ndarray,
    test_statistic: Callable[[np.ndarray, np.ndarray], float],
    num_permutations: int = 999,
    preserve_structure_method: Literal["unrestricted", "constrained", "blocks", "distance"] = "constrained",
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    permutation_indices: Sequence[np.ndarray] | None = None,
) -> SpatialPermutationResult:
    """Test a statistic with local RNG or a fully explicit permutation plan."""
    observations = np.asarray(values, dtype=float)
    points = validate_coordinates(coordinates)
    if observations.ndim != 1 or len(observations) != len(points) or np.any(~np.isfinite(observations)):
        raise ValueError("values must be one finite observation per coordinate")
    if num_permutations <= 0 or not 0 < confidence_level < 1:
        raise ValueError("num_permutations must be positive and confidence_level must be in (0, 1)")
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    plans = (
        _validate_permutation_plan(permutation_indices, num_permutations, len(observations))
        if permutation_indices is not None
        else [generator.permutation(len(observations)) for _ in range(num_permutations)]
    )
    if preserve_structure_method not in {"unrestricted", "constrained", "blocks", "distance"}:
        raise ValueError("Unknown preserve_structure_method")
    observed = float(test_statistic(observations, points))
    null = np.asarray([test_statistic(observations[indices], points) for indices in plans], dtype=float)
    if np.any(~np.isfinite(null)):
        raise ValueError("test_statistic returned a non-finite permutation result")
    standard_deviation = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    alpha = 1 - confidence_level
    return SpatialPermutationResult(
        observed_statistic=observed,
        null_distribution=null,
        p_value=float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1)),
        p_value_one_sided=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
        effect_size=float((observed - null.mean()) / standard_deviation) if standard_deviation else 0.0,
        confidence_interval=(float(np.quantile(null, alpha / 2)), float(np.quantile(null, 1 - alpha / 2))),
        num_permutations=len(null),
        permutation_method="explicit" if permutation_indices is not None else preserve_structure_method,
    )


def spatial_fdr_correction(
    p_values: np.ndarray,
    coordinates: np.ndarray,
    method: Literal["spatial_benjamini_hochberg", "spatial_bonferroni"] = "spatial_benjamini_hochberg",
    neighborhood_size: int = 8,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply BH or Bonferroni with an explicitly reported spatial effective count."""
    probabilities = np.asarray(p_values, dtype=float)
    points = validate_coordinates(coordinates)
    if probabilities.shape != (len(points),) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("p_values must provide one probability per coordinate")
    weights = create_spatial_weight_matrix(points[:, :2], WeightMatrixType.KNN, k_neighbors=neighborhood_size)
    autocorrelation = (
        calculate_morans_i(probabilities, weights, compute_local=False).observed_i
        if len(points) >= 4 and probabilities.std()
        else 0.0
    )
    effective = max(1, int(len(points) * (1 - max(0.0, autocorrelation) * 0.5)))
    if method == "spatial_bonferroni":
        corrected = np.minimum(probabilities * effective, 1.0)
    elif method == "spatial_benjamini_hochberg":
        order = np.argsort(probabilities)
        ranked = probabilities[order] * effective / np.arange(1, len(probabilities) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        corrected = np.empty_like(ranked)
        corrected[order] = np.minimum(ranked, 1.0)
    else:
        raise ValueError(f"Unknown spatial correction method: {method}")
    rejected = corrected < alpha
    return {
        "corrected_p_values": corrected,
        "rejected": rejected,
        "n_discoveries": int(rejected.sum()),
        "effective_n_tests": effective,
        "spatial_autocorr": float(autocorrelation),
        "method": method,
        "alpha": alpha,
    }


__all__ = [
    "calculate_gearys_c",
    "calculate_morans_i",
    "create_spatial_weight_matrix",
    "lisa_analysis",
    "spatial_fdr_correction",
    "spatial_hotspot_detection",
    "spatial_permutation_test",
]
