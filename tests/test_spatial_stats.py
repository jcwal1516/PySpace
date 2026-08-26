from __future__ import annotations

import numpy as np

from pyspace.core.spatial_primitives import WeightMatrixType
from pyspace.core.spatial_stats_2d import (
    calculate_morans_i,
    create_spatial_weight_matrix,
    spatial_permutation_test,
)
from pyspace.core.spatial_stats_3d import calculate_3d_spatial_connectivity


def test_distance_band_weights_and_morans_i_are_vectorized_and_finite() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    weights = create_spatial_weight_matrix(
        coordinates,
        matrix_type=WeightMatrixType.DISTANCE_BAND,
        distance_threshold=1.1,
        row_standardize=False,
    )

    result = calculate_morans_i(np.array([0.0, 1.0, 2.0, 3.0]), weights, compute_local=False)

    assert np.isfinite(result.observed_i)
    assert weights.to_dense()[0, 1] == 1
    assert weights.to_dense()[0, 2] == 0


def test_spatial_permutation_uses_explicit_plan_without_global_rng_mutation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    coordinates = np.column_stack([np.arange(4), np.zeros(4)])
    plans = [np.array([3, 2, 1, 0]), np.array([1, 0, 3, 2])]
    np.random.seed(99)
    expected = np.random.random(3)
    np.random.seed(99)

    result = spatial_permutation_test(
        values,
        coordinates,
        lambda sample, _coordinates: float(sample[0] - sample[-1]),
        num_permutations=2,
        permutation_indices=plans,
    )

    np.testing.assert_array_equal(np.random.random(3), expected)
    assert result.num_permutations == 2
    np.testing.assert_array_equal(result.null_distribution, np.array([3.0, -1.0]))


def test_3d_connectivity_distinguishes_face_from_corner_neighbors() -> None:
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=float)

    face = calculate_3d_spatial_connectivity(coordinates, connectivity_type=6, distance_threshold=2.0)
    corner = calculate_3d_spatial_connectivity(coordinates, connectivity_type=26, distance_threshold=2.0)

    assert face["num_edges"] == 1
    assert corner["num_edges"] == 3
