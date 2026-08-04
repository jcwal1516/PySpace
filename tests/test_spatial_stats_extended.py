from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

from pyspace.core.spatial_primitives import WeightMatrixType, validate_coordinates, weight_result
from pyspace.core.spatial_stats_2d import (
    calculate_gearys_c,
    calculate_morans_i,
    create_spatial_weight_matrix,
    lisa_analysis,
    spatial_fdr_correction,
    spatial_hotspot_detection,
    spatial_permutation_test,
)
from pyspace.core.spatial_stats_3d import (
    calculate_3d_morans_i,
    calculate_3d_spatial_connectivity,
    create_3d_volume_weight_matrix,
    volume_preserving_permutation_test_3d,
)

POINTS_2D = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
VALUES = np.array([1.0, 2.0, 1.5, 4.0, 5.0])


@pytest.mark.parametrize(
    "matrix_type",
    [
        WeightMatrixType.QUEEN,
        WeightMatrixType.ROOK,
        WeightMatrixType.KNN,
        WeightMatrixType.DISTANCE_BAND,
        WeightMatrixType.INVERSE_DISTANCE,
        WeightMatrixType.KERNEL,
    ],
)
def test_all_2d_weight_constructions_are_finite(matrix_type: WeightMatrixType) -> None:
    result = create_spatial_weight_matrix(
        POINTS_2D,
        matrix_type=matrix_type,
        k_neighbors=2,
        distance_threshold=1.5,
        kernel_bandwidth=0.75,
        include_self=matrix_type == WeightMatrixType.KERNEL,
        binary_weights=matrix_type == WeightMatrixType.INVERSE_DISTANCE,
    )

    assert result.matrix_type is matrix_type
    assert np.isfinite(result.to_dense()).all()
    np.testing.assert_allclose(result.to_dense().sum(axis=1), 1.0)


def test_spatial_statistics_return_all_local_outputs() -> None:
    weights = create_spatial_weight_matrix(
        POINTS_2D,
        WeightMatrixType.DISTANCE_BAND,
        distance_threshold=1.5,
        row_standardize=False,
    )

    moran = calculate_morans_i(VALUES, weights, compute_local=True, alternative="greater")
    geary = calculate_gearys_c(VALUES, weights, alternative="less")
    lisa = lisa_analysis(VALUES, weights, significance_level=1.0)
    hotspots = spatial_hotspot_detection(VALUES, weights, significance_level=1.0)

    assert moran.local_i is not None and moran.local_i.shape == VALUES.shape
    assert 0 <= geary.p_value <= 1
    assert set(lisa.cluster_types) <= {"HH", "LL", "HL", "LH"}
    assert hotspots.g_statistics.shape == VALUES.shape
    assert hotspots.hotspots.any() or hotspots.coldspots.any()


def test_spatial_fdr_and_generated_permutation_plans() -> None:
    probabilities = np.array([0.001, 0.02, 0.2, 0.6, 0.9])

    bh = spatial_fdr_correction(probabilities, POINTS_2D, neighborhood_size=2)
    bonferroni = spatial_fdr_correction(
        probabilities,
        POINTS_2D,
        method="spatial_bonferroni",
        neighborhood_size=2,
    )
    permutation = spatial_permutation_test(
        VALUES,
        POINTS_2D,
        lambda sample, _points: float(sample @ np.arange(len(sample))),
        num_permutations=3,
        preserve_structure_method="blocks",
        random_state=9,
    )

    assert bh["method"] == "spatial_benjamini_hochberg"
    assert bonferroni["method"] == "spatial_bonferroni"
    assert permutation.permutation_method == "blocks"
    assert permutation.null_distribution.shape == (3,)


def test_spatial_validation_and_sparse_storage() -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        validate_coordinates(np.empty((0, 2)))
    with pytest.raises(ValueError, match="shape"):
        validate_coordinates(np.ones((2, 4)))

    weights = np.zeros((60, 60))
    indices = np.arange(59)
    weights[indices, indices + 1] = 1
    weights[indices + 1, indices] = 1
    result = weight_result(weights, WeightMatrixType.ROOK, {}, row_standardize=False)

    assert issparse(result.weights)
    assert result.is_symmetric


def test_3d_weight_moran_connectivity_and_permutations() -> None:
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
        dtype=float,
    )
    volumes = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    values = np.array([1.0, 2.0, 1.5, 4.0, 5.0])

    weights = create_3d_volume_weight_matrix(
        points,
        volumes,
        connectivity_type=18,
        distance_threshold=1.5,
        row_standardize=False,
    )
    moran = calculate_3d_morans_i(
        values,
        points,
        volumes,
        connectivity_type=18,
        distance_threshold=1.5,
        compute_local=False,
    )
    connectivity = calculate_3d_spatial_connectivity(
        points,
        connectivity_type=26,
        distance_threshold=2.0,
        compute_clustering=False,
    )
    plans = [np.array([4, 3, 2, 1, 0]), np.array([1, 0, 3, 2, 4])]
    explicit = volume_preserving_permutation_test_3d(
        values,
        points,
        volumes,
        lambda sample, _points, _volumes: float(sample[0] - sample[-1]),
        num_permutations=2,
        permutation_indices=plans,
    )
    generated = volume_preserving_permutation_test_3d(
        values,
        points,
        volumes,
        lambda sample, _points, _volumes: float(sample @ np.arange(len(sample))),
        num_permutations=3,
        random_state=2,
    )

    assert weights.parameters["is_3d"] is True
    assert moran.interpretation.startswith("3D (18-connectivity)")
    assert "mean_clustering_coefficient" not in connectivity
    assert explicit.permutation_method == "explicit"
    assert generated.permutation_method == "3d_volume_preserving"
