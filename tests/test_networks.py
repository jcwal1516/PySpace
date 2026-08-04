from __future__ import annotations

import importlib.util
import os
from typing import Literal

import numpy as np
import pytest

from pyspace.core.networks import (
    analyze_network_topology,
    calculate_network_modularity,
    construct_spatial_network,
    create_distance_matrix_from_transmi,
    detect_communities,
)


def test_louvain_community_detection_is_seeded_and_uses_precomputed_distances() -> None:
    distances = np.array(
        [
            [0.0, 0.1, 2.0, 2.0],
            [0.1, 0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0, 0.1],
            [2.0, 2.0, 0.1, 0.0],
        ]
    )
    first = construct_spatial_network(distances, threshold_method="fixed", threshold_percentile=0.5, min_edges=0)
    second = construct_spatial_network(distances, threshold_method="fixed", threshold_percentile=0.5, min_edges=0)

    first_result = detect_communities(first, algorithm="louvain", random_state=4)
    second_result = detect_communities(second, algorithm="louvain", random_state=4)

    assert first_result["partition"] == second_result["partition"]
    assert first_result["num_communities"] == 2
    assert first_result["silhouette_score"] > 0.8


def test_network_rejects_negative_distances() -> None:
    distances = np.array([[0.0, -1.0], [-1.0, 0.0]])

    try:
        construct_spatial_network(distances)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative distances were accepted")


def test_label_propagation_modularity_topology_and_transmi_conversion() -> None:
    distances = np.array(
        [
            [0.0, 0.1, 3.0, 3.0],
            [0.1, 0.0, 3.0, 3.0],
            [3.0, 3.0, 0.0, 0.1],
            [3.0, 3.0, 0.1, 0.0],
        ]
    )
    result = construct_spatial_network(
        distances,
        group_labels=["A", "A", "B", "B"],
        sample_names=["a1", "a2", "b1", "b2"],
        threshold_method="fixed",
        threshold_percentile=0.5,
        min_edges=0,
    )

    communities = detect_communities(result, algorithm="label_prop", random_state=3, num_iterations=2)
    modularity = calculate_network_modularity(result, num_null_networks=3, random_state=4)
    topology = analyze_network_topology(result, ["degree", "pagerank"])
    matrix, names = create_distance_matrix_from_transmi(
        {
            "group_names": ["A", "B"],
            "kl_divergence": {("A", "B"): {"O1.1": 2.5}},
        },
        "O1.1",
    )

    assert communities["algorithm"] == "label_prop"
    assert modularity["observed"] > 0
    assert topology["basic_stats"]["num_edges"] == 2
    assert names == ["A", "B"]
    np.testing.assert_array_equal(matrix, [[0.0, 2.5], [2.5, 0.0]])


@pytest.mark.parametrize(("algorithm", "module"), [("leiden", "leidenalg"), ("infomap", "infomap")])
def test_optional_community_backends_are_real(algorithm: Literal["leiden", "infomap"], module: str) -> None:
    if importlib.util.find_spec(module) is None:
        if os.environ.get("PYSPACE_REQUIRE_COMMUNITY") == "1":
            pytest.fail(f"community job did not install {module}")
        pytest.skip(f"{module} is not installed")
    distances = np.array(
        [
            [0.0, 0.1, 3.0, 3.0],
            [0.1, 0.0, 3.0, 3.0],
            [3.0, 3.0, 0.0, 0.1],
            [3.0, 3.0, 0.1, 0.0],
        ]
    )
    network = construct_spatial_network(distances, threshold_method="fixed", threshold_percentile=0.5, min_edges=0)

    result = detect_communities(network, algorithm=algorithm, random_state=5, num_iterations=2)

    assert result["algorithm"] == algorithm
    assert result["num_communities"] == 2
