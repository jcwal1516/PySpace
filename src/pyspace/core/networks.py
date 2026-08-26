"""Small NetworkX layer with explicit optional community backends."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


@dataclass
class NetworkResult:
    """Network, distance data, derived communities, and statistics."""

    network: nx.Graph | None = None
    communities: dict[str, Any] = field(default_factory=dict)
    modularity_scores: dict[str, Any] = field(default_factory=dict)
    centrality_measures: dict[str, Any] = field(default_factory=dict)
    topology_stats: dict[str, Any] = field(default_factory=dict)
    distance_matrix: np.ndarray | None = None
    group_labels: list[str] | None = None
    sample_names: list[str] | None = None

    def __repr__(self) -> str:
        nodes = self.network.number_of_nodes() if self.network is not None else 0
        edges = self.network.number_of_edges() if self.network is not None else 0
        partition = self.communities.get("partition", {})
        communities = len(set(partition.values())) if isinstance(partition, dict) else 0
        observed = self.modularity_scores.get("observed")
        modularity = "N/A" if observed is None else f"{float(observed):.3f}"
        return f"NetworkResult(nodes={nodes}, edges={edges}, communities={communities}, modularity={modularity})"


def _distance_array(distance_matrix: np.ndarray | pd.DataFrame) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    if np.any(np.isinf(matrix)):
        raise ValueError("Distance matrix must not contain infinite values")
    finite = np.isfinite(matrix)
    if np.any(matrix[finite] < 0):
        raise ValueError("Distance matrix must contain non-negative distances")
    if not np.allclose(matrix, matrix.T, equal_nan=True):
        raise ValueError("Distance matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("Distance matrix diagonal must be zero")
    return matrix


def _similarities(matrix: np.ndarray, method: str) -> np.ndarray:
    finite = np.isfinite(matrix)
    similarities = np.full(matrix.shape, np.nan, dtype=float)
    if method == "inverse":
        similarities[finite] = 1 / (1 + matrix[finite])
    elif method == "negative_exp":
        positive = matrix[finite & (matrix > 0)]
        scale = float(np.median(positive)) if len(positive) else 1.0
        similarities[finite] = np.exp(-matrix[finite] / scale)
    elif method == "complement":
        maximum = float(np.max(matrix[finite])) if np.any(finite) else 0.0
        similarities[finite] = maximum - matrix[finite]
    else:
        raise ValueError("similarity_transform must be inverse, negative_exp, or complement")
    np.fill_diagonal(similarities, np.nan)
    return similarities


def _threshold(similarities: np.ndarray, method: str, value: float, min_edges: int) -> float:
    candidates = similarities[np.triu_indices_from(similarities, k=1)]
    candidates = candidates[np.isfinite(candidates)]
    if not len(candidates):
        return np.inf
    if method == "fixed":
        return float(value)
    if not 0 <= value <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100")
    percentile = float(np.percentile(candidates, value))
    if method == "percentile":
        return percentile
    if method != "adaptive":
        raise ValueError("threshold_method must be adaptive, percentile, or fixed")
    if min_edges <= 0:
        return percentile
    required = min(min_edges, len(candidates))
    minimum_threshold = float(np.sort(candidates)[-required])
    return min(percentile, minimum_threshold)


def _topology(network: nx.Graph) -> dict[str, Any]:
    components = list(nx.connected_components(network))
    degrees = [degree for _, degree in network.degree()]
    return {
        "num_nodes": network.number_of_nodes(),
        "num_edges": network.number_of_edges(),
        "density": float(nx.density(network)),
        "is_connected": bool(nx.is_connected(network)) if network.number_of_nodes() else False,
        "num_components": len(components),
        "component_sizes": [len(component) for component in components],
        "average_degree": float(np.mean(degrees)) if degrees else 0.0,
    }


def construct_spatial_network(
    distance_matrix: np.ndarray | pd.DataFrame,
    group_labels: list[str] | None = None,
    sample_names: list[str] | None = None,
    threshold_method: str = "adaptive",
    threshold_percentile: float = 75.0,
    min_edges: int = 10,
    weighted: bool = True,
    similarity_transform: str = "inverse",
) -> NetworkResult:
    """Construct a sample graph from a validated precomputed distance matrix."""
    matrix = _distance_array(distance_matrix)
    sample_count = len(matrix)
    names = sample_names or [f"sample_{index}" for index in range(sample_count)]
    if len(names) != sample_count:
        raise ValueError("sample_names length must match the distance matrix")
    if group_labels is not None and len(group_labels) != sample_count:
        raise ValueError("group_labels length must match the distance matrix")
    if min_edges < 0:
        raise ValueError("min_edges cannot be negative")
    similarities = _similarities(matrix, similarity_transform)
    cutoff = _threshold(similarities, threshold_method, threshold_percentile, min_edges)
    network = nx.Graph()
    for index, name in enumerate(names):
        attributes = {"sample_name": name}
        if group_labels is not None:
            attributes["group"] = group_labels[index]
        network.add_node(index, **attributes)
    for left in range(sample_count):
        for right in range(left + 1, sample_count):
            similarity = similarities[left, right]
            if np.isfinite(similarity) and similarity >= cutoff:
                edge_attributes: dict[str, Any] = {"distance": float(matrix[left, right])}
                if weighted:
                    edge_attributes["weight"] = float(similarity)
                network.add_edge(left, right, **edge_attributes)
    return NetworkResult(
        network=network,
        distance_matrix=matrix.copy(),
        group_labels=None if group_labels is None else list(group_labels),
        sample_names=list(names),
        topology_stats=_topology(network),
    )


def _sets(partition: dict[int, int]) -> list[set[int]]:
    communities: dict[int, set[int]] = {}
    for node, community_id in partition.items():
        communities.setdefault(int(community_id), set()).add(node)
    return list(communities.values())


def _canonical_partition(communities: list[set[int]]) -> dict[int, int]:
    ordered = sorted((set(community) for community in communities), key=lambda group: min(group))
    return {node: community_id for community_id, group in enumerate(ordered) for node in sorted(group)}


def _networkx_partition(
    network: nx.Graph,
    algorithm: Literal["louvain", "label_prop"],
    resolution: float,
    seed: int | None,
) -> dict[int, int]:
    if algorithm == "louvain":
        groups = nx.community.louvain_communities(network, weight="weight", resolution=resolution, seed=seed)
    else:
        groups = list(nx.community.asyn_lpa_communities(network, weight="weight", seed=seed))
    return _canonical_partition([set(group) for group in groups])


def _leiden_partition(network: nx.Graph, resolution: float, seed: int | None) -> dict[int, int]:
    try:
        import igraph
        import leidenalg
    except ImportError as exc:
        raise ImportError("Leiden requires the pyspace-analysis[community] extra") from exc
    nodes = list(network.nodes())
    positions = {node: index for index, node in enumerate(nodes)}
    edges = [(positions[left], positions[right]) for left, right in network.edges()]
    graph = igraph.Graph(n=len(nodes), edges=edges, directed=False)
    weights = [float(network[left][right].get("weight", 1.0)) for left, right in network.edges()]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights or None,
        resolution_parameter=resolution,
        seed=seed,
    )
    return {node: int(partition.membership[index]) for index, node in enumerate(nodes)}


def _infomap_partition(network: nx.Graph, seed: int | None) -> dict[int, int]:
    try:
        from infomap import Infomap
    except ImportError as exc:
        raise ImportError("Infomap requires the pyspace-analysis[community] extra") from exc
    nodes = list(network.nodes())
    positions = {node: index for index, node in enumerate(nodes)}
    options = "--silent" if seed is None else f"--silent --seed {seed}"
    model = Infomap(options)
    for left, right, attributes in network.edges(data=True):
        model.add_link(positions[left], positions[right], float(attributes.get("weight", 1.0)))
    result = model.run()
    result_nodes = getattr(result, "nodes", None)
    if callable(result_nodes):
        nodes_result = result_nodes(states=True)
    else:  # Infomap releases before the result-object API exposed nodes on the model.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            nodes_result = model.nodes
    membership = {int(node.node_id): int(node.module_id) for node in nodes_result}
    return {node: membership[positions[node]] for node in nodes}


def _partition_agreement(left: dict[int, int], right: dict[int, int]) -> float:
    nodes = sorted(set(left) & set(right))
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    if not pairs:
        return 1.0
    matches = sum((left[a] == left[b]) == (right[a] == right[b]) for a, b in pairs)
    return matches / len(pairs)


def _calculate_silhouette_score(distance_matrix: np.ndarray, partition: dict[int, int]) -> float:
    """Calculate silhouette quality using distances, never distance-derived features."""
    matrix = _distance_array(distance_matrix)
    nodes = sorted(partition)
    labels = np.asarray([partition[node] for node in nodes])
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    return float(silhouette_score(matrix[np.ix_(nodes, nodes)], labels, metric="precomputed"))


def detect_communities(
    network_result: NetworkResult,
    algorithm: Literal["louvain", "leiden", "label_prop", "infomap"] = "louvain",
    resolution: float = 1.0,
    random_state: int | None = None,
    num_iterations: int = 10,
) -> dict[str, Any]:
    """Detect communities without silently substituting another backend."""
    network = network_result.network
    if network is None or network.number_of_nodes() == 0:
        raise ValueError("NetworkResult must contain a non-empty network")
    if resolution <= 0 or num_iterations <= 0:
        raise ValueError("resolution and num_iterations must be positive")
    if network.number_of_edges() == 0:
        warnings.warn("Network has no edges; each node is its own community", RuntimeWarning, stacklevel=2)
        partitions = [{node: index for index, node in enumerate(network.nodes())}]
    else:
        partitions = []
        for iteration in range(num_iterations):
            iteration_seed = None if random_state is None else random_state + iteration
            if algorithm in {"louvain", "label_prop"}:
                partition = _networkx_partition(network, algorithm, resolution, iteration_seed)
            elif algorithm == "leiden":
                partition = _leiden_partition(network, resolution, iteration_seed)
            elif algorithm == "infomap":
                partition = _infomap_partition(network, iteration_seed)
            else:
                raise ValueError(f"Unsupported community algorithm: {algorithm}")
            partitions.append(partition)
    modularities = [
        float(nx.community.modularity(network, _sets(partition), weight="weight", resolution=resolution))
        for partition in partitions
    ]
    best_index = int(np.argmax(modularities))
    best = partitions[best_index]
    agreements = [
        _partition_agreement(partitions[left], partitions[right])
        for left in range(len(partitions))
        for right in range(left + 1, len(partitions))
    ]
    sizes = sorted(Counter(best.values()).values(), reverse=True)
    silhouette = (
        _calculate_silhouette_score(network_result.distance_matrix, best)
        if network_result.distance_matrix is not None
        else float("nan")
    )
    result: dict[str, Any] = {
        "partition": best,
        "modularity": modularities[best_index],
        "num_communities": len(sizes),
        "community_sizes": sizes,
        "stability": float(np.mean(agreements)) if agreements else 1.0,
        "silhouette_score": silhouette,
        "algorithm": algorithm,
        "resolution": resolution,
        "num_iterations": len(partitions),
        "all_modularities": modularities,
    }
    network_result.communities = result
    return result


def calculate_network_modularity(
    network_result: NetworkResult,
    num_null_networks: int = 100,
    resolution: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compare observed group modularity with local-RNG label permutations."""
    network = network_result.network
    labels = network_result.group_labels
    if network is None or labels is None or len(labels) != network.number_of_nodes():
        raise ValueError("NetworkResult must contain a network and one group label per node")
    if num_null_networks <= 0:
        raise ValueError("num_null_networks must be positive")
    nodes = list(network.nodes())
    observed_partition = _canonical_partition(
        [{node for node, label in zip(nodes, labels, strict=True) if label == group} for group in dict.fromkeys(labels)]
    )
    observed = float(
        nx.community.modularity(network, _sets(observed_partition), weight="weight", resolution=resolution)
    )
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    null_values = np.empty(num_null_networks, dtype=float)
    label_array = np.asarray(labels)
    for index in range(num_null_networks):
        shuffled = label_array[generator.permutation(len(label_array))]
        partition = _canonical_partition(
            [
                {node for node, label in zip(nodes, shuffled, strict=True) if label == group}
                for group in dict.fromkeys(shuffled)
            ]
        )
        null_values[index] = nx.community.modularity(network, _sets(partition), weight="weight", resolution=resolution)
    result = {
        "observed": observed,
        "null_distribution": null_values,
        "p_value": float((1 + np.count_nonzero(null_values >= observed)) / (num_null_networks + 1)),
        "z_score": float((observed - null_values.mean()) / null_values.std()) if null_values.std() else float("nan"),
        "null_model": "label_permutation",
    }
    network_result.modularity_scores = result
    return result


def analyze_network_topology(
    network_result: NetworkResult,
    centrality_measures: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate a focused set of NetworkX topology summaries."""
    network = network_result.network
    if network is None:
        raise ValueError("NetworkResult must contain a network")
    requested = centrality_measures or ["degree", "betweenness", "closeness"]
    available = {
        "degree": nx.degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "closeness": nx.closeness_centrality,
        "pagerank": nx.pagerank,
    }
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unsupported centrality measures: {unknown}")
    centrality = {name: dict(available[name](network)) for name in requested}
    result = {"basic_stats": _topology(network), "centrality": centrality}
    network_result.centrality_measures = centrality
    network_result.topology_stats.update(result["basic_stats"])
    return result


def create_distance_matrix_from_transmi(
    transmi_result: dict[str, Any],
    variable: str,
    distance_metric: str = "kl_divergence",
) -> tuple[np.ndarray, list[str]]:
    """Build a symmetric group distance matrix from structured transMI output."""
    group_names = transmi_result.get("group_names")
    distances = transmi_result.get(distance_metric)
    if not isinstance(group_names, list) or not isinstance(distances, dict):
        raise ValueError("transMI result must contain group_names and the requested distance mapping")
    matrix = np.zeros((len(group_names), len(group_names)), dtype=float)
    for pair, variables in distances.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(variables, dict)
            or variable not in variables
        ):
            continue
        left, right = (group_names.index(pair[0]), group_names.index(pair[1]))
        matrix[left, right] = matrix[right, left] = float(variables[variable])
    return matrix, list(group_names)


__all__ = [
    "NetworkResult",
    "analyze_network_topology",
    "calculate_network_modularity",
    "construct_spatial_network",
    "create_distance_matrix_from_transmi",
    "detect_communities",
]
