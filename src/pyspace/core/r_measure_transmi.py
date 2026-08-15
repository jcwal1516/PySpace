"""
R SPACE compatible measure_transMI implementation.

This module provides a 1:1 implementation of R SPACE's measure_transMI function,
following the exact algorithm from the R package.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import combinations, repeat
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

# Use R-compatible nearest-grid rounding
from .distributions import build_dist as r_build_dist
from .distributions import r_round_column as _r_nearest_round
from .distributions import smooth_dist as r_smooth_dist


def _build_similarity_from_mi(mi_vals: np.ndarray, n_specimens: int) -> np.ndarray:
    """Reconstruct similarity matrix (1 - adj/max) from a flat vector of MI values.

    The order of mi_vals matches the nested loop over 0<=spec1<spec2<n_specimens.
    """
    adj_matrix = np.zeros((n_specimens, n_specimens), dtype=float)
    idx = 0
    for spec1 in range(n_specimens):
        for spec2 in range(spec1 + 1, n_specimens):
            val = float(mi_vals[idx])
            adj_matrix[spec1, spec2] = val
            adj_matrix[spec2, spec1] = val
            idx += 1
    max_val = float(adj_matrix.max())
    sim = 1.0 - adj_matrix / max_val if max_val > 0 else np.ones_like(adj_matrix)
    return sim


def _r_radius_key(radius: Any) -> str:
    """Format list names like R's as.character() for numeric radii."""
    try:
        value = float(radius)
    except (TypeError, ValueError):
        return str(radius)
    if np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(radius)


def _partitions_from_group_labels(group_labels: np.ndarray) -> list:
    """Return list of index lists for each unique group label (for modularity)."""
    parts = [[int(i) for i, lab in enumerate(group_labels) if lab == ul] for ul in np.unique(group_labels)]
    # Drop empty partitions if any (shouldn't occur)
    return [p for p in parts if len(p) > 0]


def _modularity_for_similarity(similarity_matrix: np.ndarray, partitions: list) -> float:
    """Compute weighted modularity with R igraph's directed adjacency semantics."""
    weights = np.asarray(similarity_matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("similarity_matrix must be a square matrix")

    total_weight = float(weights.sum())
    if total_weight == 0.0 or not np.isfinite(total_weight):
        return 0.0

    out_strength = weights.sum(axis=1)
    in_strength = weights.sum(axis=0)
    modularity = 0.0
    for community in partitions:
        idx = np.asarray(community, dtype=int)
        observed = float(weights[np.ix_(idx, idx)].sum())
        expected = float(out_strength[idx].sum() * in_strength[idx].sum() / total_weight)
        modularity += observed - expected
    return modularity / total_weight


def _bootstrap_null_modularity(
    mi_vals: np.ndarray,
    n_specimens: int,
    partitions: list,
    permutation_indices: np.ndarray,
) -> float:
    """One bootstrap draw: permute MI values, rebuild similarity, compute modularity."""
    shuffled = mi_vals[permutation_indices]
    sim = _build_similarity_from_mi(shuffled, n_specimens)
    return _modularity_for_similarity(sim, partitions)


def measure_transMI(  # noqa: PLR0913, PLR0917 - direct port of the pinned R export
    censuses: list[pd.DataFrame],
    groups: pd.DataFrame,
    depth: int,
    radii: list[float] | str,
    bootstraps: int = 100,
    all: list[str] | None = None,
    alo: list[str] | None = None,
    not_: list[str] | None = None,
    max_bins: int = 100,
    cores: int | None = None,
    random_state: int | np.random.Generator | None = None,
    permutation_indices: Sequence[np.ndarray] | None = None,
    parallel_backend: Literal["serial", "thread", "process"] = "process",
) -> dict[str, pd.DataFrame]:
    """
    Calculate trans mutual information exactly like R SPACE measure_transMI().

    Quantify the difference in patterning for every ensemble, across groups of
    multiple biological specimens. This is statistically compared to null
    expectations in which sample groups are defined at random.

    Args:
        censuses: List of census DataFrames for multiple biological specimens
        groups: DataFrame with grouping factors as columns and specimens as rows
        depth: Maximum number of variables per ensemble
        radii: Radii at which to examine spatial patterns (or "all")
        bootstraps: Number of randomized transMI networks to simulate for P values
        all: Variables which ALL must be included in every max-depth ensemble
        alo: Variables of which AT LEAST ONE must be included in every max-depth ensemble
        not_: Variables which should NOT be included in any max-depth ensemble
        max_bins: Maximum number of bins per variable for rounding
        cores: Number of cores for parallel calculations
        random_state: Seed or local generator used to build Python permutation plans
        permutation_indices: Explicit bootstrap permutations of the pairwise-MI vector
        parallel_backend: Backend for bootstrap modularity calculations

    Returns:
        Dictionary mapping each radius to one R-style DataFrame of transMI results

    Examples:
        >>> # Load multiple census DataFrames
        >>> censuses = [pd.read_csv(f'census_{i}.csv') for i in range(6)]
        >>> groups = pd.DataFrame({'Status': ['A', 'A', 'B', 'B', 'C', 'C']})
        >>> results = measure_transMI(censuses, groups, depth=3, radii=[10, 20])
    """
    if bootstraps < 1:
        raise ValueError("bootstraps must be at least 1")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError(f"Unsupported parallel backend: {parallel_backend}")
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    # R SPACE line 253-254: Validate input parameters
    required_all = all or []
    required_any = alo or []
    excluded = not_ or []

    # R SPACE line 253-254: Check depth constraints
    if (len(required_all) + (1 if required_any else 0)) > depth:
        raise ValueError("Error: the number of required variables exceeds ensemble depth.")

    # R SPACE line 256-259: Handle radii parameter
    if isinstance(radii, str) and radii == "all":
        shared = set(censuses[0]["Radius"].unique())
        for census in censuses[1:]:
            shared &= set(census["Radius"].unique())
        radii = [radius for radius in censuses[0]["Radius"].unique() if radius in shared]

    # R SPACE line 260-262: Check that all censuses have same variables
    all_columns = [set(census.columns) for census in censuses]
    if len(set.union(*all_columns)) != len(set.intersection(*all_columns)):
        raise ValueError("Error: the censuses do not all contain the same variables.")

    # R SPACE line 263: Get variable columns (those with O or S)
    var_cols = [col for col in censuses[0].columns if "O" in col or "S" in col]

    # R SPACE line 264-273: Handle variable exclusions
    cols_to_drop = [False] * len(var_cols)

    if excluded:  # R SPACE line 265-266
        for i, col in enumerate(var_cols):
            if col in excluded:
                cols_to_drop[i] = True

    # R SPACE line 268-270: If all+alo equals depth, drop everything else
    if (len(required_all) + (1 if required_any else 0)) == depth:
        for i, col in enumerate(var_cols):
            if col not in required_all and col not in required_any:
                cols_to_drop[i] = True

    # R SPACE line 271-273: Filter censuses to keep only relevant variables
    final_var_cols = [col for i, col in enumerate(var_cols) if not cols_to_drop[i]]
    filtered_censuses = []
    for census in censuses:
        filtered = census[[*final_var_cols, "Radius"]].copy()
        filtered_censuses.append(filtered)
    censuses = filtered_censuses

    # R SPACE line 291-293: Calculate bins based on minimum sample size across all specimens
    min_sample_sizes = []
    for census in censuses:
        radius_counts = census["Radius"].value_counts()
        min_sample_sizes.append(radius_counts.min())
    ss = min(min_sample_sizes)
    b = min(int(np.floor(((ss + 1) / 10) ** (1 / depth))), max_bins)

    print(f"Based on a minimum sample size of {ss}, there will be {b} bins per variable.")

    # Initialize output structures
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for radius in radii:
        print(f"Beginning radius {radius} pixels.")

        # R SPACE line 297-305: Process all censuses for this radius
        sub_censuses = []
        sub_mins = []
        sub_maxs = []

        for census in censuses:
            sub_census: Any = census[census["Radius"] == radius].copy()
            sub_census = sub_census.drop("Radius", axis=1)

            sub_min = sub_census.min().to_numpy()
            sub_max = sub_census.max().to_numpy()

            # Round into bins using R-compatible nearest grid (return bin values)
            binned_census = sub_census.copy()
            for i, col in enumerate(sub_census.columns):
                binned_census[col] = _r_nearest_round(
                    sub_census[col].to_numpy(), sub_min[i], sub_max[i], b, return_bin_values=True
                )

            sub_censuses.append(binned_census)
            sub_mins.append(sub_min)
            sub_maxs.append(sub_max)

        # R SPACE line 306: Generate all pairs of specimens
        census_combos = [(i, j) for i in range(len(censuses)) for j in range(i + 1, len(censuses))]
        var_names = list(sub_censuses[0].columns)

        depth_results: dict[str, dict[str, Any]] = {}

        # R SPACE line 308-348: For each depth, calculate ensembles and distributions
        for j in range(1, depth + 1):
            print(f"Collating all requested {j}-ensembles.")

            # Generate all combinations of variables at this depth
            ensembles = list(combinations(var_names, j))

            # R SPACE line 312-319: Filter ensembles at max depth
            if j == depth:
                filtered_ensembles = []
                for ensemble in ensembles:
                    # Check 'all' constraint
                    if required_all and required_all != ["none"] and not set(required_all).issubset(ensemble):
                        continue
                    # Check 'alo' constraint
                    if required_any and required_any != ["none"] and not any(var in ensemble for var in required_any):
                        continue
                    filtered_ensembles.append(ensemble)
                ensembles = filtered_ensembles

            # R SPACE line 325: Calculate joint distributions for all ensembles
            joint_dists = np.full((len(ensembles), b**j, len(censuses)), np.nan, dtype=float)

            # Helper to compute smoothed joint dist exactly like R: build_dist -> smooth_dist
            def _joint_for_ensemble(
                idx_p: int,
                ens: tuple[str, ...],
                ensemble_depth: int = j,
                current_sub_censuses: list[pd.DataFrame] = sub_censuses,
                current_sub_mins: list[np.ndarray] = sub_mins,
                current_sub_maxs: list[np.ndarray] = sub_maxs,
            ) -> tuple[int, np.ndarray]:
                out = np.full((b**ensemble_depth, len(current_sub_censuses)), np.nan, dtype=float)
                for k_, sc in enumerate(current_sub_censuses):
                    if sc.empty:
                        continue
                    # Build joint dist on rounded bin VALUES
                    jd = r_build_dist(sc, list(ens), focal_vars="all")
                    # Min/max per var for this specimen
                    col_indices = []
                    for v in ens:
                        loc = sc.columns.get_loc(v)
                        if not isinstance(loc, (int, np.integer)):
                            raise ValueError(f"Expected unique census column for {v!r}")
                        col_indices.append(int(loc))
                    cols = np.asarray(col_indices, dtype=int)
                    mins = current_sub_mins[k_][cols]
                    maxs = current_sub_maxs[k_][cols]
                    min_max = pd.DataFrame([mins, maxs], columns=pd.Index(list(ens)))
                    # Smooth (Chao-Jost) and return only freq vector
                    freqs = r_smooth_dist(jd, b, min_max, full_dist=False)
                    # Ensure length matches b**j; pad/trim if needed
                    if isinstance(freqs, np.ndarray) and freqs.size == (b**ensemble_depth):
                        out[:, k_] = freqs
                    else:
                        actual_size = freqs.size if isinstance(freqs, np.ndarray) else "non-array"
                        raise RuntimeError(
                            f"Smoothed distribution for ensemble {ens} has size {actual_size}; "
                            f"expected {b**ensemble_depth}"
                        )
                return idx_p, out

            # Use threads; numpy releases GIL on heavy ops
            max_workers = int(cores) if cores is not None and int(cores) > 1 else None

            if max_workers and max_workers > 1 and len(ensembles) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_joint_for_ensemble, p, ens) for p, ens in enumerate(ensembles)]
                    for fut in as_completed(futs):
                        p_idx, slab = fut.result()
                        joint_dists[p_idx, :, :] = slab
            else:
                for p_idx, ens in enumerate(ensembles):
                    _, slab = _joint_for_ensemble(p_idx, ens)
                    joint_dists[p_idx, :, :] = slab

            # R SPACE line 349-381: Calculate KL divergence between all pairs
            mi_results = np.zeros((len(ensembles), len(census_combos)))

            def _mi_for_ensemble(
                idx_p: int,
                current_census_combos: list[tuple[int, int]] = census_combos,
                current_joint_dists: np.ndarray = joint_dists,
            ) -> tuple[int, np.ndarray]:
                out = np.zeros(len(current_census_combos), dtype=float)
                for r_idx, (spec1, spec2) in enumerate(current_census_combos):
                    d1 = current_joint_dists[idx_p, :, spec1]
                    d2 = current_joint_dists[idx_p, :, spec2]
                    # Mask invalid bins (NaNs) and compute symmetric KL
                    mask = np.isfinite(d1) & np.isfinite(d2)
                    p = d1[mask]
                    q = d2[mask]
                    # Avoid divisions by zero: only bins with p>0 and q>0 contribute
                    mask1 = (p > 0) & (q > 0)
                    if mask1.any():
                        kl1 = float(np.sum(p[mask1] * np.log2(p[mask1] / q[mask1])))
                        kl2 = float(np.sum(q[mask1] * np.log2(q[mask1] / p[mask1])))
                        out[r_idx] = 0.5 * (kl1 + kl2)
                    else:
                        out[r_idx] = 0.0
                return idx_p, out

            if max_workers and max_workers > 1 and len(ensembles) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_mi_for_ensemble, p) for p in range(len(ensembles))]
                    for fut in as_completed(futs):
                        p_idx, vals = fut.result()
                        mi_results[p_idx, :] = vals
            else:
                for p in range(len(ensembles)):
                    _, vals = _mi_for_ensemble(p)
                    mi_results[p, :] = vals

            # Store results for this depth
            depth_results[str(j)] = {"ensembles": ensembles, "mi_values": mi_results, "census_combos": census_combos}

        out[_r_radius_key(radius)] = depth_results

    # R SPACE line 388-397: Average transMI and format results
    print("Averaging transMI for each pair of images.")

    # R SPACE line 398-443: Test modularity for significance
    print("Testing the modularity of transMI for significance.")

    final_results = {}

    for radius_key, radius_results in out.items():
        rows: list[dict[str, Any]] = []
        pair_cols: list[str] = []

        for depth_key in sorted(radius_results, key=lambda value: int(value)):
            depth_data = radius_results[depth_key]
            ensembles = depth_data["ensembles"]
            mi_values = depth_data["mi_values"]
            census_combos = depth_data["census_combos"]

            for i, ensemble in enumerate(ensembles):
                result_row: dict[str, Any] = {f"V{chr(65 + k)}": var for k, var in enumerate(ensemble)}
                for r, (spec1, spec2) in enumerate(census_combos):
                    col = f"MI_Im{spec1 + 1}_Im{spec2 + 1}"
                    if col not in pair_cols:
                        pair_cols.append(col)
                    result_row[col] = mi_values[i, r]
                rows.append(result_row)

        mutual_info_df = pd.DataFrame(rows)
        vcols = [f"V{chr(65 + i)}" for i in range(depth) if f"V{chr(65 + i)}" in mutual_info_df.columns]
        mutual_info_df = mutual_info_df[vcols + pair_cols]
        result_df = mutual_info_df[vcols].copy()

        for group_col in groups.columns:
            group_labels = groups[group_col].astype("category").cat.codes.values
            partitions = _partitions_from_group_labels(group_labels)
            n_specimens = len(censuses)

            trans_mi_values = []
            z_scores = []
            p_values = []

            for _, row in mutual_info_df.iterrows():
                row = cast(Any, row)
                mi_vals = row[pair_cols].to_numpy(dtype=float)
                similarity_matrix = _build_similarity_from_mi(mi_vals, n_specimens)
                modularity = _modularity_for_similarity(similarity_matrix, partitions)

                if permutation_indices is None:
                    child_seeds = generator.integers(
                        0,
                        np.iinfo(np.uint64).max,
                        size=int(bootstraps),
                        dtype=np.uint64,
                    )
                    permutation_steps = [np.random.default_rng(seed).permutation(len(mi_vals)) for seed in child_seeds]
                    current_indices = np.arange(len(mi_vals))
                    row_permutations = []
                    for permutation in permutation_steps:
                        current_indices = current_indices[permutation]
                        row_permutations.append(current_indices.copy())
                else:
                    permutation_steps = [np.asarray(permutation, dtype=int) for permutation in permutation_indices]
                    if len(permutation_steps) != int(bootstraps):
                        raise ValueError(f"Expected {bootstraps} permutation arrays, got {len(permutation_steps)}")
                    expected_indices = np.arange(len(mi_vals))
                    if any(
                        permutation.shape != expected_indices.shape
                        or not np.array_equal(np.sort(permutation), expected_indices)
                        for permutation in permutation_steps
                    ):
                        raise ValueError("Each permutation must contain every pairwise-MI index exactly once")
                    current_indices = expected_indices
                    row_permutations = []
                    for permutation in permutation_steps:
                        current_indices = current_indices[permutation]
                        row_permutations.append(current_indices.copy())

                worker_count = int(cores) if cores is not None else 1
                can_parallel = worker_count > 1 and int(bootstraps) > 1

                if can_parallel and parallel_backend != "serial":
                    executor_type = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
                    with executor_type(max_workers=worker_count) as executor:
                        null_modularities = list(
                            executor.map(
                                _bootstrap_null_modularity,
                                repeat(mi_vals),
                                repeat(n_specimens),
                                repeat(partitions),
                                row_permutations,
                            )
                        )
                else:
                    null_modularities = [
                        _bootstrap_null_modularity(mi_vals, n_specimens, partitions, permutation)
                        for permutation in row_permutations
                    ]

                null_mean = float(np.mean(null_modularities)) if null_modularities else 0.0
                null_std = float(np.std(null_modularities, ddof=1)) if len(null_modularities) > 1 else 0.0
                # igraph returns exact zero for degenerate nulls; NetworkX can leave
                # floating-point cancellation residue at machine precision.
                if not np.isfinite(null_std) or np.isclose(
                    null_std,
                    0.0,
                    rtol=0.0,
                    atol=10 * np.finfo(float).eps,
                ):
                    null_std = 0.0

                z_score = (modularity - null_mean) / null_std if null_std > 0 else 0.0
                p_value = stats.norm.sf(z_score)

                trans_mi_values.append(modularity)
                z_scores.append(z_score)
                p_values.append(p_value)

            result_df[f"TransMI_{group_col}"] = trans_mi_values
            result_df[f"Zscore_{group_col}"] = z_scores
            result_df[f"Pvalue_{group_col}"] = p_values
            result_df[f"Padjust_{group_col}"] = stats.false_discovery_control(np.array(p_values), method="bh")

        final_results[radius_key] = result_df

    return final_results
