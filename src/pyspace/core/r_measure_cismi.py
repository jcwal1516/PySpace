"""
R SPACE compatible measure_cisMI implementation.

This module provides a 1:1 implementation of R SPACE's measure_cisMI function,
following the exact algorithm from the R package.
"""

import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, repeat
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

# R-compatible helpers: nearest-bin rounding and compositional bin counts
from .distributions import r_round_column as _r_nearest_round
from .patch_summary import random_census as _random_census


def _r_radius_key(radius: Any) -> str:
    """Format list names like R's as.character() for numeric radii."""
    try:
        value = float(radius)
    except (TypeError, ValueError):
        return str(radius)
    if np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(radius)


def _entropy_for_ensemble_worker(
    idx_ens: int,
    ens: tuple[str, ...],
    var_to_idx: dict[str, int],
    binned_np: list[np.ndarray],
    b: int,
) -> tuple[int, np.ndarray]:
    """Module-level worker for parallel entropy computation (must be picklable)."""
    cols_idx = [var_to_idx[v] for v in ens]
    out = np.zeros(len(binned_np), dtype=float)
    for ci, arr in enumerate(binned_np):
        if arr.size == 0:
            out[ci] = 0.0
            continue
        subset = arr[:, cols_idx]
        out[ci] = _r_entropy(subset, b)
    return idx_ens, out


def _align_random_census(rc: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    if "Nbhd" in rc.columns:
        rc = rc.drop(columns=["Nbhd"])
    for v in variables:
        if v not in rc.columns:
            rc[v] = 0.0
    return rc[variables]


def _round_census_array(
    census_df: pd.DataFrame,
    variables: list[str],
    global_mins: np.ndarray,
    global_maxs: np.ndarray,
    bins: int,
) -> np.ndarray:
    rounded = np.empty((len(census_df), len(variables)), dtype=int)
    for i, col in enumerate(variables):
        rounded[:, i] = _r_round_column(census_df[col].to_numpy(), global_mins[i], global_maxs[i], bins)
    return rounded


def measure_cisMI(  # noqa: PLR0913, PLR0917 - direct port of the pinned R export
    census: pd.DataFrame,
    patch_list: dict[str, Any] | None,
    depth: int,
    radii: list[float] | None = None,
    bootstraps: int = 100,
    all: list[str] | None = None,
    alo: list[str] | None = None,
    not_: list[str] | None = None,
    max_bins: int = 100,
    cores: int | None = None,
    allow_permutation_fallback: bool = False,
    random_state: int | np.random.Generator | None = None,
    random_censuses: Mapping[str | float, Sequence[pd.DataFrame]] | Sequence[pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate cis mutual information exactly like R SPACE measure_cisMI().

    This is a 1:1 implementation of the R SPACE algorithm from lines 20-182
    of measure_mutual_info.R.

    Args:
        census: Census DataFrame from biological specimen
        patch_list: Patch list from biological specimen
        depth: Maximum number of variables per ensemble
        radii: Radii to examine (None = all radii in census)
        bootstraps: Number of randomized censuses (default 100)
        all: Variables which ALL must be included in every max-depth ensemble
        alo: Variables of which AT LEAST ONE must be included in every max-depth ensemble
        not_: Variables which should NOT be included in any max-depth ensemble
        max_bins: Maximum number of bins per variable for rounding (default 100)
        cores: Number of cores for parallel calculations
        allow_permutation_fallback: Permit simple permutations when patch_list is missing
        random_state: Seed or local generator used only to construct Python random plans
        random_censuses: Explicit bootstrap censuses, optionally keyed by radius. This is
            the cross-language parity boundary: supply the same frames to R and Python.

    Returns:
        Dictionary mapping radius (as string) to DataFrame of cisMI results

    Examples:
        >>> # Load census and patch list from R SPACE tutorial
        >>> census_df = pd.read_csv('census_images.csv')
        >>> results = measure_cisMI(census_df, patch_list, depth=3, bootstraps=100)
        >>> print(results['10'].head())  # Results for 10 micron radius
    """
    # R SPACE line 22-23: Validate input parameters
    required_all = all or []
    required_any = alo or []
    excluded = not_ or []

    # R SPACE line 22-24: Check depth constraints
    if (len(required_all) + (1 if required_any else 0)) > depth:
        raise ValueError("Error: the number of required variables exceeds ensemble depth.")

    if bootstraps < 1:
        raise ValueError("bootstraps must be at least 1")
    worker_count = 1 if cores is None else int(cores)
    if worker_count <= 0:
        raise ValueError("cores must be a positive integer or None")
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    census = cast(Any, census)

    # R SPACE line 25-27: Handle radii parameter
    if radii is None:
        radii = list(census["Radius"].unique())

    # R SPACE line 28: Filter census to relevant radii
    census = cast(Any, census[census["Radius"].isin(radii)].copy())

    # R SPACE line 29: Get variable columns (those with O or S)
    var_cols = [col for col in census.columns if "O" in col or "S" in col]

    # R SPACE line 30-36: Handle variable exclusions
    cols_to_drop = [False] * len(var_cols)

    if excluded:  # R SPACE line 31-33
        for i, col in enumerate(var_cols):
            if col in excluded:
                cols_to_drop[i] = True

    # R SPACE line 34-36: If all+alo equals depth, drop everything else
    if (len(required_all) + (1 if required_any else 0)) == depth:
        for i, col in enumerate(var_cols):
            if col not in required_all and col not in required_any:
                cols_to_drop[i] = True

    # R SPACE line 37: Drop coordinate columns but keep radius
    # census[ , !c(cols_to_drop,T,T,T,F)]
    final_var_cols = [col for i, col in enumerate(var_cols) if not cols_to_drop[i]]
    census = cast(Any, census[[*final_var_cols, "Radius"]].copy())

    # R SPACE line 38-56: Infer object-scalar pairs (OS_pairs)
    os_pairs_cols = [col for col in final_var_cols if col.count(".") >= 2 or "_" in col]
    OS_pairs = {}

    if os_pairs_cols:
        # Extract object map names from OS pairs
        obj_maps = list({col.split(".")[0] for col in os_pairs_cols if "_" in col})

        for obj_map in obj_maps:
            # Find OS pairs for this object map
            sub_os_pairs = [col for col in os_pairs_cols if col.startswith(obj_map)]

            if sub_os_pairs:
                # Parse scalar and object IDs
                scalars = list({col.split("_")[1] for col in sub_os_pairs if "_" in col})
                objects = list({col.split("_")[0] for col in sub_os_pairs if "_" in col})

                # Create pairing matrix
                matrix = np.zeros((len(objects), len(scalars)))
                for _j, pair in enumerate(sub_os_pairs):
                    if "_" in pair:
                        obj_part, scalar_part = pair.split("_")
                        obj_idx = objects.index(obj_part)
                        scalar_idx = scalars.index(scalar_part)
                        matrix[obj_idx, scalar_idx] = 1

                # Store semantic labels for downstream patch summarization.
                OS_pairs[obj_map] = pd.DataFrame(matrix, index=pd.Index(objects), columns=pd.Index(scalars))

    # R SPACE line 57-59: Calculate bins based on minimum sample size
    ss = census.groupby("Radius").size().min()
    b = min(int(np.floor(((ss + 1) / 10) ** (1 / depth))), max_bins)

    print(f"Based on a minimum sample size of {ss}, there will be {b} bins per variable.")

    # Initialize output structures for each radius
    out = {}

    for radius in radii:
        print(f"Beginning radius {radius} microns.")

        # R SPACE line 70-71: Get data for this radius
        sub_census = census[census["Radius"] == radius].copy()
        sub_census = sub_census.drop("Radius", axis=1)
        variables = list(sub_census.columns)

        # R SPACE line 72-84: Generate bootstrap random censuses (patch-based if patch_list provided)
        print(f"Generating {bootstraps} bootstrap random censuses.")

        sub_pls: dict[str, pd.DataFrame] | None = None
        if patch_list is not None and isinstance(patch_list, dict) and len(patch_list) > 0:
            # Build per-radius patch list. R SPACE stores patch lists as
            # radius -> object/scalar map -> patches, while older Python callers
            # often pass a flat object/scalar map -> patches dictionary.
            radius_key = _r_radius_key(radius)
            radius_patch_list: Any = patch_list.get(radius_key)
            if not isinstance(radius_patch_list, dict):
                for key, value in patch_list.items():
                    try:
                        same_radius = float(key) == float(radius)
                    except (TypeError, ValueError):
                        same_radius = str(key) == radius_key
                    if same_radius and isinstance(value, dict):
                        radius_patch_list = value
                        break
            source_patch_list = radius_patch_list if isinstance(radius_patch_list, dict) else patch_list

            sub_pls = {}
            for gname, gdf in source_patch_list.items():
                if isinstance(gdf, pd.DataFrame) and "Nbhd" in gdf.columns:
                    if "Radius" in gdf.columns:
                        sdf = gdf[gdf["Radius"] == radius].copy()
                        if not sdf.empty:
                            sdf = sdf.drop(columns=["Radius"])
                            sub_pls[gname] = cast(pd.DataFrame, sdf)
                    else:
                        sub_pls[gname] = gdf.copy()

        radius_plan: list[pd.DataFrame] | None = None
        if random_censuses is not None:
            if isinstance(random_censuses, Mapping):
                candidates = (
                    random_censuses.get(radius),
                    random_censuses.get(_r_radius_key(radius)),
                )
                selected = next((candidate for candidate in candidates if candidate is not None), None)
                if selected is None:
                    raise ValueError(f"No explicit random census plan for radius {radius}")
                radius_plan = [frame.copy() for frame in selected]
            else:
                radius_plan = [frame.copy() for frame in random_censuses]
            if len(radius_plan) != bootstraps:
                raise ValueError(f"Expected {bootstraps} random censuses for radius {radius}, got {len(radius_plan)}")

        use_patch_randomization = sub_pls is not None
        if radius_plan is None and not use_patch_randomization:
            if not allow_permutation_fallback:
                raise ValueError(
                    "measure_cisMI requires a SPACE-style patch_list to build randomized nulls. "
                    "Provide patch_list or set allow_permutation_fallback=True to use simple permutations."
                )
            warnings.warn(
                "patch_list not provided; falling back to within-column permutations. "
                "Null variance may be under-estimated relative to SPACE R.",
                RuntimeWarning,
                stacklevel=2,
            )

        bootstrap_seeds = generator.integers(0, np.iinfo(np.uint64).max, size=bootstraps, dtype=np.uint64)

        def make_random_census(
            bootstrap_index: int,
            *,
            use_patch_randomization: bool = use_patch_randomization,
            sub_pls: dict[str, pd.DataFrame] | None = sub_pls,
            sub_census: pd.DataFrame = sub_census,
            variables: list[str] = variables,
            radius_plan: list[pd.DataFrame] | None = radius_plan,
            bootstrap_seeds: np.ndarray = bootstrap_seeds,
        ) -> pd.DataFrame:
            if radius_plan is not None:
                return _align_random_census(radius_plan[bootstrap_index].copy(), variables)
            bootstrap_rng = np.random.default_rng(bootstrap_seeds[bootstrap_index])
            if use_patch_randomization:
                assert sub_pls is not None
                return _align_random_census(_random_census(sub_pls, osp=OS_pairs, rng=bootstrap_rng), variables)
            random_census_df = sub_census.copy()
            for col in variables:
                random_census_df[col] = bootstrap_rng.permutation(np.asarray(sub_census[col]))
            return random_census_df

        # R SPACE line 87-88: Calculate min/max for each variable across all censuses
        global_mins = sub_census[variables].min().to_numpy(dtype=float)
        global_maxs = sub_census[variables].max().to_numpy(dtype=float)
        random_frames: list[pd.DataFrame] = []
        for bootstrap_idx in range(bootstraps):
            rc = make_random_census(bootstrap_idx)
            random_frames.append(rc)
            global_mins = np.minimum(global_mins, rc[variables].min().to_numpy(dtype=float))
            global_maxs = np.maximum(global_maxs, rc[variables].max().to_numpy(dtype=float))
            if bootstraps >= 1000 and (bootstrap_idx + 1) % 500 == 0:
                print(f"  generated {bootstrap_idx + 1}/{bootstraps} bootstrap ranges")

        # R SPACE line 89-91: Round each variable into bins
        print(f"Rounding censuses into {b} bins per variable.")

        # R SPACE line 92-105: For each depth, calculate MI
        depth_results: list[dict[str, Any]] = []
        var_to_idx = {v: i for i, v in enumerate(variables)}
        observed_binned = _round_census_array(sub_census, variables, global_mins, global_maxs, b)
        binned_censuses = [
            observed_binned,
            *[_round_census_array(frame, variables, global_mins, global_maxs, b) for frame in random_frames],
        ]

        for j in range(1, depth + 1):
            print(f"Collating all requested {j}-ensembles.")

            # R SPACE line 95: Generate all combinations of variables at this depth
            var_names = variables
            ensembles = list(combinations(var_names, j))

            # R SPACE line 97-104: Filter ensembles at max depth
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

            # R SPACE line 106-125: Calculate entropy for all ensembles
            print(f"Calculating entropy for all {j}-ensembles.")
            ensemble_entropies = np.zeros((len(ensembles), bootstraps + 1))
            if worker_count > 1 and len(ensembles) > 1:
                with ThreadPoolExecutor(max_workers=min(worker_count, len(ensembles))) as executor:
                    entropy_rows = executor.map(
                        _entropy_for_ensemble_worker,
                        range(len(ensembles)),
                        ensembles,
                        repeat(var_to_idx),
                        repeat(binned_censuses),
                        repeat(b),
                    )
                    for ensemble_index, entropy_values in entropy_rows:
                        ensemble_entropies[ensemble_index] = entropy_values
            else:
                for ensemble_index, ensemble in enumerate(ensembles):
                    _, entropy_values = _entropy_for_ensemble_worker(
                        ensemble_index,
                        ensemble,
                        var_to_idx,
                        binned_censuses,
                        b,
                    )
                    ensemble_entropies[ensemble_index] = entropy_values

            # Store results for this depth
            depth_results.append({"depth": j, "ensembles": ensembles, "entropies": ensemble_entropies})

        n_obs = max(1, len(sub_census))
        for result in depth_results:
            if result["depth"] == 1:
                result["entropies"] += b / (1.5 * n_obs)

        # R SPACE line 156-166: Calculate mutual information using inclusion-exclusion
        print("Combining entropies to calculate mutual information.")

        if depth > 1:
            for result in depth_results:
                if result["depth"] > 1:
                    j = result["depth"]
                    mi_values = np.zeros_like(result["entropies"])

                    for k, ensemble in enumerate(result["ensembles"]):
                        for p in range(1, j + 1):
                            # Find all sub-ensembles of size p
                            sub_ensembles = list(combinations(ensemble, p))

                            for sub_ensemble in sub_ensembles:
                                # Find the row for this sub-ensemble in the appropriate depth result
                                for depth_result in depth_results:
                                    if depth_result["depth"] == p:
                                        try:
                                            sub_idx = depth_result["ensembles"].index(sub_ensemble)
                                            sign = (-1) ** (p - 1)
                                            mi_values[k, :] += sign * depth_result["entropies"][sub_idx, :]
                                        except ValueError as error:
                                            raise RuntimeError(
                                                f"Missing depth-{p} entropy for sub-ensemble {sub_ensemble}"
                                            ) from error

                    result["mi_values"] = mi_values
                else:
                    result["mi_values"] = result["entropies"].copy()
        else:
            depth_results[0]["mi_values"] = depth_results[0]["entropies"].copy()

        # R SPACE line 167-174: Calculate statistics
        print("Testing mutual information for significance.")

        final_results: list[dict[str, Any]] = []
        for result in depth_results:
            ensembles = result["ensembles"]
            mi_values = result["mi_values"]

            # Calculate CisMI = true MI - mean(random MI)
            cis_mi = mi_values[:, 0] - np.mean(mi_values[:, 1:], axis=1)

            # Calculate Z-scores using R's sample standard deviation.
            std_devs = np.std(mi_values[:, 1:], axis=1, ddof=1)
            std_devs = np.where(np.isfinite(std_devs), std_devs, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                z_scores = cis_mi / std_devs
            z_scores = np.where(np.isnan(z_scores), 0.0, z_scores)

            # Calculate p-values (two-sided)
            p_values = 2 * stats.norm.sf(np.abs(z_scores))

            # Create DataFrame for this depth
            for i, ensemble in enumerate(ensembles):
                ensemble_dict = {f"V{chr(65 + j)}": var for j, var in enumerate(ensemble)}

                ensemble_dict.update({"CisMI": cis_mi[i], "Zscore": z_scores[i], "Pvalue": p_values[i]})
                final_results.append(ensemble_dict)

        # R SPACE line 175-178: Combine results and apply BH correction
        result_df = pd.DataFrame(final_results)

        # Apply Benjamini-Hochberg correction
        result_df["Padjust"] = stats.false_discovery_control(result_df["Pvalue"].to_numpy(), method="bh")

        # R SPACE line 176-177: Reorder columns (variables first, then statistics)
        var_columns = [col for col in result_df.columns if col.startswith("V")]
        stat_columns = [col for col in result_df.columns if not col.startswith("V")]
        result_df = result_df[var_columns + stat_columns]

        out[_r_radius_key(radius)] = result_df

    return out


def _r_round_column(values: np.ndarray, col_min: float, col_max: float, bins: int) -> np.ndarray:
    """Nearest-grid rounding to 1-based bin IDs (R-compatible)."""
    if col_max == col_min:
        return np.asarray(values).copy()
    return _r_nearest_round(values, col_min, col_max, bins, return_bin_values=False)


def _r_entropy(ensemble_data: np.ndarray, b: int) -> float:
    """
    Calculate entropy exactly like R SPACE entropy() function.

    Args:
        ensemble_data: 2D array (n_samples x n_vars) of bin indices
        b: Number of bins per variable

    Returns:
        Shannon entropy (bits)
    """
    if ensemble_data.shape[1] == 0:
        return 0.0

    # Convert multi-dimensional bin indices to single joint index
    n_vars = ensemble_data.shape[1]
    powers = b ** np.arange(n_vars)

    # Convert to 0-based for calculation
    joint_indices = ((ensemble_data - 1) * powers).sum(axis=1)

    # Count frequencies
    _unique_indices, counts = np.unique(joint_indices, return_counts=True)
    probabilities = counts / counts.sum()

    # Calculate Shannon entropy
    entropy_val = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy_val)
