"""Distribution primitives ported from the pinned SPACE ``utils.R``."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def r_round_column(
    focal_col: np.ndarray,
    col_min: float,
    col_max: float,
    bin_num: int,
    return_bin_values: bool = False,
) -> np.ndarray:
    """Round values to SPACE's nearest evenly spaced bin or one-based bin ID."""
    values = np.asarray(focal_col)
    if np.isnan(col_min) or np.isnan(col_max) or bin_num < 2 or col_min == col_max:
        return values.copy()
    indices = np.rint((values - col_min) * (bin_num - 1) / (col_max - col_min))
    indices = np.clip(indices, 0, bin_num - 1).astype(int)
    return np.linspace(col_min, col_max, bin_num)[indices] if return_bin_values else indices + 1


def total_comp_bins(
    dimension: int,
    bins_per_var: int,
    min_per_var: Iterable[float] | None = None,
    max_per_var: Iterable[float] | None = None,
) -> int:
    """Count valid joint bins for variables from one compositional object map."""
    if dimension <= 0 or bins_per_var <= 0:
        raise ValueError("dimension and bins_per_var must be positive")

    def unrestricted() -> int:
        if dimension == 1:
            return bins_per_var
        if dimension == 2:
            return sum(range(1, bins_per_var + 1))
        return math.comb(bins_per_var + dimension - 1, dimension - 1)

    if min_per_var is None and max_per_var is None:
        return unrestricted()
    if min_per_var is None or max_per_var is None:
        raise ValueError("min_per_var and max_per_var must be provided together")
    minima, maxima = list(min_per_var), list(max_per_var)
    if len(minima) != len(maxima) or len(minima) < dimension:
        raise ValueError("min_per_var and max_per_var must have the same length and cover dimension")
    if all(value == 0 for value in minima) and all(value == 100 for value in maxima):
        return unrestricted()
    maximum_valid = 1
    for indices in itertools.combinations(range(len(minima)), dimension):
        axes = [np.linspace(minima[index], maxima[index], bins_per_var) for index in indices]
        valid = sum(np.sum(combination) <= 100 + 1e-9 for combination in itertools.product(*axes))
        maximum_valid = max(maximum_valid, valid)
    return int(maximum_valid)


def build_dist(census: pd.DataFrame, vars: list[str], focal_vars: str | list[str] = "all") -> pd.DataFrame:
    """Count observed rounded combinations in R grouped-column order."""
    if not isinstance(census, pd.DataFrame):
        raise TypeError("census must be a DataFrame")
    if not vars:
        raise ValueError("vars cannot be empty")
    if focal_vars != "all" and not any(variable in focal_vars for variable in vars if pd.notna(variable)):
        return pd.DataFrame()
    columns = [column for column in census.columns if column in vars]
    if not columns:
        return pd.DataFrame()
    return census[columns].groupby(columns, dropna=False, sort=True).size().reset_index(name="freq")


def _bin_axes(
    variable_names: list[str],
    bin_num: int | list[int],
    min_max: pd.DataFrame,
) -> list[np.ndarray]:
    counts = [bin_num] * len(variable_names) if isinstance(bin_num, int) else list(bin_num)
    if len(counts) != len(variable_names) or any(count < 1 for count in counts):
        raise ValueError("bin_num must provide at least one bin per variable")
    if min_max.shape != (2, len(variable_names)):
        raise ValueError("min_max must have two rows and one column per variable")
    return [
        np.linspace(float(min_max.iloc[0, index]), float(min_max.iloc[1, index]), counts[index])
        for index in range(len(variable_names))
    ]


def _expand_grid(variable_names: list[str], axes: list[np.ndarray]) -> pd.DataFrame:
    if len(variable_names) == 1:
        return pd.DataFrame({variable_names[0]: axes[0]})
    combinations = [tuple(reversed(values)) for values in itertools.product(*reversed(axes))]
    return pd.DataFrame(combinations, columns=pd.Index(variable_names))


def _mask_invalid_compositions(distribution: pd.DataFrame, variable_names: list[str]) -> None:
    object_groups = {
        variable.split(".")[0] for variable in variable_names if variable.startswith("O") and "." in variable
    }
    for group in object_groups:
        columns = [variable for variable in variable_names if variable.split(".")[0] == group]
        if len(columns) > 1:
            distribution.loc[distribution[columns].sum(axis=1) > 100, "freq"] = np.nan


def smooth_dist(
    joint_dist: pd.DataFrame,
    bin_num: int | list[int],
    min_max: pd.DataFrame,
    full_dist: bool = True,
) -> pd.DataFrame | np.ndarray:
    """Apply SPACE's Chao-Jost smoothing over the complete R-order grid."""
    if not isinstance(joint_dist, pd.DataFrame) or "freq" not in joint_dist:
        raise ValueError("joint_dist must be a DataFrame with a freq column")
    if joint_dist.empty:
        return pd.DataFrame() if full_dist else np.array([], dtype=float)
    variable_names = [str(column) for column in joint_dist.columns if column != "freq"]
    if not variable_names:
        raise ValueError("joint_dist must contain at least one variable column")
    observation_count = float(joint_dist["freq"].sum() + 1)
    singletons = float(joint_dist.loc[joint_dist["freq"] == 1, "freq"].sum()) or 0.5
    doubletons = float(joint_dist.loc[joint_dist["freq"] == 2, "freq"].sum())
    missing_probability = singletons / observation_count
    f1, f2 = (observation_count - 1) * singletons, 2 * doubletons
    if f1 or f2:
        missing_probability *= f1 / (f1 + f2)
    complete = _expand_grid(variable_names, _bin_axes(variable_names, bin_num, min_max))
    complete = complete.merge(joint_dist, on=variable_names, how="left")
    complete["freq"] = complete["freq"].fillna(0)
    _mask_invalid_compositions(complete, variable_names)
    valid = complete["freq"].dropna()
    zero_count = int((complete["freq"] == 0).sum())
    if zero_count == 0:
        complete["freq"] = valid / valid.sum()
    else:
        total = float(valid.sum())
        positive = complete["freq"] > 0
        complete.loc[positive, "freq"] = complete.loc[positive, "freq"] / total * (1 - missing_probability)
        complete.loc[complete["freq"] == 0, "freq"] = missing_probability / zero_count
    return complete if full_dist else complete["freq"].to_numpy()


__all__ = ["build_dist", "r_round_column", "smooth_dist", "total_comp_bins"]
