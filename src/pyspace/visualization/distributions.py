"""Observed/random distributions and profile-table views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.distributions import build_dist, r_round_column, smooth_dist
from ..core.patch_summary import random_census


def _radius_value(radius: float | Sequence[float]) -> float:
    return float(radius[0] if isinstance(radius, Sequence) and not isinstance(radius, (str, bytes)) else radius)


def _variable_names(census: pd.DataFrame, ensemble: Sequence[str | int]) -> list[str]:
    names: list[str] = []
    for variable in ensemble:
        if isinstance(variable, int):
            if variable < 1 or variable > len(census.columns):
                raise IndexError(f"ensemble column {variable} is outside census")
            names.append(str(census.columns[variable - 1]))
        else:
            names.append(str(variable))
    missing = [name for name in names if name not in census]
    if missing:
        raise ValueError(f"Unknown ensemble variables: {missing}")
    return names


def _radius_patches(patch_list: Mapping[str, Mapping[str, pd.DataFrame]], radius: float) -> Mapping[str, pd.DataFrame]:
    matches = [patches for key, patches in patch_list.items() if float(key) == radius]
    if len(matches) != 1:
        raise ValueError(f"Exactly one patch list must match radius {radius}")
    return matches[0]


def plot_dist(
    census: pd.DataFrame,
    ensemble: Sequence[str | int],
    radius: float | Sequence[float],
    bin_num: int | None = None,
    patch_list: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
    plot_zoom: bool = False,
    plot_bkgd: str = "W",
    *,
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, Figure | None]:
    """Return and plot a one- or two-variable SPACE distribution."""
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")
    radius_number = _radius_value(radius)
    variables = _variable_names(census, ensemble)
    variable_columns = [column for column in census if "O" in str(column) or "S" in str(column)]
    working = census.loc[census["Radius"].astype(float) == radius_number, variable_columns].copy()
    if patch_list is not None:
        generator = (
            random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        )
        working = random_census(dict(_radius_patches(patch_list, radius_number)), None, rng=generator)
    if len(variables) > 2:
        return working, None
    if not variables:
        raise ValueError("ensemble cannot be empty")

    minima = working[variables].min()
    maxima = working[variables].max()
    if bin_num is not None:
        if bin_num < 1:
            raise ValueError("bin_num must be positive")
        adjusted_bins: list[int] = []
        for variable in variables:
            unique_count = working[variable].nunique(dropna=False)
            if unique_count > bin_num:
                working[variable] = r_round_column(
                    working[variable].to_numpy(dtype=float),
                    float(minima[variable]),
                    float(maxima[variable]),
                    bin_num,
                    return_bin_values=True,
                )
                adjusted_bins.append(bin_num)
            else:
                adjusted_bins.append(unique_count)
        distribution = build_dist(working, variables, "all")
        min_max = pd.DataFrame([minima[variables], maxima[variables]])
        smoothed = smooth_dist(distribution, adjusted_bins, min_max, full_dist=True)
        if not isinstance(smoothed, pd.DataFrame):
            raise TypeError("smooth_dist did not return its documented table")
        result = smoothed
    else:
        result = working[variables].reset_index(drop=True)

    background, foreground = ("black", "white") if plot_bkgd == "B" else ("white", "black")
    figure, axes = plt.subplots()
    figure.patch.set_facecolor(background)
    axes.set_facecolor(background)
    axes.tick_params(colors=foreground)
    if len(variables) == 1:
        if bin_num is None:
            axes.hist(result[variables[0]], bins="auto", density=True, histtype="step", color=foreground)
        else:
            axes.bar(result[variables[0]], result["freq"], color="gray", edgecolor=foreground)
        axes.set(xlabel=variables[0], ylabel="Probability" if bin_num is None else "Frequency")
    elif bin_num is None:
        axes.scatter(result[variables[0]], result[variables[1]], color=foreground, alpha=0.25)
        axes.set(xlabel=variables[0], ylabel=variables[1])
    else:
        plotted = axes.scatter(
            result[variables[0]],
            result[variables[1]],
            c=np.log10(result["freq"]),
            cmap="gray_r" if plot_bkgd == "W" else "gray",
            marker="s",
        )
        axes.set(xlabel=variables[0], ylabel=variables[1])
        figure.colorbar(plotted, ax=axes, label="Log Prob")
    if not plot_zoom:
        axes.set_xlim(0, 100)
        if len(variables) == 2:
            axes.set_ylim(0, 100)
    figure.tight_layout()
    return result, figure


def _normalize_profiles(profile: pd.DataFrame, compare: str, normalize: str) -> pd.DataFrame:
    if compare not in {"A", "W", "B"} or normalize not in {"U", "Z"}:
        raise ValueError("compare must be A/W/B and normalize must be U/Z")
    if not {"Object", "Count"}.issubset(profile.columns) or profile.shape[1] < 3:
        raise ValueError("prof_table requires Object, Count, and component columns")
    components = profile.drop(columns="Count").copy()
    values = components.iloc[:, 1:].to_numpy(dtype=float)
    axis = 0 if compare == "A" else 1 if compare == "W" else None
    if normalize == "U":
        minimum = np.min(values, axis=axis, keepdims=axis is not None)
        shifted = values - minimum
        maximum = np.max(shifted, axis=axis, keepdims=axis is not None)
        fill = 0.5 if compare == "W" else 0.0
        values = np.divide(shifted, maximum, out=np.full_like(shifted, fill), where=maximum != 0)
    else:
        mean = np.mean(values, axis=axis, keepdims=axis is not None)
        deviation = np.std(values, axis=axis, ddof=1, keepdims=axis is not None)
        values = np.divide(values - mean, deviation, out=np.zeros_like(values), where=deviation != 0)
    components.iloc[:, 1:] = values
    return components


def plot_table(
    prof_table: pd.DataFrame,
    compare: str = "A",
    normalize: str = "U",
    tile_plots: bool = False,
    plot_bkgd: str = "W",
) -> tuple[pd.DataFrame, Figure]:
    """Normalize and render a SPACE profile table."""
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")
    normalized = _normalize_profiles(prof_table, compare, normalize)
    background, foreground = ("black", "white") if plot_bkgd == "B" else ("white", "black")
    figure, axes = plt.subplots(2 if tile_plots else 1, 1, squeeze=False)
    heatmap_axis = axes[-1, 0]
    values = normalized.iloc[:, 1:].to_numpy(dtype=float).T
    image = heatmap_axis.imshow(values, aspect="auto", cmap="viridis")
    heatmap_axis.set_xticks(np.arange(len(normalized)), normalized["Object"].astype(str))
    heatmap_axis.set_yticks(np.arange(values.shape[0]), normalized.columns[1:])
    heatmap_axis.set(xlabel="Object", ylabel="Component")
    if tile_plots:
        axes[0, 0].bar(prof_table["Object"], prof_table["Count"], color=foreground)
        axes[0, 0].set_ylabel("Count")
    for axis in axes.ravel():
        axis.set_facecolor(background)
        axis.tick_params(colors=foreground)
        axis.xaxis.label.set_color(foreground)
        axis.yaxis.label.set_color(foreground)
    figure.patch.set_facecolor(background)
    figure.colorbar(image, ax=heatmap_axis, label="Amount")
    figure.tight_layout()
    return normalized, figure


__all__ = ["plot_dist", "plot_table"]
