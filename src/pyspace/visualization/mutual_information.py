"""R-compatible mutual-information selection and plotting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _variable_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if str(column).count("V") == 1]


def _one_column(candidates: list[str], purpose: str) -> str:
    if len(candidates) != 1:
        raise ValueError(f"Exactly one {purpose} column is required; found {candidates}")
    return candidates[0]


def _columns(frame: pd.DataFrame, group: str | None, adjusted: bool) -> tuple[list[str], str, str, str]:
    variables = _variable_columns(frame)
    if group is None:
        metric = _one_column([str(column) for column in frame if str(column).count("MI") == 1], "MI")
        prefix = "Padjust" if adjusted else "Pvalue"
        probability = _one_column(
            [str(column) for column in frame if str(column).count(prefix) == 1],
            prefix,
        )
        zscore = _one_column([str(column) for column in frame if str(column).count("Zscore") == 1], "Zscore")
    else:
        metric = _one_column([f"TransMI_{group}"] if f"TransMI_{group}" in frame else [], "group MI")
        probability_name = f"{'Padjust' if adjusted else 'Pvalue'}_{group}"
        probability = _one_column([probability_name] if probability_name in frame else [], "group probability")
        zscore = _one_column([f"Zscore_{group}"] if f"Zscore_{group}" in frame else [], "group Zscore")
    return variables, metric, probability, zscore


def _radius_frame(mi: Mapping[str, pd.DataFrame], radius: float | str) -> pd.DataFrame:
    requested = float(radius)
    matches = [frame for key, frame in mi.items() if float(key) == requested]
    if len(matches) != 1:
        raise ValueError(f"Exactly one MI table must match radius {radius}")
    return matches[0].copy()


def _ensemble_row(frame: pd.DataFrame, variable_columns: list[str], ensemble: Sequence[str]) -> int:
    requested = set(map(str, ensemble))
    matches: list[int] = []
    for row_index, row in frame[variable_columns].iterrows():
        observed = {str(value) for value in row.dropna()}
        if requested.issubset(observed):
            matches.append(int(row_index))
    if len(matches) > 1:
        matches = [
            index for index in matches if int(frame.loc[index, variable_columns].notna().sum()) == len(requested)
        ]
    if len(matches) != 1:
        raise ValueError(f"Exactly one ensemble row must match {list(ensemble)}")
    return matches[0]


def _estimated_log_probability(frame: pd.DataFrame, probability: str, zscore: str, row_index: int) -> float:
    log_probability = np.log10(frame[probability].astype(float).to_numpy())
    target_position = frame.index.get_loc(row_index)
    if np.isfinite(log_probability[target_position]):
        return float(log_probability[target_position])
    finite = np.isfinite(log_probability)
    x = np.abs(frame[zscore].astype(float).to_numpy()[finite])
    if len(x) < 3:
        raise ValueError("At least three finite probabilities are required to estimate an exact zero P value")
    design = np.column_stack([np.ones(len(x)), x, x**2])
    coefficients = np.linalg.lstsq(design, log_probability[finite], rcond=None)[0]
    target = abs(float(frame.loc[row_index, zscore]))
    return float(coefficients @ np.array([1.0, target, target**2]))


def _background(figure: Figure, axes: Any, plot_bkgd: str) -> str:
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")
    color = "white" if plot_bkgd == "B" else "black"
    if plot_bkgd == "B":
        figure.patch.set_facecolor("black")
        axes.set_facecolor("black")
        axes.tick_params(colors="white")
        axes.xaxis.label.set_color("white")
        axes.yaxis.label.set_color("white")
        axes.title.set_color("white")
    return color


def plot_MI_radius(
    mi: Mapping[str, pd.DataFrame],
    ensemble: Sequence[str],
    p_thr: float | None = None,
    p_adj: bool = False,
    group: str | None = None,
    plot_bkgd: str = "W",
) -> tuple[pd.DataFrame, Figure]:
    """Select one ensemble across radii and plot its log P value."""
    threshold = 0.05 if p_thr is None else float(p_thr)
    if not 0 < threshold <= 1:
        raise ValueError("p_thr must be in (0, 1]")
    rows: list[dict[str, float]] = []
    metric_name: str | None = None
    for radius_key, frame in mi.items():
        variables, metric, probability, zscore = _columns(frame, group, p_adj)
        metric_name = metric_name or metric
        if metric != metric_name:
            raise ValueError("All radius tables must use the same MI metric column")
        row_index = _ensemble_row(frame, variables, ensemble)
        rows.append(
            {
                "radius": float(radius_key),
                metric: float(frame.loc[row_index, metric]),
                "Padjust": _estimated_log_probability(frame, probability, zscore, row_index),
            }
        )
    if not rows or metric_name is None:
        raise ValueError("mi cannot be empty")
    result = pd.DataFrame(rows).sort_values("radius", kind="stable").reset_index(drop=True)
    magnitudes = result[metric_name].abs()
    spread = float(magnitudes.max() - magnitudes.min())
    result["size"] = 1.0 if len(result) == 1 else (magnitudes - magnitudes.min()) / spread + 0.5 if spread else 0.5

    figure, axes = plt.subplots()
    color = _background(figure, axes, plot_bkgd)
    axes.axhline(np.log10(threshold), linestyle=":", color=color)
    axes.plot(result["radius"], result["Padjust"], color=color, linewidth=0.5)
    axes.scatter(result["radius"], result["Padjust"], s=50 * result["size"], color=color)
    axes.set(xlabel="Length Scale (um)", ylabel="Corrected Log P Value")
    axes.invert_yaxis()
    figure.tight_layout()
    return result, figure


def _required_filter(
    frame: pd.DataFrame,
    variable_columns: list[str],
    required_all: Sequence[str] | None,
    required_any: Sequence[str] | None,
    excluded: Sequence[str] | None,
) -> pd.Series:
    def keep(row: pd.Series) -> bool:
        observed = {str(value) for value in row.dropna()}
        return (
            (required_all is None or set(required_all).issubset(observed))
            and (required_any is None or bool(set(required_any) & observed))
            and (excluded is None or not bool(set(excluded) & observed))
        )

    return frame[variable_columns].apply(keep, axis=1)


# This public signature follows the pinned R export; grouping it would break parity.
def plot_MI_rank(  # noqa: PLR0913, PLR0917
    mi: Mapping[str, pd.DataFrame],
    radius: float,
    depth: Sequence[int] | None = None,
    col_pals: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    p_thr: float | None = None,
    mi_thr: float | None = None,
    p_adj: bool = True,
    all: Sequence[str] | None = None,
    alo: Sequence[str] | None = None,
    not_: Sequence[str] | None = None,
    group: str | None = None,
    plot_bkgd: str = "W",
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], Figure]:
    """Filter and rank significant ensembles, returning SPACE's two data tables."""
    if col_pals is None:
        raise ValueError("col_pals must be provided")
    frame = _radius_frame(mi, radius)
    variables, metric, probability, zscore = _columns(frame, group, p_adj)
    selected_depths = list(depth) if depth is not None else list(range(1, len(variables) + 1))
    if not selected_depths or any(value <= 0 for value in selected_depths):
        raise ValueError("depth must contain positive integers")
    minimum_depth = min(selected_depths)
    if len(all or []) + int(alo is not None) > minimum_depth:
        raise ValueError("The number of required variables exceeds ensemble depth")
    frame = frame.loc[frame[variables].notna().sum(axis=1).isin(selected_depths)]
    frame = frame.loc[_required_filter(frame, variables, all, alo, not_)]
    probability_threshold = 0.05 if p_thr is None else float(p_thr)
    metric_threshold = (0.1 if group is None else 0.0) if mi_thr is None else float(mi_thr)
    frame = frame.loc[frame[probability] <= probability_threshold]
    metric_keep = frame[metric].abs() >= metric_threshold if metric == "CisMI" else frame[metric] >= metric_threshold
    frame = frame.loc[metric_keep].copy()
    frame = frame.iloc[np.argsort(-np.abs(frame[zscore].to_numpy(dtype=float)))]
    if not frame.empty:
        frame[probability] = [
            _estimated_log_probability(frame, probability, zscore, int(index)) for index in frame.index
        ]
    depths = frame[variables].notna().sum(axis=1)
    frame["size"] = 0.5
    for ensemble_depth in depths.unique():
        locations = depths == ensemble_depth
        magnitudes = frame.loc[locations, metric].abs()
        spread = float(magnitudes.max() - magnitudes.min())
        frame.loc[locations, "size"] = (magnitudes - magnitudes.min()) / spread + 0.1 if spread else 0.5

    unique_variables = pd.unique(frame[variables].to_numpy().ravel())
    unique_variables = [str(value) for value in unique_variables if pd.notna(value)]
    aggregate_rows: list[dict[str, float | str]] = []
    denominator = max(len(frame), 1)
    for variable in unique_variables:
        included = frame[variables].eq(variable).any(axis=1)
        aggregate_rows.append(
            {
                "V": variable,
                zscore: float(frame.loc[included, zscore].abs().sum() / denominator),
                "size": float(frame.loc[included, "size"].sum() / denominator),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows, columns=["V", zscore, "size"])
    if not aggregate.empty:
        aggregate = aggregate.sort_values(zscore, ascending=False, kind="stable").reset_index(drop=True)
        spread = float(aggregate["size"].max() - aggregate["size"].min())
        aggregate["size"] = (aggregate["size"] - aggregate["size"].min()) / spread + 0.1 if spread else 0.5

    figure, axes = plt.subplots()
    color = _background(figure, axes, plot_bkgd)
    axes.axhline(np.log10(probability_threshold), linestyle=":", color=color)
    axes.scatter(np.arange(1, len(frame) + 1), frame[probability], s=50 * frame["size"], color=color)
    axes.set(xlabel="Ensemble", ylabel="Corrected Log P Value")
    axes.invert_yaxis()
    figure.tight_layout()
    return (frame.reset_index(drop=True), aggregate), figure


__all__ = ["plot_MI_radius", "plot_MI_rank"]
