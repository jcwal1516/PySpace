"""SPACE-compatible covariation pattern learning over a deterministic SOM."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .patch_summary import random_census
from .pattern_models import PatternResult, SelfOrganizingMap, SOMResult


def _radius_key(mapping: Mapping[str | float, Any], radius: float) -> Any:
    matches = [value for key, value in mapping.items() if float(key) == radius]
    if len(matches) != 1:
        raise ValueError(f"Exactly one value must match radius {radius}")
    return matches[0]


def _linked_pairs(columns: Sequence[str]) -> dict[str, pd.DataFrame] | None:
    linked = [column for column in columns if sum(token in column for token in ("O", "S")) == 2 and "_" in column]
    if not linked:
        return None
    groups: dict[str, tuple[list[str], list[str]]] = {}
    for name in linked:
        object_name, scalar_name = name.split("_", 1)
        object_map = object_name.split(".", 1)[0]
        objects, scalars = groups.setdefault(object_map, ([], []))
        if object_name not in objects:
            objects.append(object_name)
        if scalar_name not in scalars:
            scalars.append(scalar_name)
    result: dict[str, pd.DataFrame] = {}
    for object_map, (objects, scalars) in groups.items():
        matrix = pd.DataFrame(0, index=objects, columns=scalars)
        for name in linked:
            object_name, scalar_name = name.split("_", 1)
            if object_name.startswith(f"{object_map}."):
                matrix.loc[object_name, scalar_name] = 1
        result[object_map] = matrix
    return result


def _focal_reference(
    census: pd.DataFrame | Sequence[pd.DataFrame],
    radius: float,
    group: pd.DataFrame | None,
    focal: Sequence[str] | str | None,
    reference: Sequence[str] | str | None,
    patch_list: Mapping[str | float, Mapping[str, pd.DataFrame]] | None,
    generator: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    censuses = [census] if isinstance(census, pd.DataFrame) else list(census)
    if not censuses:
        raise ValueError("census cannot be empty")
    narrowed = [frame.loc[frame["Radius"].astype(float) == radius].copy() for frame in censuses]
    if any(frame.empty for frame in narrowed):
        raise ValueError(f"Each census must contain radius {radius}")
    if len(narrowed) > 1 and patch_list is not None:
        raise ValueError("Provide groups for multiple censuses or a patch list for one census, not both")
    if len(narrowed) > 1 and group is not None:
        if len(group) != len(narrowed):
            raise ValueError("group must have one row per census")
        focal_values = {str(focal)} if isinstance(focal, str) else set(map(str, focal or []))
        reference_values = {str(reference)} if isinstance(reference, str) else set(map(str, reference or []))
        candidates = [
            column
            for column in group
            if focal_values.issubset(set(group[column].astype(str)))
            and reference_values.issubset(set(group[column].astype(str)))
        ]
        if len(candidates) != 1:
            raise ValueError("focal and reference must identify exactly one grouping factor")
        column = candidates[0]
        focal_frames = [
            frame for frame, label in zip(narrowed, group[column], strict=True) if str(label) in focal_values
        ]
        reference_frames = [
            frame for frame, label in zip(narrowed, group[column], strict=True) if str(label) in reference_values
        ]
        if not focal_frames or not reference_frames:
            raise ValueError("focal and reference must each select at least one census")
        return pd.concat(focal_frames, ignore_index=True), pd.concat(reference_frames, ignore_index=True)
    focal_frame = pd.concat(narrowed, ignore_index=True)
    if patch_list is None:
        return focal_frame, None
    patches = dict(_radius_key(patch_list, radius))
    return focal_frame, random_census(patches, _linked_pairs(focal_frame.columns), rng=generator)


def _rolling_summary(
    values: np.ndarray,
    assignments: np.ndarray,
    half_window: int,
    confidence: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if half_window == 0:
        return values.copy(), values.copy(), values.copy()
    left = values[assignments == assignments.min()]
    right = values[assignments == assignments.max()]
    padded = np.concatenate(
        [
            generator.choice(left, half_window, replace=True),
            values,
            generator.choice(right, half_window, replace=True),
        ]
    )
    windows = sliding_window_view(padded, 2 * half_window + 1)
    mean = windows.mean(axis=1)
    lower = np.quantile(windows, (1 - confidence) / 2, axis=1)
    upper = np.quantile(windows, 1 - (1 - confidence) / 2, axis=1)

    def smooth_bound(bound: np.ndarray) -> np.ndarray:
        left_pool = bound[:half_window]
        right_pool = bound[max(len(bound) - half_window - 1, 0) :]
        extended = np.concatenate(
            [
                generator.choice(left_pool, half_window, replace=True),
                bound,
                generator.choice(right_pool, half_window, replace=True),
            ]
        )
        return sliding_window_view(extended, 2 * half_window + 1).mean(axis=1)

    return mean, smooth_bound(lower), smooth_bound(upper)


def _normalize_output(output: pd.DataFrame, variables: Sequence[str], mode: str | None, scalar: bool) -> None:
    selected = [variable for variable in variables if ("S" in variable) is scalar]
    if not selected or mode is None:
        return
    if mode not in {"all", "ind"}:
        raise ValueError("Normalization modes must be None, 'all', or 'ind'")
    shared_maximum = float(output.loc[output["V"].isin(selected), "Ymax"].max())
    for variable in selected:
        denominator = shared_maximum if mode == "all" else float(output.loc[output["V"] == variable, "Ymax"].max())
        if denominator == 0:
            continue
        locations = output["V"] == variable
        for source, target in (("Y", "Y_norm"), ("Ymin", "Ymin_norm"), ("Ymax", "Ymax_norm")):
            output.loc[locations, target] = 100 * output.loc[locations, source] / denominator


def _learn_pattern(  # noqa: PLR0913, PLR0917
    census: pd.DataFrame | Sequence[pd.DataFrame],
    ensemble: Sequence[str],
    radius: float,
    col_pal: Mapping[str, Sequence[str]],
    group: pd.DataFrame | None = None,
    focal: Sequence[str] | str | None = None,
    reference: Sequence[str] | str | None = None,
    patch_list: Mapping[str | float, Mapping[str, pd.DataFrame]] | None = None,
    conf_int: float = 0.95,
    obj_norm: str | None = None,
    scl_norm: str | None = "all",
    som_reps: int = 50,
    toroidal: bool = False,
    smooth_window: int = 100,
    sub_sample: bool = False,
    plot_bkgd: str = "W",
    *,
    random_state: int | np.random.Generator | None = None,
    som_initialization: np.ndarray | None = None,
    som_epoch_orders: Sequence[np.ndarray] | None = None,
) -> tuple[pd.DataFrame, SOMResult]:
    if not ensemble:
        raise ValueError("ensemble cannot be empty")
    if not 0 < conf_int < 1:
        raise ValueError("conf_int must be in (0, 1)")
    if smooth_window < 0:
        raise ValueError("smooth_window cannot be negative")
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")
    if not col_pal:
        raise ValueError("col_pal cannot be empty")
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    focal_frame, reference_frame = _focal_reference(
        census, float(radius), group, focal, reference, patch_list, generator
    )
    missing = [variable for variable in ensemble if variable not in focal_frame]
    if missing:
        raise ValueError(f"Unknown ensemble variables: {missing}")
    focal_values = focal_frame[list(ensemble)].to_numpy(dtype=float)
    if np.any(~np.isfinite(focal_values)):
        raise ValueError("ensemble values must be finite")
    if sub_sample and len(focal_values) > 10_000:
        indices = generator.choice(len(focal_values), 10_000, replace=False)
        focal_values = focal_values[indices]
    reference_values = None if reference_frame is None else reference_frame[list(ensemble)].to_numpy(dtype=float)
    if reference_values is not None and sub_sample and len(reference_values) > 10_000:
        indices = generator.choice(len(reference_values), 10_000, replace=False)
        reference_values = reference_values[indices]

    som = SelfOrganizingMap(
        grid_size=min(len(focal_values), 1000),
        max_iterations=som_reps,
        topology="toroidal" if toroidal else "linear",
        random_seed=int(generator.integers(0, np.iinfo(np.int32).max)),
    ).fit(focal_values, initial_weights=som_initialization, epoch_orders=som_epoch_orders)
    if som.weights is None:
        raise RuntimeError("SOM fitting did not produce weights")
    if not toroidal and np.sum(som.weights[0] ** 2) > np.sum(som.weights[-1] ** 2):
        som.weights = som.weights[::-1].copy()
    focal_assignments = som.predict(focal_values)
    focal_errors = som.get_quantization_errors(focal_values)
    order = np.argsort(focal_assignments, kind="stable")
    ordered_values = focal_values[order]
    ordered_assignments = focal_assignments[order]
    half_window = round(smooth_window / 2)

    enrichment = np.ones(len(ordered_values), dtype=float)
    if reference_values is not None:
        reference_assignments = som.predict(reference_values)
        focal_distance = np.linalg.norm((focal_values - som.weights[focal_assignments]) / 100, axis=1)
        reference_distance = np.linalg.norm((reference_values - som.weights[reference_assignments]) / 100, axis=1)
        maximum_distance = math.sqrt(len(ensemble))
        focal_weights = (maximum_distance - focal_distance) / maximum_distance
        reference_weights = (maximum_distance - reference_distance) / maximum_distance
        focal_frequency = np.bincount(focal_assignments, weights=focal_weights, minlength=som.grid_size) / len(
            focal_values
        )
        reference_frequency = np.bincount(
            reference_assignments, weights=reference_weights, minlength=som.grid_size
        ) / len(reference_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            node_enrichment = np.log(focal_frequency / reference_frequency)
        node_enrichment[focal_frequency == 0] = 0
        finite = node_enrichment[np.isfinite(node_enrichment)]
        replacement = float(finite.max()) if len(finite) else 0.0
        node_enrichment[(focal_frequency > 0) & (reference_frequency == 0)] = replacement
        raw = node_enrichment[ordered_assignments]
        enrichment = _rolling_summary(raw, ordered_assignments, half_window, conf_int, generator)[0]

    blocks: list[pd.DataFrame] = []
    x_values = np.linspace(0, 100, len(ordered_values))
    for variable_index, variable in enumerate(ensemble):
        mean, lower, upper = _rolling_summary(
            ordered_values[:, variable_index], ordered_assignments, half_window, conf_int, generator
        )
        lower = np.minimum(lower, mean)
        upper = np.maximum(upper, mean)
        blocks.append(
            pd.DataFrame(
                {
                    "X": x_values,
                    "Y": mean,
                    "Ymin": lower,
                    "Ymax": upper,
                    "V": variable,
                    "Enr": enrichment,
                }
            )
        )
    output = pd.concat(blocks, ignore_index=True)
    output["Y_norm"] = output["Y"]
    output["Ymin_norm"] = output["Ymin"]
    output["Ymax_norm"] = output["Ymax"]
    _normalize_output(output, ensemble, scl_norm, scalar=True)
    _normalize_output(output, ensemble, obj_norm, scalar=False)
    som_result = SOMResult(
        weights=som.weights.copy(),
        node_assignments=focal_assignments,
        quantization_errors=focal_errors,
        training_history=som.training_history,
        grid_size=som.grid_size,
        input_dimensions=len(ensemble),
    )
    return output, som_result


def learn_pattern(  # noqa: PLR0913, PLR0917
    census: pd.DataFrame | Sequence[pd.DataFrame],
    ensemble: Sequence[str],
    radius: float,
    col_pal: Mapping[str, Sequence[str]],
    group: pd.DataFrame | None = None,
    focal: Sequence[str] | str | None = None,
    reference: Sequence[str] | str | None = None,
    patch_list: Mapping[str | float, Mapping[str, pd.DataFrame]] | None = None,
    conf_int: float = 0.95,
    obj_norm: str | None = None,
    scl_norm: str | None = "all",
    som_reps: int = 50,
    toroidal: bool = False,
    smooth_window: int = 100,
    sub_sample: bool = False,
    plot_bkgd: str = "W",
    **random_plan: Any,
) -> pd.DataFrame:
    """Return the covariation table produced by the pinned SPACE workflow."""
    return _learn_pattern(
        census,
        ensemble,
        radius,
        col_pal,
        group,
        focal,
        reference,
        patch_list,
        conf_int,
        obj_norm,
        scl_norm,
        som_reps,
        toroidal,
        smooth_window,
        sub_sample,
        plot_bkgd,
        **random_plan,
    )[0]


def learn_pattern_result(
    census: pd.DataFrame | Sequence[pd.DataFrame],
    ensemble: Sequence[str],
    radius: float,
    col_pal: Mapping[str, Sequence[str]],
    **kwargs: Any,
) -> PatternResult:
    """Return the Python-friendly typed layer over :func:`learn_pattern`."""
    output, som_result = _learn_pattern(census, ensemble, radius, col_pal, **kwargs)
    return PatternResult(
        som_result=som_result,
        covariation_data=output,
        enrichment_scores=output.loc[output["V"] == ensemble[0], "Enr"].to_numpy(),
        variable_names=list(ensemble),
        significance_threshold=float(kwargs.get("conf_int", 0.95)),
    )


__all__ = ["learn_pattern", "learn_pattern_result"]
