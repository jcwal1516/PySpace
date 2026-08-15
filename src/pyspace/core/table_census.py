"""Coordinate-table census with an exact pinned-SPACE compatibility path."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..io.table_loader import read_table
from .census_models import CensusResult, Neighborhood, summarize_neighborhoods
from .census_sampling import create_neighborhoods, normalize_radii, normalize_sample_sizes
from .patch_summary import summarize_patches


def _radius_key(radius: float) -> str:
    return str(int(radius)) if radius.is_integer() else str(radius)


def _normalize_os_pairs(os_pairs: Any, object_ids: list[int], scalar_count: int) -> dict[str, pd.DataFrame] | None:
    if os_pairs is None:
        return None
    entry = os_pairs.get("O1") if isinstance(os_pairs, dict) else os_pairs
    if entry is None:
        raise ValueError("OS_pairs must contain an O1 link table")
    if isinstance(entry, pd.DataFrame):
        return {"O1": entry.copy()}
    matrix = np.asarray(entry)
    expected = (len(object_ids), scalar_count)
    if matrix.ndim != 2 or matrix.shape != expected:
        raise ValueError(f"OS_pairs has shape {matrix.shape}; expected {expected}")
    return {
        "O1": pd.DataFrame(
            matrix,
            index=pd.Index([f"O1.{item}" for item in object_ids]),
            columns=pd.Index([f"S1.{index}" for index in range(1, scalar_count + 1)]),
        )
    }


def _eligible_indices(frame: pd.DataFrame, seed_points: Any) -> np.ndarray:
    if seed_points is None:
        allowed = frame["Object"].drop_duplicates().to_numpy()
    elif isinstance(seed_points, dict):
        value = seed_points.get("O1")
        allowed = frame["Object"].drop_duplicates().to_numpy() if value is None else np.asarray(value).ravel()
    else:
        allowed = np.asarray(seed_points).ravel()
    return np.flatnonzero(frame["Object"].isin(allowed).to_numpy())


def _explicit_seed_indices(plan: Any, radius_index: int, eligible: np.ndarray) -> np.ndarray | None:
    if plan is None:
        return None
    candidate = plan[radius_index] if isinstance(plan, (list, tuple)) and plan and np.asarray(plan[0]).ndim else plan
    indices = np.asarray(candidate, dtype=int).ravel()
    if np.any(indices < 0) or np.any(indices >= len(eligible)) or len(np.unique(indices)) != len(indices):
        raise ValueError("sample_indices must be unique valid zero-based indices into eligible rows")
    return eligible[indices]


def _r_census_table(
    frame: pd.DataFrame,
    radii: float | list[float],
    sample_size: int | list[int],
    os_pairs: Any,
    seed_points: Any,
    random_state: int | np.random.Generator | None,
    sample_indices: Any,
    source: str,
    cores: int,
) -> CensusResult:
    required = ["X", "Y", "Z", "Object"]
    if not all(name in frame.columns for name in required):
        raise ValueError("Error: Object table must be a data frame with at least X, Y, Z, and Object columns.")
    coordinates = frame[["X", "Y", "Z"]].to_numpy(dtype=float)
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("X, Y, and Z must contain only finite values")
    radius_values = normalize_radii(radii)
    sample_sizes = normalize_sample_sizes(sample_size, len(radius_values))
    eligible = _eligible_indices(frame, seed_points)
    scalar_columns = list(frame.columns[4:])
    if any(not pd.api.types.is_numeric_dtype(frame[name]) for name in scalar_columns):
        raise ValueError("Scalar columns after Object must be numeric")
    object_ids = sorted(int(value) for value in frame["Object"].drop_duplicates())
    normalized_pairs = _normalize_os_pairs(os_pairs, object_ids, len(scalar_columns))
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tree = cKDTree(coordinates)
    neighborhoods: list[Neighborhood] = []
    census_frames: list[pd.DataFrame] = []
    patch_list: dict[str, dict[str, pd.DataFrame]] = {}
    objects = frame["Object"].to_numpy()
    scalars = frame[scalar_columns].to_numpy(dtype=float) if scalar_columns else np.empty((len(frame), 0))
    for radius_index, (radius, requested) in enumerate(zip(radius_values, sample_sizes, strict=True)):
        if requested is None:
            raise ValueError("sample_size is required for R-compatible census_table calls.")
        if requested > len(eligible):
            raise ValueError("Cannot sample more seed points than eligible objects without replacement.")
        planned = _explicit_seed_indices(sample_indices, radius_index, eligible)
        seeds = generator.choice(eligible, size=requested, replace=False) if planned is None else planned[:requested]
        if len(seeds) < requested:
            raise ValueError("Explicit sample_indices does not contain enough rows")
        patch_chunks: list[pd.DataFrame] = []
        radius_neighborhoods: list[Neighborhood] = []
        for neighborhood_id, seed_index in enumerate(seeds, start=1):
            neighbor_indices = np.asarray(tree.query_ball_point(coordinates[seed_index], radius), dtype=int)
            distances = np.linalg.norm(coordinates[neighbor_indices] - coordinates[seed_index], axis=1)
            neighbor_indices = neighbor_indices[np.lexsort((neighbor_indices, distances))]
            patch_data: dict[str, Any] = {
                "Area": np.ones(len(neighbor_indices), dtype=int),
                "O1": objects[neighbor_indices],
            }
            for scalar_index in range(len(scalar_columns)):
                patch_data[f"S1.{scalar_index + 1}"] = scalars[neighbor_indices, scalar_index]
            patch_data["Nbhd"] = np.full(len(neighbor_indices), neighborhood_id, dtype=int)
            patch_chunks.append(pd.DataFrame(patch_data))
            radius_neighborhoods.append(
                Neighborhood(
                    center=tuple(float(value) for value in coordinates[seed_index]),
                    radius=radius,
                    points=coordinates[neighbor_indices],
                    object_ids=objects[neighbor_indices],
                    intensities=scalars[neighbor_indices] if scalar_columns else None,
                    patch_id=f"patch_{neighborhood_id}_{_radius_key(radius)}",
                    is_3d=True,
                )
            )
        patch_frame = pd.concat(patch_chunks, ignore_index=True)
        radius_patch_list = {"O1": patch_frame}
        radius_key = _radius_key(radius)
        patch_list[radius_key] = radius_patch_list
        summary = summarize_patches(radius_patch_list, normalized_pairs)
        for row_index, neighborhood in enumerate(radius_neighborhoods):
            neighborhood.variable_measurements = {
                name: float(summary.iloc[row_index][name]) for name in summary.columns
            }
        summary = pd.concat(
            [summary.reset_index(drop=True), frame.iloc[seeds][["X", "Y", "Z"]].reset_index(drop=True)], axis=1
        )
        summary["Radius"] = radius
        census_frames.append(summary)
        neighborhoods.extend(radius_neighborhoods)
    census = pd.concat(census_frames, ignore_index=True)
    variables = [name for name in census.columns if name not in {"X", "Y", "Z", "Radius"}]
    return CensusResult(
        neighborhoods=neighborhoods,
        metadata={
            "source_file": source,
            "table_shape": list(frame.shape),
            "coordinate_columns": ["X", "Y", "Z"],
            "radii": radius_values,
            "sample_size": sample_sizes,
            "total_neighborhoods": len(neighborhoods),
            "cores": cores,
            "r_parity_mode": "strict"
            if sample_indices is not None or len(eligible) == 1
            else "algorithmic_with_python_rng",
        },
        variables=variables,
        summary_stats=summarize_neighborhoods(neighborhoods),
        biomolecule_pairings=normalized_pairs,
        patch_list=patch_list,
        census=census,
    )


def _coordinate_columns(frame: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested is not None:
        columns = list(requested)
    else:
        lower = {str(name).lower(): str(name) for name in frame.columns}
        columns = [lower[name] for name in ("x", "y", "z") if name in lower]
    if len(columns) < 2 or any(name not in frame.columns for name in columns):
        raise ValueError("Coordinate table must contain at least X and Y columns")
    if not all(pd.api.types.is_numeric_dtype(frame[name]) for name in columns):
        raise ValueError("Coordinate columns must be numeric")
    if not np.all(np.isfinite(frame[columns].to_numpy(dtype=float))):
        raise ValueError("Coordinate columns must contain only finite values")
    return columns


def _generic_census_table(
    frame: pd.DataFrame,
    radii: float | list[float],
    variables: list[str] | None,
    coordinate_cols: list[str] | None,
    object_id_column: str | None,
    sample_size: int | list[int] | None,
    random_state: int | np.random.Generator | None,
    source: str,
) -> CensusResult:
    coordinates = _coordinate_columns(frame, coordinate_cols)
    object_column = object_id_column or next((name for name in ("object_id", "Object") if name in frame), None)
    excluded = {*coordinates, *((object_column,) if object_column else ())}
    value_columns = [
        str(name) for name in frame.columns if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    ]
    selected_variables = value_columns if variables is None else list(variables)
    missing = [name for name in selected_variables if name not in frame.columns]
    if missing:
        raise ValueError(f"Unknown variables: {missing}")
    if any(not pd.api.types.is_numeric_dtype(frame[name]) for name in selected_variables):
        raise ValueError("Census variables must be numeric")
    radius_values = normalize_radii(radii)
    counts = normalize_sample_sizes(sample_size, len(radius_values))
    points = frame[coordinates].to_numpy(dtype=float)
    generator = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    maximum = max(len(points) if count is None else count for count in counts)
    if maximum > len(points):
        raise ValueError("Cannot sample more seed points than table rows without replacement.")
    sequence = points[generator.permutation(len(points))[:maximum]]
    neighborhoods: list[Neighborhood] = []
    rows: list[dict[str, float]] = []
    object_values = sorted(frame[object_column].drop_duplicates(), key=str) if object_column else []
    for radius, count in zip(radius_values, counts, strict=True):
        selected = sequence[: len(points) if count is None else count]
        radius_neighborhoods = create_neighborhoods(
            frame,
            radius,
            seed_points=selected,
            object_id_column=object_column,
            random_state=generator,
        )[radius]
        for neighborhood in radius_neighborhoods:
            row: dict[str, float] = {}
            if object_column and neighborhood.object_ids is not None:
                denominator = len(neighborhood.object_ids)
                for value in object_values:
                    row[f"O1.{value}"] = 100 * float(np.count_nonzero(neighborhood.object_ids == value)) / denominator
            if selected_variables and neighborhood.intensities is not None:
                all_numeric = list(value_columns)
                positions = [all_numeric.index(name) for name in selected_variables]
                means = np.mean(neighborhood.intensities[:, positions], axis=0)
                row.update({name: float(value) for name, value in zip(selected_variables, means, strict=True)})
            for position, name in enumerate(("X", "Y", "Z")[: len(neighborhood.center)]):
                row[name] = float(neighborhood.center[position])
            if len(neighborhood.center) == 2:
                row["Z"] = 0.0
            row["Radius"] = radius
            neighborhood.variable_measurements = {
                key: value for key, value in row.items() if key not in {"X", "Y", "Z", "Radius"}
            }
            rows.append(row)
        neighborhoods.extend(radius_neighborhoods)
    census = pd.DataFrame(rows)
    variable_names = [name for name in census.columns if name not in {"X", "Y", "Z", "Radius"}]
    census = census[[*variable_names, "X", "Y", "Z", "Radius"]]
    return CensusResult(
        neighborhoods=neighborhoods,
        metadata={
            "source_file": source,
            "table_shape": list(frame.shape),
            "coordinate_columns": coordinates,
            "radii": radius_values,
            "sample_size": counts,
            "total_neighborhoods": len(neighborhoods),
            "r_parity_mode": "non_parity_python_coordinates",
        },
        variables=variable_names,
        summary_stats=summarize_neighborhoods(neighborhoods),
        census=census,
    )


def _table_input(table_data: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if isinstance(table_data, pd.DataFrame):
        return table_data.copy(deep=True), "DataFrame input"
    return read_table(table_data), str(Path(table_data))


def census_coordinates(
    table_data: str | Path | pd.DataFrame,
    radii: float | list[float],
    variables: list[str] | None = None,
    coordinate_cols: list[str] | None = None,
    object_id_column: str | None = None,
    sample_size: int | list[int] | None = None,
    *,
    random_state: int | np.random.Generator | None = None,
) -> CensusResult:
    """Census an ordinary coordinate table through the explicit Python-only path."""
    frame, source = _table_input(table_data)
    return _generic_census_table(
        frame,
        radii,
        variables,
        coordinate_cols,
        object_id_column,
        sample_size,
        random_state,
        source,
    )


def census_table(
    table_data: str | Path | pd.DataFrame,
    radii: float | list[float],
    sample_size: int | list[int],
    os_pairs: Any = None,
    seed_points: Any = None,
    cores: int | None = None,
    *,
    sample_indices: Any = None,
    random_state: int | np.random.Generator | None = None,
) -> CensusResult:
    """Match ``SPACE::census_table`` with local or explicitly planned randomness."""
    worker_count = 1 if cores is None else int(cores)
    if worker_count <= 0:
        raise ValueError("cores must be a positive integer or None")
    frame, source = _table_input(table_data)
    return _r_census_table(
        frame,
        radii,
        sample_size,
        os_pairs,
        seed_points,
        random_state,
        sample_indices,
        source,
        worker_count,
    )


def standardize_censuses(censuses: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Match ``SPACE::standardize_censuses`` natural ordering and zero fill."""
    if not censuses:
        return []

    def natural_key(value: str) -> list[int | str]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

    coordinates = ["X", "Y", "Z", "Radius"]
    variables = sorted(
        {str(column) for frame in censuses for column in frame.columns if column not in coordinates},
        key=natural_key,
    )
    columns = [*variables, *coordinates]
    output: list[pd.DataFrame] = []
    for census in censuses:
        aligned = census.copy(deep=True)
        for missing in set(columns) - set(aligned.columns):
            aligned[missing] = 0
        output.append(aligned[columns])
    return output


__all__ = ["census_coordinates", "census_table", "standardize_censuses"]
