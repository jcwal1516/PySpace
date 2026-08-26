"""Explicit table loading, including the three SPACE table contracts."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".parquet"}


def read_table(path: str | Path, *, index_col: int | None = None) -> pd.DataFrame:
    """Read a supported tabular format using an explicit extension dispatch."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Table file not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in _TABLE_SUFFIXES:
        raise ValueError(f"Unsupported table format: {source.suffix or '<none>'}")
    if suffix == ".csv":
        return pd.read_csv(source, index_col=index_col)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t", index_col=index_col)
    if suffix == ".xlsx":
        return pd.read_excel(source, index_col=index_col)
    if index_col is not None:
        frame = pd.read_parquet(source)
        return frame.set_index(frame.columns[index_col])
    return pd.read_parquet(source)


def _profile_by_coordinates(profile: pd.DataFrame, image: np.ndarray, palette: list[str]) -> pd.DataFrame:
    coordinate_columns = {"Row", "Column", "Z"} if {"Row", "Column"}.issubset(profile.columns) else {"X", "Y", "Z"}
    if not coordinate_columns.issubset(profile.columns):
        raise ValueError("Profile coordinate matching requires X/Y/Z or Row/Column/Z columns")
    row_name, column_name = ("Row", "Column") if "Row" in coordinate_columns else ("X", "Y")
    labels = np.asarray(image).squeeze(axis=-1) if np.asarray(image).shape[-1:] == (1,) else np.asarray(image)
    expected = len(palette) - int(np.any(labels == 0))
    positions: list[int | None] = [None] * expected
    for row_index, row in profile.iterrows():
        coordinates = [int(row[row_name]) - 1, int(row[column_name]) - 1]
        if labels.ndim == 3:
            coordinates.append(int(row["Z"]) - 1)
        object_id = int(labels[tuple(coordinates)])
        if 0 < object_id <= expected:
            if not isinstance(row_index, (int, np.integer)):
                raise ValueError("Profile coordinates require an integer row index")
            positions[object_id - 1] = int(row_index)
    if any(position is None for position in positions):
        raise ValueError("Profile coordinates do not identify every image object")
    resolved_positions = [position for position in positions if position is not None]
    result = profile.loc[resolved_positions].reset_index(drop=True)
    result["Object"] = np.arange(1, expected + 1)
    return result.drop(columns=list(coordinate_columns))


def _profile_by_counts(profile: pd.DataFrame, image: np.ndarray, palette: list[str]) -> pd.DataFrame:
    if "Count" not in profile or profile["Count"].duplicated().any():
        raise ValueError("Cannot match objects by non-unique counts; provide one pixel coordinate per object")
    labels = np.asarray(image)
    if float(profile["Count"].sum()) >= float(np.count_nonzero(labels > 0)):
        counts = [int(np.count_nonzero(labels == object_id)) for object_id in range(1, len(palette) + 1)]
    else:
        structure = np.ones((3,) * labels.ndim, dtype=np.uint8)
        counts = [
            ndimage.label(labels == object_id, structure=structure)[1] for object_id in range(1, len(palette) + 1)
        ]
    count_to_index = {int(count): int(index) for index, count in profile["Count"].items()}
    missing = [count for count in counts if count not in count_to_index]
    if missing:
        raise ValueError(f"Image object counts are absent from profile table: {missing}")
    result = profile.loc[[count_to_index[count] for count in counts]].reset_index(drop=True)
    result["Object"] = np.arange(1, len(palette) + 1)
    return result


def load_table(
    in_file: str | Path,
    table_type: str,
    img: np.ndarray | None,
    col_pal: list[str] | None,
) -> pd.DataFrame:
    """Load a profile (``P``), object (``O``), or link (``L``) table.

    Pass ``None`` for both matching inputs when loading an object/link table or
    when intentionally accepting SPACE's unmatched-profile warning.
    """
    if table_type not in {"P", "O", "L"}:
        raise ValueError("table_type must be P, O, or L")
    result = read_table(in_file, index_col=0 if table_type == "L" else None)
    if table_type == "L":
        return result
    if table_type == "O":
        if "Z" not in result.columns:
            result.insert(2, "Z", 1)
        return result

    if "Object" not in result.columns:
        raise ValueError("Profile tables require an Object column")
    result = result.sort_values("Object", kind="stable").reset_index(drop=True)
    if img is None and col_pal is None:
        warnings.warn(
            "Without an image and color palette, objects might not match the palette order",
            RuntimeWarning,
            stacklevel=2,
        )
        return result
    if img is None or col_pal is None:
        raise ValueError("Profile matching requires both img and col_pal, or neither")
    if ({"X", "Y", "Z"}.issubset(result.columns)) or ({"Row", "Column", "Z"}.issubset(result.columns)):
        return _profile_by_coordinates(result, img, col_pal)
    return _profile_by_counts(result, img, col_pal)


def load_coordinate_table(path: str | Path) -> pd.DataFrame:
    """Read and validate a generic X/Y coordinate table."""
    result = read_table(path)
    lower = {str(column).lower(): column for column in result.columns}
    if not {"x", "y"}.issubset(lower):
        raise ValueError("Coordinate table requires X/Y or x/y columns")
    coordinates = [lower["x"], lower["y"]]
    if not all(pd.api.types.is_numeric_dtype(result[column]) for column in coordinates):
        raise ValueError("Coordinate columns must be numeric")
    if result[coordinates].isna().any().any():
        raise ValueError("Coordinate columns must not contain missing values")
    return result


__all__ = ["load_coordinate_table", "load_table", "read_table"]
