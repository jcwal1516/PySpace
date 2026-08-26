"""Validation reports for supported public input boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .table_loader import read_table

_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".parquet"}


def _table_validation(frame: pd.DataFrame) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    if frame.empty:
        errors.append("Table must contain at least one row")
    lower = {str(column).lower(): column for column in frame.columns}
    if not {"x", "y"}.issubset(lower):
        errors.append("Table must contain X/Y or x/y coordinate columns")
    else:
        coordinates = [lower["x"], lower["y"]]
        if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in coordinates):
            errors.append("Coordinate columns must be numeric")
        elif frame[coordinates].isna().any().any():
            errors.append("Coordinate columns must not contain missing values")
    return errors, [], {"row_count": len(frame), "column_count": len(frame.columns)}


def _census_validation(frame: pd.DataFrame) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    if "Radius" not in frame:
        errors.append("Census table requires a Radius column")
        radius_counts: dict[float, int] = {}
    else:
        radius_counts = {float(radius): int(count) for radius, count in frame["Radius"].value_counts().items()}
    variables = [column for column in frame if str(column).startswith(("O", "S"))]
    if not variables:
        errors.append("Census table requires at least one O*/S* variable column")
    elif not all(pd.api.types.is_numeric_dtype(frame[column]) for column in variables):
        errors.append("Census variable columns must be numeric")
    return errors, [], {"row_count": len(frame), "column_count": len(frame.columns), "radius_counts": radius_counts}


def validate_inputs(
    data: Any,
    data_type: str = "auto",
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Validate a supported public input without guessing unknown formats."""
    metadata: dict[str, Any] = {}
    value = data
    inferred = data_type
    if isinstance(value, dict) and "census" in value:
        metadata = dict(value.get("metadata") or {})
        value = value["census"]
        inferred = "census" if data_type == "auto" else data_type
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input path not found: {path}")
        suffix = path.suffix.lower()
        if suffix in _TABLE_SUFFIXES:
            value = read_table(path)
            inferred = "table" if data_type == "auto" else data_type
        elif suffix in _IMAGE_SUFFIXES:
            inferred = "image" if data_type == "auto" else data_type
            value = path
        else:
            raise ValueError(f"Unsupported input format: {path.suffix or '<none>'}")
    if data_type == "auto" and isinstance(value, pd.DataFrame):
        inferred = "census" if "Radius" in value else "table"

    if inferred == "table" and isinstance(value, pd.DataFrame):
        errors, warnings, stats = _table_validation(value)
    elif inferred == "census" and isinstance(value, pd.DataFrame):
        errors, warnings, stats = _census_validation(value)
    elif inferred == "image" and isinstance(value, Path):
        errors, warnings, stats = [], [], {"path": str(value), "size_bytes": value.stat().st_size}
    else:
        errors, warnings, stats = [f"Unsupported {inferred} input value"], [], {}
    if raise_on_error and errors:
        raise ValueError("; ".join(errors))
    return {
        "valid": not errors,
        "data_type": inferred,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "metadata": metadata,
    }


__all__ = ["validate_inputs"]
