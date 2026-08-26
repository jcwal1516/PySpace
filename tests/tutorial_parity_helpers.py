"""Comparison helpers owned by the tutorial parity tests."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def structure_diff(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_columns = list(left.columns)
    right_columns = list(right.columns)
    return {
        "left_shape": [int(left.shape[0]), int(left.shape[1])],
        "right_shape": [int(right.shape[0]), int(right.shape[1])],
        "shape_equal": left.shape == right.shape,
        "column_order_equal": left_columns == right_columns,
        "missing_in_left": sorted(set(right_columns) - set(left_columns)),
        "extra_in_left": sorted(set(left_columns) - set(right_columns)),
    }


def coordinate_overlap_stats(
    left: pd.DataFrame,
    right: pd.DataFrame,
    coordinate_columns: tuple[str, str, str] = ("X", "Y", "Z"),
) -> dict[str, Any]:
    missing = [name for name in coordinate_columns if name not in left or name not in right]
    if missing:
        return {
            "left_unique": 0,
            "right_unique": 0,
            "intersection": 0,
            "overlap_vs_left": 0.0,
            "overlap_vs_right": 0.0,
            "note": f"missing coordinate columns: {missing}",
        }
    left_coordinates = set(
        map(tuple, np.rint(left.loc[:, list(coordinate_columns)].to_numpy(dtype=float)).astype(np.int64).tolist())
    )
    right_coordinates = set(
        map(tuple, np.rint(right.loc[:, list(coordinate_columns)].to_numpy(dtype=float)).astype(np.int64).tolist())
    )
    intersection_count = len(left_coordinates & right_coordinates)
    return {
        "left_unique": len(left_coordinates),
        "right_unique": len(right_coordinates),
        "intersection": intersection_count,
        "overlap_vs_left": float(intersection_count / len(left_coordinates)) if left_coordinates else 0.0,
        "overlap_vs_right": float(intersection_count / len(right_coordinates)) if right_coordinates else 0.0,
    }


def distribution_gap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    if columns is None:
        left_variables = {name for name in left if str(name).startswith(("O", "S"))}
        right_variables = {name for name in right if str(name).startswith(("O", "S"))}
        columns = sorted(left_variables & right_variables)
    if not columns:
        return {
            "n_columns": 0,
            "avg_abs_mean_diff": 0.0,
            "median_abs_mean_diff": 0.0,
            "max_abs_mean_diff": 0.0,
            "avg_abs_std_diff": 0.0,
            "top_abs_mean_diffs": [],
        }
    left_numeric = left[columns].apply(pd.to_numeric, errors="coerce")
    right_numeric = right[columns].apply(pd.to_numeric, errors="coerce")
    statistics = pd.DataFrame(
        {
            "left_mean": left_numeric.mean(),
            "right_mean": right_numeric.mean(),
            "left_std": left_numeric.std(ddof=0),
            "right_std": right_numeric.std(ddof=0),
        }
    )
    statistics["abs_mean_diff"] = (statistics["left_mean"] - statistics["right_mean"]).abs()
    statistics["abs_std_diff"] = (statistics["left_std"] - statistics["right_std"]).abs()
    top = statistics.nlargest(top_n, "abs_mean_diff").reset_index(names="variable")
    return {
        "n_columns": len(columns),
        "avg_abs_mean_diff": float(statistics["abs_mean_diff"].mean()),
        "median_abs_mean_diff": float(statistics["abs_mean_diff"].median()),
        "max_abs_mean_diff": float(statistics["abs_mean_diff"].max()),
        "avg_abs_std_diff": float(statistics["abs_std_diff"].mean()),
        "top_abs_mean_diffs": top.to_dict(orient="records"),
    }


def _object_ids(columns: Any, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)$")
    return sorted({int(match.group(1)) for column in columns if (match := pattern.match(str(column)))})


def infer_object_permutation(left: pd.DataFrame, right: pd.DataFrame, prefix: str) -> dict[int, int]:
    left_ids = _object_ids(left.columns, prefix)
    right_ids = _object_ids(right.columns, prefix)
    if not left_ids or len(left_ids) != len(right_ids):
        return {}
    left_means = np.array([pd.to_numeric(left[f"{prefix}.{item}"], errors="coerce").mean() for item in left_ids])
    right_means = np.array([pd.to_numeric(right[f"{prefix}.{item}"], errors="coerce").mean() for item in right_ids])
    rows, columns = linear_sum_assignment(np.abs(left_means[:, None] - right_means[None, :]))
    return {left_ids[row]: right_ids[column] for row, column in zip(rows, columns, strict=True)}


def apply_object_permutation(frame: pd.DataFrame, prefix: str, mapping: dict[int, int]) -> pd.DataFrame:
    if not mapping:
        return frame.copy()
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)(.*)$")
    planned: dict[str, str] = {}
    for column in frame.columns:
        match = pattern.match(str(column))
        if match and int(match.group(1)) in mapping:
            planned[str(column)] = f"{prefix}.{mapping[int(match.group(1))]}{match.group(2)}"
    temporary = {old: f"__tmp__{index}__" for index, old in enumerate(planned)}
    return frame.rename(columns=temporary).rename(columns={temp: planned[old] for old, temp in temporary.items()})
