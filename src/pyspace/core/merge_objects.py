"""Merge corresponding SPACE object representations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd


def _groups(value: Sequence[int] | Sequence[Sequence[int]] | None) -> list[list[int]]:
    if value is None or len(value) == 0:
        raise ValueError("obj_groups must contain at least one object group")
    first = value[0]
    if isinstance(first, (int, np.integer)):
        groups = [[int(object_id) for object_id in cast(Sequence[int], value)]]
    else:
        groups = [[int(object_id) for object_id in group] for group in cast(Sequence[Sequence[int]], value)]
    if any(not group or any(object_id <= 0 for object_id in group) for group in groups):
        raise ValueError("obj_groups must contain non-empty groups of positive object IDs")
    return groups


def _merge_profile(profile: pd.DataFrame, groups: list[list[int]]) -> pd.DataFrame:
    if profile.shape[1] < 3:
        raise ValueError("prof_table requires object, count, and at least one profile column")
    object_column, count_column, *profile_columns = profile.columns
    if not all(pd.api.types.is_numeric_dtype(profile[column]) for column in profile.columns):
        raise TypeError("prof_table columns must be numeric")
    merged_ids = {object_id for group in groups for object_id in group}
    remaining = profile.loc[~profile[object_column].isin(merged_ids)].copy()
    rows: list[dict[object, float | int]] = []
    for group in groups:
        selected = profile.loc[profile[object_column].isin(group)]
        if len(selected) != len(group):
            missing = sorted(set(group) - set(selected[object_column].astype(int)))
            raise ValueError(f"prof_table is missing object IDs: {missing}")
        weights = selected[count_column].to_numpy(dtype=float)
        total = float(weights.sum())
        if total == 0:
            raise ValueError("Cannot merge profiles whose total Count is zero")
        row: dict[object, float | int] = {object_column: min(group), count_column: total}
        for column in profile_columns:
            row[column] = float(np.sum(selected[column].to_numpy(dtype=float) * weights) / total)
        rows.append(row)
    result = pd.concat([remaining, pd.DataFrame(rows)], ignore_index=True)
    result = result.sort_values(object_column, kind="stable").reset_index(drop=True)
    result[object_column] = np.arange(1, len(result) + 1)
    return result.astype(profile.dtypes.to_dict())


def _merge_codes(values: np.ndarray, groups: list[list[int]], *, preserve_zero: bool) -> np.ndarray:
    result = np.asarray(values).copy()
    for group in groups:
        result[np.isin(result, group)] = min(group)
    remaining = np.sort(np.unique(result))
    if preserve_zero:
        remaining = remaining[remaining != 0]
    original = result.copy()
    for new_id, old_id in enumerate(remaining, start=1):
        result[original == old_id] = new_id
    return result


def _merge_palette(palette: Sequence[str], groups: list[list[int]]) -> list[str]:
    deleted = {object_id for group in groups for object_id in group if object_id != min(group)}
    return [color for object_id, color in enumerate(palette, start=1) if object_id not in deleted]


def merge_objects(
    prof_table: pd.DataFrame | None,
    img: np.ndarray | None,
    col_pal: Sequence[str] | None,
    obj_table: pd.DataFrame | None,
    obj_groups: Sequence[int] | Sequence[Sequence[int]] | None,
) -> dict[str, pd.DataFrame | np.ndarray | list[str]]:
    """Merge object groups in each supplied representation without mutation.

    All five arguments mirror the pinned R export; pass ``None`` explicitly for
    representations that are not available.
    """
    groups = _groups(obj_groups)
    result: dict[str, pd.DataFrame | np.ndarray | list[str]] = {}
    if prof_table is not None:
        result["profile_table"] = _merge_profile(prof_table, groups)
    if img is not None:
        result["image"] = _merge_codes(img, groups, preserve_zero=True)
    if col_pal is not None:
        result["color_palette"] = _merge_palette(col_pal, groups)
    if obj_table is not None:
        if "Object" not in obj_table:
            raise ValueError("obj_table requires an Object column")
        table = obj_table.copy()
        table["Object"] = _merge_codes(table["Object"].to_numpy(), groups, preserve_zero=False)
        result["object_table"] = table
    return result


__all__ = ["merge_objects"]
