"""R-compatible patch summarization and patch-preserving randomization."""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import pandas as pd


def _natural_sort(labels: list[Any]) -> list[str]:
    def key(value: Any) -> list[Any]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]

    return sorted((str(label) for label in labels), key=key)


def _pair_names(entry: pd.DataFrame | np.ndarray) -> tuple[str, ...]:
    if isinstance(entry, pd.DataFrame):
        matrix = entry.to_numpy()
        rows = [str(value) for value in entry.index]
        columns = [str(value) for value in entry.columns]
    else:
        matrix = np.asarray(entry)
        if matrix.ndim != 2:
            raise ValueError("Object-scalar link tables must be two-dimensional")
        rows = [str(index) for index in range(1, matrix.shape[0] + 1)]
        columns = [str(index) for index in range(1, matrix.shape[1] + 1)]
    return tuple(
        f"{row}_{column}"
        for column_index, column in enumerate(columns)
        for row_index, row in enumerate(rows)
        if matrix[row_index, column_index] > 0
    )


def _pair_lookup(
    os_pairs: dict[str, pd.DataFrame | np.ndarray] | None,
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    lookup: dict[str, tuple[str, ...]] = {}
    linked_scalars: set[str] = set()
    for key, entry in (os_pairs or {}).items():
        names = _pair_names(entry)
        lookup[str(key)] = names
        lookup.setdefault(str(key).split(".")[0], names)
        if isinstance(entry, pd.DataFrame):
            linked_scalars.update(str(column).split(".")[0] for column in entry.columns)
    return lookup, linked_scalars


def summarize_patches(
    nbhd_patches: dict[str, pd.DataFrame],
    OS_pairs: dict[str, pd.DataFrame | np.ndarray] | None = None,
) -> pd.DataFrame:
    """Transform SPACE patch tables into one row per neighborhood."""
    pair_lookup, linked_scalars = _pair_lookup(OS_pairs)
    outputs: list[pd.DataFrame] = []
    for group_name, source in nbhd_patches.items():
        if not isinstance(source, pd.DataFrame):
            raise TypeError(f"Patch group {group_name!r} must be a DataFrame")
        if not {"Area", "Nbhd"}.issubset(source.columns):
            raise ValueError("Patch list must contain 'Nbhd' and 'Area' columns")
        if len(source) and (source["Nbhd"].isna().any() or source["Area"].isna().any()):
            raise ValueError("Patch Nbhd and Area values cannot be missing")
        neighborhood_count = int(source["Nbhd"].max()) if len(source) else 0
        index = pd.Index(range(1, neighborhood_count + 1), dtype=int, name="Nbhd")
        frame = source[source["Area"] > 0].copy()
        object_columns = [column for column in frame.columns if str(column).startswith("O")]
        object_column = object_columns[0] if object_columns else None
        if len(object_columns) > 1:
            raise ValueError(f"Patch group {group_name!r} contains multiple object columns")
        if object_column is None:
            if str(group_name).split(".")[0] in linked_scalars:
                continue
            scalar_columns = _natural_sort([column for column in frame.columns if str(column).startswith("S")])
            output = pd.DataFrame(index=index)
            if scalar_columns and not frame.empty:
                aggregate = frame.groupby("Nbhd", as_index=False).agg(
                    {**dict.fromkeys(scalar_columns, "sum"), "Area": "sum"}
                )
                denominator = aggregate["Area"].replace(0, np.nan)
                for column in scalar_columns:
                    aggregate[column] /= denominator
                output = aggregate.drop(columns=["Area"]).set_index("Nbhd").reindex(index, fill_value=0.0).fillna(0.0)
            outputs.append(output)
            continue

        object_output = pd.DataFrame(index=index)
        if not frame.empty:
            aggregate = frame.groupby(["Nbhd", object_column], as_index=False)["Area"].sum()
            wide = aggregate.pivot(index="Nbhd", columns=object_column, values="Area").fillna(0.0)
            wide.columns = wide.columns.map(str)
            wide = wide[_natural_sort(list(wide.columns))]
            wide = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0.0)
            wide.columns = [f"{object_column}.{column}" for column in wide.columns]
            object_output = wide.reindex(index, fill_value=0.0)

        linked_output = pd.DataFrame(index=index)
        scalar_columns = _natural_sort([column for column in frame.columns if str(column).startswith("S")])
        names = pair_lookup.get(str(group_name)) or pair_lookup.get(str(object_column))
        if names and scalar_columns and not frame.empty:
            aggregate = frame.groupby(["Nbhd", object_column], as_index=False).agg(
                {**dict.fromkeys(scalar_columns, "sum"), "Area": "sum"}
            )
            denominator = aggregate["Area"].replace(0, np.nan)
            for column in scalar_columns:
                aggregate[column] /= denominator
            unstacked = aggregate.drop(columns=["Area"]).set_index(["Nbhd", object_column])[scalar_columns].unstack()
            unstacked.columns = [
                f"{object_column}.{object_value}_{scalar}" for scalar, object_value in unstacked.columns
            ]
            unstacked = unstacked.reindex(index, fill_value=0.0).fillna(0.0)
            linked_output = unstacked.reindex(columns=[name for name in names if name in unstacked], fill_value=0.0)
        outputs.append(pd.concat([object_output, linked_output], axis=1))
    if not outputs:
        return pd.DataFrame({"Nbhd": []})
    return pd.concat(outputs, axis=1).fillna(0.0).sort_index().reset_index(drop=True)


def _os_names(
    group_name: str,
    object_column: str,
    os_pairs: dict[str, pd.DataFrame | np.ndarray] | None,
) -> tuple[str, ...] | None:
    lookup, _ = _pair_lookup(os_pairs)
    return lookup.get(group_name) or lookup.get(object_column) or lookup.get(object_column.split(".")[0])


def _fast_unit_area_randomization(
    patches: dict[str, pd.DataFrame],
    os_pairs: dict[str, pd.DataFrame | np.ndarray] | None,
    generator: np.random.Generator,
) -> pd.DataFrame | None:
    outputs: list[pd.DataFrame] = []
    for group_name, frame in patches.items():
        if len(frame) < 10_000 or not {"Area", "Nbhd"}.issubset(frame) or not np.allclose(frame["Area"], 1):
            return None
        object_columns = [column for column in frame if str(column).startswith("O")]
        if len(object_columns) != 1:
            return None
        object_column = str(object_columns[0])
        scalar_columns = _natural_sort([column for column in frame if str(column).startswith("S")])
        cache = frame.attrs.get("_pyspace_unit_area_random_cache")
        if not isinstance(cache, dict):
            neighborhood_count = int(frame["Nbhd"].max())
            sizes = np.bincount(frame["Nbhd"].to_numpy(dtype=int), minlength=neighborhood_count + 1)[1:]
            labels = sorted(pd.unique(frame[object_column]), key=lambda value: (float(value), str(value)))
            codes = np.zeros(len(frame), dtype=int)
            for code, label in enumerate(labels):
                codes[frame[object_column].to_numpy() == label] = code
            cache = {
                "neighborhood_count": neighborhood_count,
                "sizes": sizes,
                "new_neighborhood": np.repeat(np.arange(neighborhood_count), sizes),
                "labels": [str(label) for label in labels],
                "codes": codes,
                "scalars": frame[scalar_columns].to_numpy(dtype=float, copy=False)
                if scalar_columns
                else np.empty((len(frame), 0)),
                "pair_names": _os_names(str(group_name), object_column, os_pairs),
            }
            frame.attrs["_pyspace_unit_area_random_cache"] = cache
        sizes = cast(np.ndarray, cache["sizes"])
        new_neighborhood = cast(np.ndarray, cache["new_neighborhood"])
        labels = cast(list[str], cache["labels"])
        permutation = generator.permutation(len(frame))
        codes = cast(np.ndarray, cache["codes"])[permutation]
        combined = new_neighborhood * len(labels) + codes
        counts = np.bincount(combined, minlength=len(sizes) * len(labels)).reshape(len(sizes), len(labels))
        output = pd.DataFrame(index=pd.RangeIndex(len(sizes)))
        for label_index, label in enumerate(labels):
            output[f"{object_column}.{label}"] = 100 * np.divide(
                counts[:, label_index], sizes, out=np.zeros(len(sizes), dtype=float), where=sizes > 0
            )
        pair_names = cast(tuple[str, ...] | None, cache["pair_names"])
        scalars = cast(np.ndarray, cache["scalars"])
        if pair_names and scalar_columns:
            for scalar_index, scalar in enumerate(scalar_columns):
                sums = np.bincount(
                    combined,
                    weights=scalars[:, scalar_index][permutation],
                    minlength=len(sizes) * len(labels),
                ).reshape(len(sizes), len(labels))
                for label_index, label in enumerate(labels):
                    name = f"{object_column}.{label}_{scalar}"
                    if name in pair_names:
                        output[name] = np.divide(
                            sums[:, label_index],
                            counts[:, label_index],
                            out=np.zeros(len(sizes), dtype=float),
                            where=counts[:, label_index] > 0,
                        )
        outputs.append(output)
    return pd.concat(outputs, axis=1).fillna(0.0).reset_index(drop=True) if outputs else None


def random_census(
    pls: dict[str, pd.DataFrame],
    osp: dict[str, pd.DataFrame | np.ndarray] | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Shuffle whole patches, split at original neighborhood sizes, and summarize."""
    generator = rng or np.random.default_rng()
    fast = _fast_unit_area_randomization(pls, osp, generator)
    if fast is not None:
        return fast
    randomized: dict[str, pd.DataFrame] = {}
    for group_name, source in pls.items():
        if not {"Area", "Nbhd"}.issubset(source):
            raise ValueError("Patch list must contain 'Nbhd' and 'Area' columns")
        neighborhood_count = int(source["Nbhd"].max()) if len(source) else 0
        eligible = source.groupby("Nbhd", as_index=False)["Area"].sum().sort_values("Nbhd")
        neighborhood_ends = np.cumsum(eligible["Area"].to_numpy())
        frame = source[source["Area"] > 0].reset_index(drop=True)
        if frame.empty:
            randomized[group_name] = source.copy()
            continue
        frame = frame.iloc[generator.permutation(len(frame))].reset_index(drop=True)
        scalar_columns = [column for column in frame if str(column).startswith("S")]
        for column in scalar_columns:
            frame[column] /= frame["Area"]
        patch_ends = np.cumsum(frame["Area"].to_numpy())
        split_ends = np.unique(np.concatenate([patch_ends, neighborhood_ends]))
        new_sizes = np.diff(np.concatenate([[0], split_ends])).astype(float)
        source_indices = np.searchsorted(patch_ends, split_ends, side="left")
        expanded = frame.iloc[np.clip(source_indices, 0, len(frame) - 1)].copy().reset_index(drop=True)
        expanded["Area"] = new_sizes
        ending_rows = np.flatnonzero(np.isin(np.cumsum(new_sizes), neighborhood_ends))
        beginning_rows = np.array([0, *(ending_rows[:-1] + 1).tolist()])
        expanded["Nbhd"] = np.repeat(np.arange(1, len(ending_rows) + 1), ending_rows - beginning_rows + 1)
        for column in scalar_columns:
            expanded[column] *= expanded["Area"]
        maximum = int(expanded["Nbhd"].max()) if len(expanded) else 0
        if neighborhood_count > maximum:
            padding = pd.DataFrame(0, index=range(neighborhood_count - maximum), columns=expanded.columns)
            padding["Nbhd"] = np.arange(maximum + 1, neighborhood_count + 1)
            expanded = pd.concat([expanded, padding], ignore_index=True)
        randomized[group_name] = expanded
    return summarize_patches(randomized, OS_pairs=osp)


__all__ = ["random_census", "summarize_patches"]
