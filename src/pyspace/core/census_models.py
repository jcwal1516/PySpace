"""Typed results shared by image and table census implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Neighborhood:
    """One sampled spatial neighborhood."""

    center: tuple[float, ...]
    radius: float
    points: np.ndarray
    object_ids: np.ndarray | None = None
    intensities: np.ndarray | None = None
    patch_id: str | None = None
    is_3d: bool = False
    variable_measurements: dict[str, float] | None = None


@dataclass
class CensusResult:
    """Census table, source patches, sampled neighborhoods, and provenance."""

    neighborhoods: list[Neighborhood]
    metadata: dict[str, Any]
    variables: list[str]
    summary_stats: dict[str, Any] | None = None
    biomolecule_pairings: dict[str, Any] | None = None
    patch_list: dict[str, Any] | None = None
    census: pd.DataFrame | None = None


def summarize_neighborhoods(neighborhoods: list[Neighborhood]) -> dict[str, Any]:
    """Return descriptive counts without probing hardware or mutating state."""
    if not neighborhoods:
        return {"total_neighborhoods": 0, "total_points": 0}
    sizes = np.asarray([len(neighborhood.points) for neighborhood in neighborhoods], dtype=float)
    return {
        "total_neighborhoods": len(neighborhoods),
        "total_points": int(sizes.sum()),
        "mean_patch_size": float(sizes.mean()),
        "median_patch_size": float(np.median(sizes)),
        "std_patch_size": float(sizes.std()),
        "min_patch_size": int(sizes.min()),
        "max_patch_size": int(sizes.max()),
    }


__all__ = ["CensusResult", "Neighborhood", "summarize_neighborhoods"]
