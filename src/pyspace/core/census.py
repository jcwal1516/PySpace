"""Public census façade over cohesive image, sampling, and table modules."""

from __future__ import annotations

import pandas as pd

from .census_models import CensusResult, Neighborhood
from .census_sampling import create_neighborhoods
from .image_census import census_image
from .table_census import census_coordinates, census_table, standardize_censuses


def export_multi_image_census_dataframe(
    neighborhoods: list[Neighborhood],
    variables: list[str],
    include_coordinates: bool = True,
    r_compatible: bool = True,
) -> pd.DataFrame:
    """Export measured neighborhoods with SPACE variable/coordinate ordering."""
    rows: list[dict[str, float]] = []
    for neighborhood in neighborhoods:
        measurements = neighborhood.variable_measurements or {}
        row = {name: float(measurements.get(name, 0.0)) for name in variables}
        if include_coordinates:
            offset = 1 if r_compatible else 0
            row["X"] = float(neighborhood.center[0] + offset)
            row["Y"] = float(neighborhood.center[1] + offset)
            row["Z"] = float((neighborhood.center[2] if len(neighborhood.center) > 2 else 0) + offset)
            row["Radius"] = float(neighborhood.radius)
        rows.append(row)
    columns = list(variables)
    if include_coordinates:
        columns.extend(["X", "Y", "Z", "Radius"])
    return pd.DataFrame(rows, columns=pd.Index(columns))


__all__ = [
    "CensusResult",
    "Neighborhood",
    "census_image",
    "census_coordinates",
    "census_table",
    "create_neighborhoods",
    "export_multi_image_census_dataframe",
    "standardize_censuses",
]
