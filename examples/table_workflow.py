"""Run a deterministic table census using only synthetic coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pyspace import census_table, save_result


def main(output_dir: str | Path = "example-output/table") -> dict[str, Any]:
    """Build, census, and save a tiny synthetic coordinate table."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame(
        {
            "X": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Object": [1, 1, 2, 2, 1],
        }
    )
    result = census_table(
        cells,
        radii=[1.5],
        sample_size=[3],
        seed_points=[1, 2],
        sample_indices=[0, 2, 4],
    )
    bundle = save_result(result, destination / "census.pyspace")
    return {
        "bundle": bundle,
        "rows": 0 if result.census is None else len(result.census),
        "variables": result.variables,
    }


if __name__ == "__main__":
    print(main())
