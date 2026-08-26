"""Deterministic synthetic data for documentation and smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_example_data() -> dict[str, Any]:
    """Generate the deterministic, non-clinical example census in memory."""
    positions = np.arange(20, dtype=float)
    first = np.linspace(5.0, 95.0, 20)
    base = pd.DataFrame(
        {
            "O1.1": first,
            "O1.2": 100.0 - first,
            "S1.1": np.sin(positions / 3.0) + 2.0,
            "X": positions,
            "Y": positions % 4,
            "Z": 0.0,
            "Radius": 10.0,
        }
    )
    second = base.copy()
    second["Radius"] = 20.0
    second["S1.1"] = np.cos(positions / 4.0) + 2.0
    return {
        "census": pd.concat([base, second], ignore_index=True),
        "metadata": {
            "dataset": "synthetic_tissue",
            "description": "Deterministic synthetic census; no human or clinical source data.",
            "generator": "pyspace.load_example_data",
            "radii_pixels": [10.0, 20.0],
        },
    }


__all__ = ["load_example_data"]
