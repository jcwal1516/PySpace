"""Round-trip tables and arrays through the non-pickle result format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyspace import load_result, save_result


def main(output_dir: str | Path = "example-output/bundle") -> dict[str, Any]:
    """Save and reload a small, entirely synthetic result payload."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / "result.pyspace"
    save_result(
        {
            "summary": pd.DataFrame({"radius": [10.0], "score": [0.25]}),
            "matrix": np.eye(2, dtype=float),
        },
        bundle,
    )
    loaded = load_result(bundle)
    return {"bundle": bundle, "keys": sorted(loaded)}


if __name__ == "__main__":
    print(main())
