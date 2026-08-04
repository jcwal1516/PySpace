from __future__ import annotations

import numpy as np
import pandas as pd


def transmi_inputs() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Return the shared non-degenerate transMI parity case."""
    patterns = (
        (
            np.concatenate([np.repeat(20.0, 20), np.repeat(80.0, 20)]),
            np.concatenate([np.repeat(1.0, 20), np.repeat(4.0, 20)]),
        ),
        (
            np.concatenate([np.repeat(20.0, 10), np.repeat(80.0, 30)]),
            np.concatenate([np.repeat(1.0, 30), np.repeat(4.0, 10)]),
        ),
        (np.tile(np.array([20.0, 80.0]), 20), np.tile(np.array([1.0, 4.0]), 20)),
        (np.tile(np.array([20.0, 20.0, 80.0, 80.0]), 10), np.tile(np.array([1.0, 4.0, 1.0, 4.0]), 10)),
    )
    censuses = [
        pd.DataFrame(
            {
                "O1.1": o11,
                "O1.2": 100.0 - o11,
                "S1.1": s11,
                "O1.1_S1.1": o11 * s11 / 100.0,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "Radius": 1.1,
            }
        )
        for o11, s11 in patterns
    ]
    return censuses, pd.DataFrame({"Status": ["A", "A", "B", "B"]})
