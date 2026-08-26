from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyspace import learn_pattern, map_pattern


def _census() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "O1.1": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "S1.1": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "X": [1, 1, 1, 2, 2, 2],
            "Y": [1, 2, 3, 1, 2, 3],
            "Z": [1] * 6,
            "Radius": [1.0] * 6,
        }
    )


def test_learn_pattern_returns_space_covariation_schema_deterministically() -> None:
    first = learn_pattern(
        _census(),
        ["O1.1", "S1.1"],
        1.0,
        {"O1": ["red"], "S1": ["blue"]},
        som_reps=2,
        smooth_window=2,
        random_state=12,
    )
    second = learn_pattern(
        _census(),
        ["O1.1", "S1.1"],
        1.0,
        {"O1": ["red"], "S1": ["blue"]},
        som_reps=2,
        smooth_window=2,
        random_state=12,
    )

    assert list(first.columns) == ["X", "Y", "Ymin", "Ymax", "V", "Enr", "Y_norm", "Ymin_norm", "Ymax_norm"]
    assert first.shape == (12, 9)
    assert first["V"].tolist() == ["O1.1"] * 6 + ["S1.1"] * 6
    assert first.groupby("V", sort=False)["X"].apply(list).tolist() == [
        list(np.linspace(0, 100, 6)),
        list(np.linspace(0, 100, 6)),
    ]
    pd.testing.assert_frame_equal(first, second)


def test_learn_pattern_validates_radius_and_variables() -> None:
    with pytest.raises(ValueError, match="radius"):
        learn_pattern(_census(), ["O1.1"], 9.0, {"O1": ["red"]})
    with pytest.raises(ValueError, match="ensemble"):
        learn_pattern(_census(), ["missing"], 1.0, {"O1": ["red"]})


def test_map_pattern_votes_in_space_coordinates_and_preserves_background() -> None:
    covariation = pd.DataFrame(
        {
            "X": [0.0, 50.0, 100.0, 0.0, 50.0, 100.0],
            "Y": [0.0, 50.0, 100.0, 0.0, 0.0, 0.0],
            "V": ["O1.1"] * 3 + ["S1.1"] * 3,
        }
    )
    census = pd.DataFrame({"O1.1": [0.0], "S1.1": [0.0], "X": [2], "Y": [2], "Z": [1], "Radius": [1.0]})
    original = np.ones((3, 3, 1, 1), dtype=np.uint8)
    original[0, 0, 0, 0] = 0

    image, palette = map_pattern(
        covariation,
        [[0.0, 40.0], [60.0, 100.0]],
        {"O1": original},
        census,
        1.0,
        {1.0: [1, 1, 1]},
        ["red", "blue"],
    )

    assert image.shape == (3, 3, 1, 1)
    assert image[1, 1, 0, 0] == 1
    assert image[0, 0, 0, 0] == 0
    assert palette == ["red", "blue"]
