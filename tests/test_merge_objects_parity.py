from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from pyspace import merge_objects


def test_merge_objects_matches_space_across_corresponding_inputs() -> None:
    profile = pd.DataFrame(
        {
            "Object": [1, 2, 3],
            "Count": [10, 30, 20],
            "S1": [1.0, 3.0, 9.0],
            "S2": [5.0, 1.0, 2.0],
        }
    )
    image = np.array([[0, 1, 2], [3, 2, 1]])
    object_table = pd.DataFrame({"X": [0, 1, 2], "Y": [0, 0, 0], "Object": [1, 2, 3]})

    result = merge_objects(profile, image, ["red", "green", "blue"], object_table, [1, 2])

    pd.testing.assert_frame_equal(
        result["profile_table"],
        pd.DataFrame(
            {
                "Object": [1, 2],
                "Count": [40, 20],
                "S1": [2.5, 9.0],
                "S2": [2.0, 2.0],
            }
        ),
    )
    np.testing.assert_array_equal(result["image"], np.array([[0, 1, 1], [2, 1, 1]]))
    assert result["color_palette"] == ["red", "blue"]
    assert cast(pd.DataFrame, result["object_table"])["Object"].tolist() == [1, 1, 2]

    # Public functions must not mutate caller-owned inputs.
    assert profile["Object"].tolist() == [1, 2, 3]
    np.testing.assert_array_equal(image, np.array([[0, 1, 2], [3, 2, 1]]))


def test_merge_objects_rejects_missing_groups() -> None:
    with pytest.raises(ValueError, match="obj_groups"):
        merge_objects(None, None, None, None, [])
