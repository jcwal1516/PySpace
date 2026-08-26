from __future__ import annotations

import numpy as np

from pyspace import suggest_number, suggest_radii


def test_suggest_radii_matches_pinned_r_algorithm() -> None:
    assert suggest_radii([10.0, 20.0], {"x": 0.284, "y": 0.284, "z": 0.5}) == {
        10.0: [35, 35, 20],
        20.0: [70, 70, 40],
    }


def test_suggest_number_uses_foreground_pixels_and_discrete_volumes() -> None:
    images = np.ones((5, 5, 1), dtype=np.uint8)
    radii = {1.0: [1, 1, 1], 2.0: [2, 2, 1]}

    counts = suggest_number(2.0, radii, images)

    # Exactly ceiling(coverage * non-background pixels / calc_vol(radius)).
    assert counts == [10, 4]


def test_suggest_number_drops_linked_scalar_maps() -> None:
    object_map = np.ones((3, 3, 1), dtype=np.uint8)
    linked_scalar = np.full((3, 3, 1), 100, dtype=np.uint8)

    linked = suggest_number(1.0, {1.0: [1, 1, 1]}, {"O1": object_map, "S1": linked_scalar})
    object_only = suggest_number(1.0, {1.0: [1, 1, 1]}, object_map)

    assert linked == object_only
