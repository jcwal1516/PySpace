from __future__ import annotations

import numpy as np
import pandas as pd

from pyspace.core.census import create_neighborhoods
from pyspace.core.pattern_models import SelfOrganizingMap
from pyspace.visualization.palette import make_palette


def _assert_global_rng_unchanged(operation) -> None:
    np.random.seed(8675309)
    expected = np.random.random(5)
    np.random.seed(8675309)

    operation()

    np.testing.assert_array_equal(np.random.random(5), expected)


def test_seeded_palette_is_local_and_repeatable() -> None:
    first: list[str] = []

    def generate() -> None:
        first.extend(make_palette(8, random_state=42))

    _assert_global_rng_unchanged(generate)
    assert first == make_palette(8, random_state=42)


def test_som_random_state_is_local_and_repeatable() -> None:
    data = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.25, 0.75]])
    first: list[np.ndarray] = []

    def train() -> None:
        model = SelfOrganizingMap(grid_size=4, max_iterations=3, random_seed=7).fit(data)
        assert model.weights is not None
        first.append(model.weights.copy())

    _assert_global_rng_unchanged(train)
    second = SelfOrganizingMap(grid_size=4, max_iterations=3, random_seed=7).fit(data)
    assert second.weights is not None
    np.testing.assert_array_equal(first[0], second.weights)


def test_census_random_state_is_local_and_repeatable() -> None:
    coordinates = pd.DataFrame({"x": np.arange(8, dtype=float), "y": np.zeros(8)})
    first: list[list[tuple[float, ...]]] = []

    def census() -> None:
        result = create_neighborhoods(
            coordinates,
            radii=[1.0],
            max_neighborhoods=4,
            random_state=19,
        )
        first.append([neighborhood.center for neighborhood in result[1.0]])

    _assert_global_rng_unchanged(census)
    repeated = create_neighborhoods(
        coordinates,
        radii=[1.0],
        max_neighborhoods=4,
        random_state=19,
    )
    assert first[0] == [neighborhood.center for neighborhood in repeated[1.0]]
