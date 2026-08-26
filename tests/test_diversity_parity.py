from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyspace import alpha_diversity, beta_diversity


def test_alpha_diversity_matches_space_object_composition() -> None:
    image = np.array([[0, 1], [1, 2]])

    composition, entropy = alpha_diversity(image, "O", ["#111111", "#eeeeee"])

    pd.testing.assert_frame_equal(
        composition,
        pd.DataFrame({"V": [1, 2], "P": [200 / 3, 100 / 3]}),
    )
    assert entropy == pytest.approx(0.9182958340544896)


def test_alpha_diversity_matches_space_scalar_channel_sums() -> None:
    image = np.zeros((2, 2, 1, 2), dtype=float)
    image[..., 0] = np.array([[[1.0], [2.0]], [[0.0], [1.0]]])
    image[..., 1] = np.array([[[0.0], [2.0]], [[2.0], [2.0]]])

    composition, entropy = alpha_diversity(image, "S", ["red", "blue"])

    pd.testing.assert_frame_equal(composition, pd.DataFrame({"V": [1, 2], "P": [40.0, 60.0]}))
    assert entropy == pytest.approx(0.9709505944546686)


def test_beta_diversity_matches_space_weighted_kl_definition() -> None:
    parents = np.array([[1, 1], [2, 2]])
    constituents = np.array([[1, 2], [1, 1]])

    composition, beta = beta_diversity(
        [parents, constituents],
        "O",
        [["black", "white"], ["red", "blue"]],
    )

    expected = pd.DataFrame(
        {
            "O": [1, 2],
            "V1": [50.0, 100.0],
            "V2": [50.0, 0.0],
            "E": [1.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(composition, expected)

    tiny = 0.000001 / 2
    region_1 = np.array([0.5, 0.5])
    region_2 = np.array([1.0, tiny])
    region_2 /= region_2.sum()
    average = np.array([0.75, 0.25])
    expected_beta = 0.5 * np.sum(region_1 * np.log2(region_1 / average))
    expected_beta += 0.5 * np.sum(region_2 * np.log2(region_2 / average))
    assert beta == pytest.approx(expected_beta)


@pytest.mark.parametrize("image_type", ["", "object", "X"])
def test_diversity_rejects_unknown_image_type(image_type: str) -> None:
    with pytest.raises(ValueError, match="img_type must be 'O' or 'S'"):
        alpha_diversity(np.ones((2, 2)), image_type, ["red"])


def test_beta_diversity_requires_exactly_two_images_and_palettes() -> None:
    with pytest.raises(ValueError, match="Exactly two images"):
        beta_diversity([np.ones((2, 2))], "O", [["red"]])
