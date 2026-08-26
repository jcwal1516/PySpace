from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyspace import alpha_diversity, beta_diversity
from pyspace.core.distributions import build_dist, r_round_column, smooth_dist, total_comp_bins


def test_round_column_covers_bin_ids_values_clipping_and_degenerate_ranges() -> None:
    values = np.array([-1.0, 0.0, 2.5, 5.0, 6.0])
    np.testing.assert_array_equal(r_round_column(values, 0, 5, 3), [1, 1, 2, 3, 3])
    np.testing.assert_array_equal(r_round_column(values, 0, 5, 3, return_bin_values=True), [0, 0, 2.5, 5, 5])
    np.testing.assert_array_equal(r_round_column(values, np.nan, 5, 3), values)
    np.testing.assert_array_equal(r_round_column(values, 0, 5, 1), values)


def test_total_composition_bins_covers_dimensions_ranges_and_validation() -> None:
    assert total_comp_bins(1, 4) == 4
    assert total_comp_bins(2, 4) == 10
    assert total_comp_bins(3, 4) == 15
    assert total_comp_bins(2, 3, [0, 0], [100, 100]) == 6

    with pytest.raises(ValueError, match="positive"):
        total_comp_bins(0, 2)
    with pytest.raises(ValueError, match="provided together"):
        total_comp_bins(2, 3, [0, 0], None)
    with pytest.raises(ValueError, match="same length"):
        total_comp_bins(2, 3, [0], [100, 100])


def test_build_distribution_preserves_group_order_nan_and_focal_filtering() -> None:
    census = pd.DataFrame({"B": [2, 1, 1, np.nan], "A": [1, 1, 1, 2], "ignored": range(4)})
    result = build_dist(census, ["A", "B"])
    assert result.columns.tolist() == ["B", "A", "freq"]
    assert result["freq"].sum() == 4
    assert result.iloc[-1]["B"] != result.iloc[-1]["B"]
    assert build_dist(census, ["A"], focal_vars=["missing"]).empty
    assert build_dist(census, ["missing"]).empty

    with pytest.raises(TypeError, match="DataFrame"):
        build_dist(np.ones((2, 2)), ["A"])
    with pytest.raises(ValueError, match="cannot be empty"):
        build_dist(census, [])


def test_smooth_distribution_validates_shape_and_returns_requested_representation() -> None:
    empty = pd.DataFrame(columns=["A", "freq"])
    assert isinstance(smooth_dist(empty, 2, pd.DataFrame([[0], [1]])), pd.DataFrame)
    assert smooth_dist(empty, 2, pd.DataFrame([[0], [1]]), full_dist=False).size == 0

    with pytest.raises(ValueError, match="freq"):
        smooth_dist(pd.DataFrame({"A": [1]}), 2, pd.DataFrame([[0], [1]]))
    with pytest.raises(ValueError, match="variable"):
        smooth_dist(pd.DataFrame({"freq": [1]}), 2, pd.DataFrame(index=[0, 1]))

    joint = pd.DataFrame({"A": [0.0], "freq": [1]})
    with pytest.raises(ValueError, match="at least one bin"):
        smooth_dist(joint, [0], pd.DataFrame([[0.0], [1.0]], columns=["A"]))
    with pytest.raises(ValueError, match="two rows"):
        smooth_dist(joint, 2, pd.DataFrame([[0.0]], columns=["A"]))

    complete = pd.DataFrame({"A": [0.0, 1.0], "freq": [3, 4]})
    probabilities = smooth_dist(
        complete,
        2,
        pd.DataFrame([[0.0], [1.0]], columns=["A"]),
        full_dist=False,
    )
    np.testing.assert_allclose(probabilities, [3 / 7, 4 / 7])


def test_smooth_distribution_masks_impossible_compositions() -> None:
    joint = pd.DataFrame({"O1.1": [0.0], "O1.2": [0.0], "freq": [1]})
    min_max = pd.DataFrame([[0.0, 0.0], [100.0, 100.0]], columns=["O1.1", "O1.2"])

    result = smooth_dist(joint, [2, 2], min_max)

    assert isinstance(result, pd.DataFrame)
    assert np.isnan(result.loc[(result["O1.1"] == 100) & (result["O1.2"] == 100), "freq"]).all()
    assert result["freq"].dropna().sum() == pytest.approx(1.0)


def test_alpha_diversity_unwraps_inputs_and_validates_palette_shape() -> None:
    composition, entropy = alpha_diversity([np.array([[1, 1]])], "O", [["red"]], plot_bkgd="B")
    assert composition["P"].tolist() == [100.0]
    assert entropy == 0.0

    zero_composition, zero_entropy = alpha_diversity(np.zeros((2, 2), dtype=int), "O", ["red"])
    assert zero_composition["P"].isna().all()
    assert zero_entropy == 0.0

    with pytest.raises(ValueError, match="one image"):
        alpha_diversity([np.ones((2, 2)), np.ones((2, 2))], "O", ["red"])
    with pytest.raises(ValueError, match="one color palette"):
        alpha_diversity(np.ones((2, 2)), "O", [["red"], ["blue"]])
    with pytest.raises(ValueError, match="at least one color"):
        alpha_diversity(np.ones((2, 2)), "O", [])
    with pytest.raises(ValueError, match="final-axis channel"):
        alpha_diversity(np.ones((2, 2, 2)), "S", ["red"])
    with pytest.raises(ValueError, match="plot_bkgd"):
        alpha_diversity(np.ones((2, 2)), "O", ["red"], plot_bkgd="transparent")


def test_beta_diversity_scalar_path_empty_parents_and_validation() -> None:
    parents = np.array([[1, 1], [2, 2]])
    scalar = np.zeros((2, 2, 2), dtype=float)
    scalar[parents == 1] = [1.0, 3.0]
    scalar[parents == 2] = [2.0, 2.0]
    composition, beta = beta_diversity(
        [parents, scalar],
        "S",
        [["parent-1", "parent-2", "unused"], ["red", "blue"]],
        plot_bkgd="B",
    )
    assert composition.shape == (3, 4)
    assert composition.loc[2, ["V1", "V2", "E"]].tolist() == [0.0, 0.0, 0.0]
    assert beta >= 0

    with pytest.raises(ValueError, match="plot_bkgd"):
        beta_diversity([parents, parents], "O", [["p"], ["v"]], plot_bkgd="x")
    with pytest.raises(ValueError, match="color palettes"):
        beta_diversity([parents, parents], "O", [["only-one"]])
    with pytest.raises(ValueError, match="at least one color"):
        beta_diversity([parents, parents], "O", [[], ["v"]])
    with pytest.raises(ValueError, match="matching spatial"):
        beta_diversity([parents, np.ones((3, 3), dtype=int)], "O", [["p1", "p2"], ["v"]])
    with pytest.raises(ValueError, match="No parent object"):
        beta_diversity(
            [np.ones((2, 2), dtype=int), np.zeros((2, 2), dtype=int)],
            "O",
            [["p"], ["v"]],
        )
