from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation

from pyspace import make_palette, plot_dist, plot_image, plot_MI_radius, plot_MI_rank, plot_palette, plot_table
from pyspace.visualization.animation import animate_spatial_slices
from pyspace.visualization.images import plot_spatial_map
from pyspace.visualization.interactive import interactive_mi_plot, interactive_spatial_plot
from pyspace.visualization.networks import plot_network


def _census() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "O1.1": [0.0, 25.0, 50.0, 75.0, 100.0, 50.0],
            "S1.1": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "O2.1": [100.0, 75.0, 50.0, 25.0, 0.0, 50.0],
            "X": np.arange(6),
            "Y": np.zeros(6),
            "Z": np.zeros(6),
            "Radius": [10.0] * 6,
        }
    )


def test_distribution_plots_cover_histogram_scatter_smoothing_and_large_ensembles() -> None:
    census = _census()
    one, one_figure = plot_dist(census, [1], [10.0], plot_bkgd="B")
    two, two_figure = plot_dist(census, ["O1.1", "S1.1"], 10.0, plot_zoom=True)
    smooth, smooth_figure = plot_dist(census, ["O1.1", "S1.1"], 10.0, bin_num=3)
    raw, no_figure = plot_dist(census, ["O1.1", "S1.1", "O2.1"], 10.0)

    assert list(one.columns) == ["O1.1"]
    assert two_figure is not None
    assert two_figure.axes[0].get_xlim() != (0.0, 100.0)
    assert "freq" in smooth
    assert len(raw) == len(census)
    assert no_figure is None
    for figure in (one_figure, two_figure, smooth_figure):
        assert figure is not None
        plt.close(figure)


@pytest.mark.parametrize(("compare", "normalize"), [("A", "U"), ("W", "U"), ("B", "Z")])
def test_profile_table_modes(compare: str, normalize: str) -> None:
    profile = pd.DataFrame({"Object": [1, 2, 3], "Count": [4, 5, 6], "S1": [1.0, 2.0, 4.0], "S2": [8.0, 4.0, 2.0]})
    normalized, figure = plot_table(profile, compare=compare, normalize=normalize, tile_plots=True, plot_bkgd="B")

    assert normalized.shape == (3, 3)
    assert len(figure.axes) >= 2
    plt.close(figure)


def test_palette_random_plan_and_vertical_rendering() -> None:
    plan = {
        "hue_offset": 0.1,
        "hue_order": [2, 0, 1],
        "saturation": [0.5, 0.6, 0.7],
        "value": [0.8, 0.9, 1.0],
    }
    palette = make_palette(3, random_plan=plan)
    figure = plot_palette(palette, "Objects", ["A", "B", "C"], vertical=True, plot_bkgd="B")

    assert len(palette) == 3
    assert all(color.startswith("#") and len(color) == 7 for color in palette)
    assert figure.axes[0].get_xlabel() == "Objects"
    plt.close(figure)


def test_scalar_image_spatial_and_network_plots() -> None:
    scalar = np.zeros((3, 3, 1, 2), dtype=float)
    scalar[:, :, 0, 0] = np.arange(9).reshape(3, 3)
    scalar[:, :, 0, 1] = 1
    encoded, image_figure = plot_image(
        scalar,
        "S",
        ["#ff0000", "#00ff00"],
        scalars=[1, 2],
        enh_cnt=0.2,
    )
    spatial_figure = plot_spatial_map(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([2.0, 3.0]))
    network_figure = plot_network(nx.path_graph(3), seed=2)

    assert encoded.shape == (3, 3, 3)
    assert len(spatial_figure.axes) == 2
    assert len(network_figure.axes) == 1
    for figure in (image_figure, spatial_figure, network_figure):
        plt.close(figure)


def test_mi_rank_and_group_radius_plots() -> None:
    frame = pd.DataFrame(
        {
            "VA": ["O1.1", "O1.1", "O2.1", "S1.1"],
            "VB": [np.nan, "S1.1", "S1.1", np.nan],
            "CisMI": [0.2, 0.5, 0.3, 0.15],
            "Pvalue": [0.01, 0.001, 0.02, 0.03],
            "Padjust": [0.02, 0.002, 0.03, 0.04],
            "Zscore": [2.0, 4.0, 3.0, 1.5],
        }
    )
    (ranked, aggregate), rank_figure = plot_MI_rank(
        {"10": frame},
        10.0,
        depth=[2],
        col_pals=["red", "blue"],
        p_adj=True,
        all=["O1.1"],
        alo=["S1.1"],
        not_=["never"],
        plot_bkgd="B",
    )
    group_frame = pd.DataFrame(
        {
            "VA": ["O1.1"],
            "VB": [np.nan],
            "TransMI_A": [0.4],
            "Pvalue_A": [0.01],
            "Padjust_A": [0.02],
            "Zscore_A": [2.5],
        }
    )
    radius_data, radius_figure = plot_MI_radius({"10": group_frame}, ["O1.1"], p_adj=True, group="A")

    assert not ranked.empty
    assert set(aggregate["V"]) == {"O1.1", "S1.1"}
    assert radius_data.loc[0, "TransMI_A"] == 0.4
    plt.close(rank_figure)
    plt.close(radius_figure)


def test_interactive_3d_and_animation_axes() -> None:
    figure = interactive_spatial_plot(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        np.array([1.0, 2.0]),
    )
    animation = animate_spatial_slices(np.arange(24).reshape(2, 3, 4), axis=0, interval=10)

    assert figure.data[0].type == "scatter3d"
    assert isinstance(animation, FuncAnimation)
    assert list(animation.new_frame_seq()) == [0, 1]
    animation.__dict__["_draw_was_started"] = True
    plt.close(animation._fig)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: plot_dist(_census(), [], 10), "empty"),
        (lambda: plot_dist(_census(), [99], 10), "outside"),
        (lambda: plot_dist(_census(), ["missing"], 10), "Unknown"),
        (lambda: plot_dist(_census(), ["O1.1"], 10, bin_num=0), "positive"),
        (lambda: plot_table(pd.DataFrame({"x": [1]})), "requires"),
        (lambda: make_palette(0), "positive"),
        (lambda: plot_palette([], "", [], plot_bkgd="W"), "empty"),
        (lambda: interactive_mi_plot(np.ones((2, 3)), ["A", "B"]), "square"),
        (lambda: interactive_spatial_plot(np.ones((2, 4)), np.ones(2)), "shape"),
        (lambda: plot_network("not-a-graph"), "NetworkX"),
        (lambda: animate_spatial_slices(np.ones((2, 2)), axis=0), "three-dimensional"),
    ],
)
def test_visualization_validation_errors(call: object, match: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((IndexError, TypeError, ValueError), match=match):
            assert callable(call)
            call()
