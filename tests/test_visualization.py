from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

from pyspace import plot_image, plot_MI_radius, plot_table
from pyspace.visualization.animation import animate_spatial_slices
from pyspace.visualization.interactive import interactive_mi_plot, interactive_spatial_plot
from pyspace.visualization.styling import publication_style


def test_plot_mi_radius_returns_encoded_space_table_and_labeled_figure(tmp_path: Path) -> None:
    mi = {
        "10": pd.DataFrame(
            {
                "VA": ["O1.1", "O1.2"],
                "VB": [np.nan, np.nan],
                "CisMI": [0.4, 0.2],
                "Pvalue": [0.01, 0.2],
                "Padjust": [0.02, 0.3],
                "Zscore": [2.0, 1.0],
            }
        ),
        "20": pd.DataFrame(
            {
                "VA": ["O1.1"],
                "VB": [np.nan],
                "CisMI": [0.6],
                "Pvalue": [0.001],
                "Padjust": [0.002],
                "Zscore": [3.0],
            }
        ),
    }

    data, figure = plot_MI_radius(mi, ["O1.1"], p_adj=False)

    assert data[["radius", "CisMI", "Padjust"]].to_dict(orient="list") == {
        "radius": [10.0, 20.0],
        "CisMI": [0.4, 0.6],
        "Padjust": [-2.0, -3.0],
    }
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Length Scale (um)"
    assert axis.get_ylabel() == "Corrected Log P Value"
    destination = tmp_path / "mi-radius.png"
    figure.savefig(destination)
    assert destination.stat().st_size > 0
    plt.close(figure)


def test_plot_image_encodes_object_slice_without_writing_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    image = np.array([[[[0]], [[1]]], [[[2]], [[1]]]])

    rgb, figure = plot_image(image, "O", ["#ff0000", "#0000ff"], slice={"Z": 1})

    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(rgb[0, 1], [255, 0, 0])
    assert list(tmp_path.iterdir()) == []
    plt.close(figure)


def test_plot_table_returns_normalized_values_and_figure() -> None:
    profile = pd.DataFrame({"Object": [1, 2], "Count": [3, 4], "S1": [2.0, 4.0], "S2": [10.0, 10.0]})

    normalized, figure = plot_table(profile, compare="A", normalize="U")

    np.testing.assert_allclose(normalized[["S1", "S2"]], [[0.0, 0.0], [1.0, 0.0]])
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_publication_style_is_scoped() -> None:
    original = mpl.rcParams["font.size"]
    with publication_style(font_size=17):
        assert mpl.rcParams["font.size"] == 17
    assert mpl.rcParams["font.size"] == original


def test_plotly_and_animation_features_are_real_objects() -> None:
    mi_figure = interactive_mi_plot(np.array([[0.0, 0.5], [0.5, 0.0]]), ["A", "B"])
    spatial_figure = interactive_spatial_plot(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([4.0, 5.0]))
    animation = animate_spatial_slices(np.arange(24).reshape(2, 3, 4), axis=2)

    assert isinstance(mi_figure, PlotlyFigure)
    assert isinstance(spatial_figure, PlotlyFigure)
    assert isinstance(animation, FuncAnimation)
    assert list(animation.new_frame_seq()) == [0, 1, 2, 3]
    animation.__dict__["_draw_was_started"] = True
