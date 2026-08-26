"""Palette generation and display."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def make_palette(
    num_cols: int,
    *,
    random_state: int | np.random.Generator | None = None,
    random_plan: Mapping[str, Any] | None = None,
) -> list[str]:
    """Generate SPACE-style evenly spaced random HSV colors.

    ``random_plan`` can supply ``hue_offset``, ``hue_order``, ``saturation``,
    and ``value`` arrays when exact cross-language random draws are required.
    """
    if not isinstance(num_cols, int) or num_cols <= 0:
        raise ValueError("num_cols must be a positive integer")
    if random_plan is None:
        generator = (
            random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        )
        hue_offset = float(generator.uniform(0, 1 / num_cols))
        order = generator.permutation(num_cols)
        saturation = generator.beta(1.5, 1, num_cols)
        value = generator.beta(4.5, 1, num_cols)
    else:
        hue_offset = float(random_plan["hue_offset"])
        order = np.asarray(random_plan["hue_order"], dtype=int)
        saturation = np.asarray(random_plan["saturation"], dtype=float)
        value = np.asarray(random_plan["value"], dtype=float)
        if order.shape != (num_cols,) or saturation.shape != (num_cols,) or value.shape != (num_cols,):
            raise ValueError("random_plan arrays must have num_cols entries")
        if sorted(order.tolist()) != list(range(num_cols)):
            raise ValueError("hue_order must be a zero-based permutation")
    hues = (np.arange(num_cols) / num_cols + hue_offset) % 1
    hsv = np.column_stack([hues[order], saturation, value])
    rgb_colors = mcolors.hsv_to_rgb(hsv)
    return [
        mcolors.to_hex(cast(tuple[float, float, float], tuple(map(float, color))), keep_alpha=False).upper()
        for color in rgb_colors
    ]


def plot_palette(
    col_pal: Sequence[str],
    axis_label: str,
    col_labels: Sequence[str],
    vertical: bool = False,
    plot_bkgd: str = "W",
) -> Figure:
    """Render a labeled palette without changing global Matplotlib settings."""
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")
    colors = list(col_pal)
    if not colors:
        raise ValueError("col_pal cannot be empty")
    labels = list(col_labels)
    if len(labels) != len(colors):
        raise ValueError("col_labels must have one label per color")
    if vertical:
        colors.reverse()
        labels.reverse()
    background, foreground = ("black", "white") if plot_bkgd == "B" else ("white", "black")
    figure, axis = plt.subplots(figsize=(3, max(2, len(colors) * 0.45)) if vertical else (max(4, len(colors)), 2))
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)
    positions = np.arange(len(colors))
    if vertical:
        axis.barh(positions, np.ones(len(colors)), color=colors, edgecolor=foreground)
        axis.set_yticks(positions, labels, color=foreground)
        axis.set_xlabel(axis_label, color=foreground)
        axis.set_xticks([])
    else:
        axis.bar(positions, np.ones(len(colors)), color=colors, edgecolor=foreground)
        axis.set_xticks(positions, labels, color=foreground)
        axis.set_xlabel(axis_label, color=foreground)
        axis.set_yticks([])
    figure.tight_layout()
    return figure


__all__ = ["make_palette", "plot_palette"]
