"""Purposeful Plotly views for MI matrices and spatial values."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


def interactive_mi_plot(matrix: np.ndarray, labels: Sequence[str]) -> go.Figure:
    """Create an interactive labeled MI heatmap."""
    values = np.asarray(matrix, dtype=float)
    names = [str(label) for label in labels]
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if len(names) != values.shape[0]:
        raise ValueError("labels must match the matrix dimensions")
    figure = go.Figure(go.Heatmap(z=values, x=names, y=names, colorscale="Viridis", colorbar={"title": "MI"}))
    figure.update_layout(xaxis_title="Variable", yaxis_title="Variable")
    return figure


def interactive_spatial_plot(coordinates: np.ndarray, values: np.ndarray) -> go.Figure:
    """Create an interactive spatial scatter plot with value-aware hover text."""
    points = np.asarray(coordinates, dtype=float)
    measurements = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("coordinates must have shape (n, 2) or (n, 3)")
    if measurements.shape != (len(points),):
        raise ValueError("values must contain one number per coordinate")
    marker = {"color": measurements, "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "Value"}}
    if points.shape[1] == 3:
        trace: go.BaseTraceType = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", marker=marker
        )
    else:
        trace = go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", marker=marker)
    figure = go.Figure(trace)
    figure.update_layout(xaxis_title="X", yaxis_title="Y")
    return figure


__all__ = ["interactive_mi_plot", "interactive_spatial_plot"]
