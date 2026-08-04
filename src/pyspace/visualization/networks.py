"""Network views kept separate from MI table selection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure


def plot_network(network: nx.Graph, *, seed: int = 0) -> Figure:
    """Render a NetworkX graph with deterministic layout coordinates."""
    if not isinstance(network, nx.Graph):
        raise TypeError("network must be a NetworkX graph")
    figure, axes = plt.subplots()
    positions = nx.spring_layout(network, seed=seed)
    nx.draw_networkx(network, positions, ax=axes, with_labels=True, node_size=350)
    axes.set_axis_off()
    figure.tight_layout()
    return figure


__all__ = ["plot_network"]
