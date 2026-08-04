"""Side-effect-free static, interactive, and animated visualizations."""

from .animation import animate_spatial_slices
from .distributions import plot_dist, plot_table
from .images import plot_image, plot_spatial_map
from .interactive import interactive_mi_plot, interactive_spatial_plot
from .mutual_information import plot_MI_radius, plot_MI_rank
from .networks import plot_network
from .palette import make_palette, plot_palette
from .styling import publication_style

__all__ = [
    "animate_spatial_slices",
    "interactive_mi_plot",
    "interactive_spatial_plot",
    "make_palette",
    "plot_MI_radius",
    "plot_MI_rank",
    "plot_dist",
    "plot_image",
    "plot_network",
    "plot_palette",
    "plot_spatial_map",
    "plot_table",
    "publication_style",
]
