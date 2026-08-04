"""Python implementation of Spatial Patterning Analysis of Cellular Ensembles."""

from __future__ import annotations

from .core.census import CensusResult, census_image, census_table, standardize_censuses
from .core.diversity import alpha_diversity, beta_diversity
from .core.merge_objects import merge_objects
from .core.networks import NetworkResult
from .core.operations import calc_vol, calc_vols, patch_3D
from .core.parameters import suggest_number, suggest_radii
from .core.pattern_learning import learn_pattern
from .core.pattern_mapping import map_pattern
from .core.pattern_models import PatternResult
from .core.r_measure_cismi import measure_cisMI
from .core.r_measure_transmi import measure_transMI
from .example_data import load_example_data
from .io.image_loader import load_image
from .io.table_loader import load_table
from .pipeline import PipelineState, SpacePipeline
from .serialization import load_result, save_result
from .visualization.distributions import plot_dist, plot_table
from .visualization.images import plot_image
from .visualization.mutual_information import (
    plot_MI_radius,
    plot_MI_rank,
)
from .visualization.palette import (
    make_palette,
    plot_palette,
)

__version__ = "0.1.0"

__all__ = [
    "CensusResult",
    "NetworkResult",
    "PatternResult",
    "PipelineState",
    "SpacePipeline",
    "__version__",
    "alpha_diversity",
    "beta_diversity",
    "calc_vol",
    "calc_vols",
    "census_image",
    "census_table",
    "learn_pattern",
    "load_example_data",
    "load_image",
    "load_result",
    "load_table",
    "make_palette",
    "map_pattern",
    "measure_cisMI",
    "measure_transMI",
    "merge_objects",
    "patch_3D",
    "plot_MI_radius",
    "plot_MI_rank",
    "plot_dist",
    "plot_image",
    "plot_palette",
    "plot_table",
    "save_result",
    "standardize_censuses",
    "suggest_number",
    "suggest_radii",
]
