# Python API

## R-compatible surface

The pinned R-compatible functions are `alpha_diversity`, `beta_diversity`,
`calc_vol`, `calc_vols`, `census_image`, `census_table`, `learn_pattern`,
`load_image`, `load_table`, `make_palette`, `map_pattern`, `measure_cisMI`,
`measure_transMI`, `merge_objects`, `patch_3D`, `plot_MI_radius`,
`plot_MI_rank`, `plot_dist`, `plot_image`, `plot_palette`, `plot_table`,
`standardize_censuses`, `suggest_number`, and `suggest_radii`.

## Python workflow surface

The additive Python layer consists of `CensusResult`, `NetworkResult`,
`PatternResult`, `PipelineState`, `SpacePipeline`, `load_example_data`,
`load_result`, `save_result`, and `__version__`.

For generic coordinate tables, use
`pyspace.core.census.census_coordinates`. It is explicitly Python-only and
records `r_parity_mode="non_parity_python_coordinates"` in result metadata;
`pyspace.census_table` remains the R-compatible export.

::: pyspace
    options:
      members_order: source
      show_root_heading: true
      show_signature_annotations: true
      separate_signature: true

::: pyspace.core.census.census_coordinates
    options:
      show_root_heading: true
      show_signature_annotations: true
      separate_signature: true
