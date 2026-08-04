# R-to-Python map

The mapping below is pinned to the 24 exports in SPACE commit
`94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8`. Python-safe parameter names use
`os_pairs`, `pixel_resolution`, and `not_` for R's `OS_pairs`, `pix_res`, and
`not`. Required R inputs remain required. `census_image` and `census_table`
place required radii/sample arguments before optional inputs because Python
does not permit required positional parameters after defaulted ones.

| R SPACE export | PySpace symbol |
| --- | --- |
| `alpha_diversity` | `pyspace.alpha_diversity` |
| `beta_diversity` | `pyspace.beta_diversity` |
| `calc_vol` | `pyspace.calc_vol` |
| `calc_vols` | `pyspace.calc_vols` |
| `census_image` | `pyspace.census_image` |
| `census_table` | `pyspace.census_table` |
| `learn_pattern` | `pyspace.learn_pattern` |
| `load_image` | `pyspace.load_image` |
| `load_table` | `pyspace.load_table` |
| `make_palette` | `pyspace.make_palette` |
| `map_pattern` | `pyspace.map_pattern` |
| `measure_cisMI` | `pyspace.measure_cisMI` |
| `measure_transMI` | `pyspace.measure_transMI` |
| `merge_objects` | `pyspace.merge_objects` |
| `patch_3D` | `pyspace.patch_3D` |
| `plot_MI_radius` | `pyspace.plot_MI_radius` |
| `plot_MI_rank` | `pyspace.plot_MI_rank` |
| `plot_dist` | `pyspace.plot_dist` |
| `plot_image` | `pyspace.plot_image` |
| `plot_palette` | `pyspace.plot_palette` |
| `plot_table` | `pyspace.plot_table` |
| `standardize_censuses` | `pyspace.standardize_censuses` |
| `suggest_number` | `pyspace.suggest_number` |
| `suggest_radii` | `pyspace.suggest_radii` |

Ordinary `x`/`y` coordinate tables are intentionally separate from R SPACE's
`X`/`Y`/`Z`/`Object` contract. Use `SpacePipeline.load_table(...).census(...)`
or `pyspace.core.census.census_coordinates`; top-level `census_table` always
uses the pinned R behavior.
