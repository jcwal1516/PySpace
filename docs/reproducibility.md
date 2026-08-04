# Reproducibility

PySpace does not mutate NumPy's global random state. Stochastic public APIs
accept `random_state`; repeated calls with the same integer and inputs are
repeatable within the supported Python/NumPy environment.

Cross-language tests avoid assuming that NumPy and R generate the same stream.
Use explicit plans at the relevant boundary:

- census: `seed_points`, `seed_indices`, or `sample_indices`;
- cisMI: `random_censuses`;
- transMI: `permutation_indices`;
- SOM: initialization weights and epoch orders in the explicit random-plan
  arguments.

Record the PySpace version, upstream SPACE commit, dependency lock or installed
versions, platform, input checksums, and every random plan with an analysis.
`cores=1` selects serial execution; larger positive values select ordered
standard-library workers where the pinned algorithm has independent work.
Parallel backends preserve result ordering, but floating-point libraries may
still differ at the documented iterative tolerance.
