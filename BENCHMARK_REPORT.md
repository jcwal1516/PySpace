# PySpace R/Python benchmark report

Generated: 2026-08-04T02:10:15.072914+00:00

Pinned SPACE commit: `94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8`

Outputs were compared before timing. Ratios are measurements from this environment, not a package speed claim.

| Scenario | Items | Operations/sample | Python median (s) | R median (s) | R/Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic-distribution-small-matching | 200 | 20 | 0.000487 | 0.000200 | 0.411 |
| synthetic-distribution-small-nonmatching | 200 | 20 | 0.000477 | 0.000200 | 0.419 |
| synthetic-distribution-large-matching | 20000 | 20 | 0.000775 | 0.001100 | 1.420 |
| synthetic-distribution-large-nonmatching | 20000 | 20 | 0.000769 | 0.001150 | 1.496 |
| upstream-tutorial-census-distribution | 11763 | 20 | 0.000797 | 0.001100 | 1.379 |
| synthetic-volume-batch-serial | 200 | 1 | 0.007642 | 0.019000 | 2.486 |
| synthetic-volume-batch-process | 200 | 1 | 0.010815 | 0.010000 | 0.925 |

## Environment

```json
{
  "logical_cpus": 12,
  "machine": "arm64",
  "memory_bytes": 51539607552,
  "numpy": "2.4.6",
  "pandas": "3.0.5",
  "physical_cpus": 12,
  "platform": "macOS-26.5.2-arm64-arm-64bit",
  "processor": "arm",
  "python": "3.11.14 (main, Oct  9 2025, 16:16:55) [Clang 17.0.0 (clang-1700.3.19.1)]",
  "r": "R version 4.5.2 (2025-10-31)"
}
```

See `BENCHMARK_REPORT.json` for samples, checksums, and provenance.
