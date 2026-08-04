# PySpace R/Python benchmark report

Generated: 2026-08-03T22:59:17.438935+00:00

Pinned SPACE commit: `94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8`

Outputs were compared before timing. Ratios are measurements from this environment, not a package speed claim.

| Scenario | Items | Operations/sample | Python median (s) | R median (s) | R/Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic-distribution-small-matching | 200 | 20 | 0.000450 | 0.000200 | 0.445 |
| synthetic-distribution-small-nonmatching | 200 | 20 | 0.000463 | 0.000150 | 0.324 |
| synthetic-distribution-large-matching | 20000 | 20 | 0.000735 | 0.001050 | 1.428 |
| synthetic-distribution-large-nonmatching | 20000 | 20 | 0.000735 | 0.001150 | 1.564 |
| upstream-tutorial-census-distribution | 11763 | 20 | 0.000748 | 0.001000 | 1.337 |
| synthetic-volume-batch-serial | 200 | 1 | 0.007318 | 0.019000 | 2.596 |
| synthetic-volume-batch-process | 200 | 1 | 0.011085 | 0.011000 | 0.992 |

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
