# Benchmark results

Benchmarks verify semantic checksums before recording timings. They use
equivalent synthetic inputs, repeated samples, and report environment and
hardware metadata. No fixed speedup claim is part of the package contract.

Run `make benchmark` to refresh `BENCHMARK_REPORT.json` and
`BENCHMARK_REPORT.md`. The release-candidate run on 2026-08-03 used Python
3.11.14, R 4.5.2, NumPy 2.4.6, pandas 3.0.5, and an arm64 macOS host with 12
physical CPUs. Five samples were recorded per scenario:

| Scenario | Items | Python median (s) | R median (s) | R/Python |
| --- | ---: | ---: | ---: | ---: |
| Small matching distribution | 200 | 0.000450 | 0.000200 | 0.445 |
| Small non-matching distribution | 200 | 0.000463 | 0.000150 | 0.324 |
| Large matching distribution | 20,000 | 0.000735 | 0.001050 | 1.428 |
| Large non-matching distribution | 20,000 | 0.000735 | 0.001150 | 1.564 |
| Upstream tutorial census distribution | 11,763 | 0.000748 | 0.001000 | 1.337 |
| Volume batch, serial | 200 | 0.007318 | 0.019000 | 2.596 |
| Volume batch, two processes | 200 | 0.011085 | 0.011000 | 0.992 |

The distribution timings batch 20 identical operations per sample to avoid
timer-resolution zeros; the table reports time per operation. The volume rows
time one complete 200-item batch. Results from another machine are not directly
comparable. The root JSON report contains every sample, output checksum, input
provenance, and complete environment metadata.
