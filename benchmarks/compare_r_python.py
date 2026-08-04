#!/usr/bin/env python3
"""Benchmark parity-checked Python and pinned R SPACE workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from pyspace.core.distributions import build_dist
from pyspace.core.operations import calc_vol, calc_vols
from pyspace.parity import UPSTREAM_SPACE_COMMIT, check_upstream_source, pristine_upstream_checkout

ROOT = Path(__file__).resolve().parents[1]
R_DRIVER = ROOT / "benchmarks" / "benchmark_driver.R"
TUTORIAL_RELATIVE_PATH = Path("inst/doc/CEN_table.Rdata")
TUTORIAL_SHA256 = "bbd5e6763049bdd9eb372d99dd8ff07a65b51a89df734b6f44553e9137e87fa4"
DISTRIBUTION_TIMING_BATCH = 20


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _frame_checksum(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    numeric = canonical.select_dtypes(include=["number"]).columns
    for column in numeric:
        canonical[column] = canonical[column].astype(float).round(10)
    payload = canonical.to_csv(index=False, float_format="%.10g")
    return hashlib.sha256(payload.encode()).hexdigest()


def _values_checksum(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _r_version() -> str:
    completed = subprocess.run(
        ["Rscript", "-e", "cat(R.version.string)"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to determine R version: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _time(operation: Callable[[], Any], repetitions: int) -> tuple[Any, list[float]]:
    operation()
    samples: list[float] = []
    result: Any = None
    for _ in range(repetitions):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return result, samples


def _volume_worker(radius: list[float]) -> int:
    return calc_vol(radius, [100, 100, 20])


@contextmanager
def _r_source(configured: Path | None) -> Iterator[Path]:
    if configured is not None:
        failures = [check for check in check_upstream_source(configured) if not check.passed]
        if failures:
            raise RuntimeError(f"Configured R source is not the pristine pinned checkout: {failures}")
        yield configured
        return
    with pristine_upstream_checkout() as checkout:
        yield checkout


def _run_r(
    r_repo: Path,
    mode: str,
    input_path: Path,
    result_path: Path,
    metadata_path: Path,
    repetitions: int,
    *,
    parallel: bool = False,
    ensemble: list[str] | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    environment = os.environ.copy()
    if ensemble:
        environment["PYSPACE_BENCH_ENSEMBLE"] = ",".join(ensemble)
    environment["PYSPACE_BENCH_BATCH"] = str(batch_size)
    completed = subprocess.run(
        [
            "Rscript",
            str(R_DRIVER),
            str(r_repo),
            mode,
            str(input_path),
            str(result_path),
            str(metadata_path),
            str(repetitions),
            str(parallel).lower(),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"R benchmark failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _distribution_input(size: int, *, matching: bool) -> pd.DataFrame:
    object_values = np.tile(np.array([0.0, 25.0, 50.0, 75.0, 100.0]), int(np.ceil(size / 5)))[:size]
    scalar_values = object_values.copy() if matching else np.roll(object_values, max(1, size // 3))
    return pd.DataFrame({"O1.1": object_values, "S1.1": scalar_values})


def _distribution_scenario(
    name: str,
    frame: pd.DataFrame,
    ensemble: list[str],
    r_repo: Path,
    working: Path,
    repetitions: int,
    *,
    input_provenance: str,
) -> dict[str, Any]:
    input_path = working / f"{name}-input.csv"
    r_result_path = working / f"{name}-r.csv"
    r_metadata_path = working / f"{name}-r.json"
    frame.to_csv(input_path, index=False)

    def operation_batch() -> pd.DataFrame:
        result = pd.DataFrame()
        for _ in range(DISTRIBUTION_TIMING_BATCH):
            result = build_dist(frame, ensemble, "all")
        return result

    python_result, python_batch_samples = _time(operation_batch, repetitions)
    python_samples = [sample / DISTRIBUTION_TIMING_BATCH for sample in python_batch_samples]
    r_metadata = _run_r(
        r_repo,
        "distribution",
        input_path,
        r_result_path,
        r_metadata_path,
        repetitions,
        ensemble=ensemble,
        batch_size=DISTRIBUTION_TIMING_BATCH,
    )
    r_result = pd.read_csv(r_result_path)
    pd.testing.assert_frame_equal(
        python_result.reset_index(drop=True),
        r_result.reset_index(drop=True),
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )
    python_checksum = _frame_checksum(python_result)
    r_checksum = _frame_checksum(r_result)
    if python_checksum != r_checksum:
        raise RuntimeError(f"Checksum mismatch after numeric parity for {name}: {python_checksum} != {r_checksum}")
    return _scenario_record(
        name,
        python_samples,
        list(r_metadata["samples_seconds"]),
        python_checksum,
        len(frame),
        input_provenance,
        timed_operations_per_sample=DISTRIBUTION_TIMING_BATCH,
    )


def _volume_scenario(
    name: str,
    radii: list[list[float]],
    r_repo: Path,
    working: Path,
    repetitions: int,
    *,
    parallel: bool,
) -> dict[str, Any]:
    input_path = working / f"{name}-input.json"
    r_result_path = working / f"{name}-r.json"
    r_metadata_path = working / f"{name}-metadata.json"
    input_path.write_text(json.dumps(radii), encoding="utf-8")
    if parallel:
        with ProcessPoolExecutor(max_workers=2) as executor:
            operation = lambda: list(executor.map(_volume_worker, radii))  # noqa: E731 - timed closure.
            python_result, python_samples = _time(operation, repetitions)
    else:
        python_result, python_samples = _time(lambda: calc_vols(radii, [100, 100, 20]), repetitions)
    r_metadata = _run_r(
        r_repo,
        "volume",
        input_path,
        r_result_path,
        r_metadata_path,
        repetitions,
        parallel=parallel,
    )
    r_result = [int(value) for value in json.loads(r_result_path.read_text(encoding="utf-8"))]
    if python_result != r_result:
        raise RuntimeError(f"Volume output mismatch for {name}")
    checksum = _values_checksum(python_result)
    return _scenario_record(
        name,
        python_samples,
        list(r_metadata["samples_seconds"]),
        checksum,
        len(radii),
        "Synthetic radius batch; identical ordered vectors in R and Python.",
        timed_operations_per_sample=1,
    )


def _scenario_record(
    name: str,
    python_samples: list[float],
    r_samples: list[float],
    checksum: str,
    input_rows: int,
    provenance: str,
    *,
    timed_operations_per_sample: int,
) -> dict[str, Any]:
    python_median = statistics.median(python_samples)
    r_median = statistics.median(r_samples)
    return {
        "name": name,
        "input_rows_or_items": input_rows,
        "input_provenance": provenance,
        "timed_operations_per_sample": timed_operations_per_sample,
        "output_checksum": checksum,
        "parity_verified_before_timing": True,
        "python_samples_seconds": python_samples,
        "r_samples_seconds": r_samples,
        "python_median_seconds": python_median,
        "r_median_seconds": r_median,
        "measured_r_over_python_ratio": r_median / python_median if python_median else None,
    }


def _load_tutorial_census(r_repo: Path, destination: Path) -> pd.DataFrame:
    source = r_repo / TUTORIAL_RELATIVE_PATH
    if _sha256(source) != TUTORIAL_SHA256:
        raise RuntimeError("Pinned tutorial census checksum does not match the reviewed artifact")
    expression = (
        f"load({json.dumps(str(source))}); write.csv(CEN_table, {json.dumps(str(destination))}, row.names=FALSE)"
    )
    completed = subprocess.run(["Rscript", "-e", expression], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to export pinned tutorial census:\n{completed.stderr}")
    return pd.read_csv(destination)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PySpace R/Python benchmark report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Pinned SPACE commit: `{report['upstream_space_commit']}`",
        "",
        "Outputs were compared before timing. Ratios are measurements from this environment, "
        "not a package speed claim.",
        "",
        "| Scenario | Items | Operations/sample | Python median (s) | R median (s) | R/Python |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            f"| {scenario['name']} | {scenario['input_rows_or_items']} | "
            f"{scenario['timed_operations_per_sample']} | "
            f"{scenario['python_median_seconds']:.6f} | {scenario['r_median_seconds']:.6f} | "
            f"{scenario['measured_r_over_python_ratio']:.3f} |"
        )
        for scenario in report["scenarios"]
    )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(report["environment"], indent=2, sort_keys=True),
            "```",
            "",
            "See `BENCHMARK_REPORT.json` for samples, checksums, and provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmarks(r_repo: Path, repetitions: int) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pyspace-benchmark-") as temporary:
        working = Path(temporary)
        for size_name, size in (("small", 200), ("large", 20_000)):
            for relation in ("matching", "nonmatching"):
                frame = _distribution_input(size, matching=relation == "matching")
                scenarios.append(
                    _distribution_scenario(
                        f"synthetic-distribution-{size_name}-{relation}",
                        frame,
                        ["O1.1", "S1.1"],
                        r_repo,
                        working,
                        repetitions,
                        input_provenance="Deterministic synthetic object/scalar abundance table.",
                    )
                )
        tutorial = _load_tutorial_census(r_repo, working / "tutorial-census.csv")
        scenarios.append(
            _distribution_scenario(
                "upstream-tutorial-census-distribution",
                tutorial,
                ["O1.1", "O1.2"],
                r_repo,
                working,
                repetitions,
                input_provenance=(
                    f"Runtime-only {TUTORIAL_RELATIVE_PATH.as_posix()} from the pinned Apache-2.0 upstream; "
                    f"reviewed SHA-256 {TUTORIAL_SHA256}; 18 numeric SPACE census columns and no identifier column."
                ),
            )
        )
        radii = [[float(1 + index % 12), float(1 + (index * 3) % 12), float(1 + index % 4)] for index in range(200)]
        scenarios.append(
            _volume_scenario("synthetic-volume-batch-serial", radii, r_repo, working, repetitions, parallel=False)
        )
        scenarios.append(
            _volume_scenario("synthetic-volume-batch-process", radii, r_repo, working, repetitions, parallel=True)
        )

    return {
        "schema": "pyspace-r-python-benchmark",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream_space_commit": UPSTREAM_SPACE_COMMIT,
        "repetitions": repetitions,
        "environment": {
            "python": sys.version,
            "r": _r_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "scenarios": scenarios,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-repo", type=Path, default=Path(os.environ["SPACE_R_REPO"]) if "SPACE_R_REPO" in os.environ else None
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--json-output", type=Path, default=ROOT / "BENCHMARK_REPORT.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "BENCHMARK_REPORT.md")
    args = parser.parse_args(argv)
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    with _r_source(args.r_repo) as r_repo:
        report = run_benchmarks(r_repo, args.repetitions)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.json_output} and {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
