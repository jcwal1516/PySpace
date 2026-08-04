#!/usr/bin/env python3
"""Enforce branch coverage for the parity primitives and public I/O kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KERNEL_FILES = (
    "src/pyspace/core/census.py",
    "src/pyspace/core/census_models.py",
    "src/pyspace/core/census_sampling.py",
    "src/pyspace/core/distributions.py",
    "src/pyspace/core/diversity.py",
    "src/pyspace/core/image_census.py",
    "src/pyspace/core/operations.py",
    "src/pyspace/core/patch_measurements.py",
    "src/pyspace/core/patch_summary.py",
    "src/pyspace/core/table_census.py",
    "src/pyspace/io/image_loader.py",
    "src/pyspace/io/table_loader.py",
    "src/pyspace/serialization.py",
)


def kernel_coverage(report: dict[str, Any], files: tuple[str, ...] = KERNEL_FILES) -> float:
    """Return combined line-and-branch coverage for the declared kernel."""
    coverage_files = report.get("files")
    if not isinstance(coverage_files, dict):
        raise ValueError("Coverage report has no files object")
    normalized_files = {str(path).replace("\\", "/"): details for path, details in coverage_files.items()}
    missing = [path for path in files if path not in normalized_files]
    if missing:
        raise ValueError(f"Coverage report is missing kernel files: {', '.join(missing)}")

    covered = 0
    total = 0
    for path in files:
        summary = normalized_files[path].get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"Coverage report has no summary for {path}")
        covered += int(summary["covered_lines"]) + int(summary["covered_branches"])
        total += int(summary["num_statements"]) + int(summary["num_branches"])
    if total == 0:
        raise ValueError("Coverage report contains no measurable kernel statements or branches")
    return 100.0 * covered / total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=Path("coverage.json"))
    parser.add_argument("--minimum", type=float, default=90.0)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    percentage = kernel_coverage(report)
    if percentage < args.minimum:
        print(f"Kernel coverage {percentage:.2f}% is below {args.minimum:.2f}%")
        return 1
    print(f"Kernel coverage {percentage:.2f}% meets {args.minimum:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
