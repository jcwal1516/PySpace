#!/usr/bin/env python3
"""Inspect built wheels and source archives without extracting them."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

UNSAFE_SERIALIZED_SUFFIXES = {".joblib", ".pickle", ".pkl", ".rdata", ".rds", ".sav"}
CACHE_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
REQUIRED_WHEEL_SUFFIXES = {
    ".dist-info/METADATA",
    ".dist-info/entry_points.txt",
    ".dist-info/licenses/LICENSE",
    ".dist-info/licenses/NOTICE",
    "pyspace/__init__.py",
    "pyspace/data/DATA_MANIFEST.json",
    "pyspace/py.typed",
}
REQUIRED_SDIST_PATHS = {
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
    "src/pyspace/__init__.py",
    "src/pyspace/data/DATA_MANIFEST.json",
}


def inspect_member_names(names: list[str]) -> list[str]:
    """Return unsafe generic archive-member findings."""
    findings: list[str] = []
    for raw_name in names:
        normalized = raw_name.replace("\\", "/")
        member = PurePosixPath(normalized)
        if member.is_absolute() or ".." in member.parts:
            findings.append(f"unsafe archive path: {raw_name}")
        if member.suffix.casefold() in UNSAFE_SERIALIZED_SUFFIXES:
            findings.append(f"unsafe serialized object: {raw_name}")
        if member.suffix.casefold() in {".pyc", ".pyo"} or any(part in CACHE_PARTS for part in member.parts):
            findings.append(f"cache or compiled file: {raw_name}")
    return findings


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported distribution artifact: {path.name}")


def _has_suffix(names: list[str], suffix: str) -> bool:
    return any(name == suffix or name.endswith(f"/{suffix}") or name.endswith(suffix) for name in names)


def inspect_distribution(path: str | Path) -> list[str]:
    """Validate archive paths and required release metadata."""
    artifact = Path(path)
    names = _members(artifact)
    findings = inspect_member_names(names)
    required = REQUIRED_WHEEL_SUFFIXES if artifact.suffix == ".whl" else REQUIRED_SDIST_PATHS
    for expected in sorted(required):
        if not _has_suffix(names, expected):
            findings.append(f"missing required distribution member: {expected}")
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args(argv)
    failed = False
    for artifact in args.artifacts:
        findings = inspect_distribution(artifact)
        if findings:
            failed = True
            for finding in findings:
                print(f"FAIL {artifact.name}: {finding}")
        else:
            print(f"PASS {artifact.name}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
