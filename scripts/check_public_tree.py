#!/usr/bin/env python3
"""Reject files that do not belong in the public PySpace source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_MAX_BYTES = 1_000_000
CACHE_DIRECTORIES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis"}
IGNORED_SCAN_DIRECTORIES = {".git", ".venv", "build", "dist", "htmlcov", "site"}
PROHIBITED_DIRECTORIES = {
    "clinical data",
    "hetma",
    "if_tma",
    "manuscripts",
    "modernpathology",
    "outputs",
    "results",
    "tma",
}
PROHIBITED_SUFFIXES = {
    ".7z",
    ".bak",
    ".docx",
    ".geojson",
    ".h5",
    ".hdf5",
    ".joblib",
    ".pickle",
    ".pkl",
    ".pptx",
    ".rar",
    ".rdata",
    ".rds",
    ".sav",
    ".svs",
    ".tar",
    ".tif",
    ".tiff",
    ".tgz",
    ".xlsm",
    ".xlsx",
    ".zip",
}
PROHIBITED_FILENAMES = {".DS_Store", "Thumbs.db"}
PROHIBITED_PATH_TERMS = {"sulf1"}
DATA_MANIFEST = Path("src/pyspace/data/DATA_MANIFEST.json")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_candidates(root: Path) -> list[Path] | None:
    if not (root / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return [root / item.decode() for item in completed.stdout.split(b"\0") if item]


def candidate_files(root: Path) -> list[Path]:
    """Return tracked and prospective tracked files, excluding Git-ignored files."""
    git_files = _git_candidates(root)
    if git_files is not None:
        return sorted((path for path in git_files if path.is_file()), key=lambda path: path.as_posix())
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part in IGNORED_SCAN_DIRECTORIES for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.as_posix(),
    )


def _validate_data_manifest(root: Path, files: list[Path]) -> list[str]:
    data_root = root / DATA_MANIFEST.parent
    if not data_root.is_dir():
        return []
    manifest_path = root / DATA_MANIFEST
    if not manifest_path.is_file():
        return [f"missing packaged-data manifest: {DATA_MANIFEST.as_posix()}"]
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid packaged-data manifest: {exc}"]
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return ["invalid packaged-data manifest: files must be a list"]

    declared: dict[str, dict[str, Any]] = {}
    findings: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            findings.append("invalid packaged-data manifest entry")
            continue
        declared[entry["path"]] = entry
    actual = {
        path.relative_to(data_root).as_posix(): path
        for path in files
        if data_root in path.parents and path.name not in {"__init__.py", DATA_MANIFEST.name}
    }
    findings.extend(
        f"packaged data lacks provenance metadata: {missing}" for missing in sorted(set(actual) - set(declared))
    )
    findings.extend(
        f"packaged-data manifest names missing file: {stale}" for stale in sorted(set(declared) - set(actual))
    )
    for relative in sorted(set(actual) & set(declared)):
        entry = declared[relative]
        if entry.get("sha256") != _sha256(actual[relative]):
            findings.append(f"packaged-data checksum mismatch: {relative}")
        if entry.get("bytes") != actual[relative].stat().st_size:
            findings.append(f"packaged-data size mismatch: {relative}")
        if not entry.get("provenance") or not entry.get("license"):
            findings.append(f"packaged data lacks provenance or license: {relative}")
        if entry.get("contains_identifiers") is not False:
            findings.append(f"packaged data is not approved as identifier-free: {relative}")
    return findings


def audit_tree(root: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> list[str]:
    """Return deterministic findings for files unsafe or unsuitable for publication."""
    root_path = Path(root).resolve()
    files = candidate_files(root_path)
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(root_path)
        folded_parts = {part.casefold() for part in relative.parts}
        folded_path = relative.as_posix().casefold()
        if any(part in CACHE_DIRECTORIES for part in relative.parts):
            findings.append(f"cache directory is not publishable: {relative.as_posix()}")
        if folded_parts & PROHIBITED_DIRECTORIES:
            findings.append(f"prohibited research/output directory: {relative.as_posix()}")
        if any(term in folded_path for term in PROHIBITED_PATH_TERMS):
            findings.append(f"study-specific path is prohibited: {relative.as_posix()}")
        if path.name in PROHIBITED_FILENAMES:
            findings.append(f"prohibited generated file: {relative.as_posix()}")
        if path.suffix.casefold() in PROHIBITED_SUFFIXES:
            findings.append(f"prohibited extension: {relative.as_posix()}")
        if path.stat().st_size > max_bytes:
            findings.append(f"file exceeds {max_bytes} bytes without an approved exception: {relative.as_posix()}")
    findings.extend(_validate_data_manifest(root_path, files))
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    findings = audit_tree(args.root, max_bytes=args.max_bytes)
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        return 1
    print("Public-tree audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
