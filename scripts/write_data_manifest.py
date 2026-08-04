#!/usr/bin/env python3
"""Create the checksum, provenance, and identifier audit for packaged data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "DATA_MANIFEST.json"
UPSTREAM_COMMIT = "94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _provenance(relative: str) -> str:
    if relative.startswith("parity/upstream/"):
        return f"Verbatim metadata from eschrom/SPACE commit {UPSTREAM_COMMIT}."
    if relative == "parity/random_plans.json":
        return "Hand-authored synthetic random plan shared by R and Python parity tests."
    return (
        f"Synthetic oracle generated from eschrom/SPACE commit {UPSTREAM_COMMIT} "
        "with scripts/generate_core_parity_oracles.R."
    )


def build_manifest(data_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in {"__init__.py", MANIFEST_NAME}:
            continue
        relative = path.relative_to(data_root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "provenance": _provenance(relative),
                "contains_identifiers": False,
                "license": "Apache-2.0",
            }
        )
    return {
        "schema": "pyspace-packaged-data",
        "schema_version": 1,
        "upstream_space_commit": UPSTREAM_COMMIT,
        "files": entries,
    }


def _encoded(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "src" / "pyspace" / "data")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = args.data_root / MANIFEST_NAME
    expected = _encoded(build_manifest(args.data_root))
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            print(f"Packaged-data manifest is missing or stale: {output}")
            return 1
        print("Packaged-data manifest is current.")
        return 0
    output.write_text(expected, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
