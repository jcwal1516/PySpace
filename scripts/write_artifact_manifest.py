#!/usr/bin/env python3
"""Create or verify a deterministic manifest for public release files."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

_candidate_module = importlib.import_module("scripts.check_public_tree" if __package__ else "check_public_tree")
_candidate_files = cast(Callable[[Path], list[Path]], _candidate_module.candidate_files)

MANIFEST_NAME = "ARTIFACT_MANIFEST.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_manifest(root: str | Path) -> dict[str, Any]:
    """Describe every prospective release file except the manifest itself."""
    root_path = Path(root).resolve()
    entries = []
    for path in _candidate_files(root_path):
        relative = path.relative_to(root_path).as_posix()
        if relative == MANIFEST_NAME:
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "schema": "pyspace-public-artifacts",
        "schema_version": 1,
        "package_version": "0.1.0",
        "upstream_space_commit": "94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8",
        "files": entries,
    }


def _encoded(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / MANIFEST_NAME
    expected = _encoded(build_manifest(root))
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            print(f"Artifact manifest is missing or stale: {output}")
            return 1
        print("Artifact manifest is current.")
        return 0
    output.write_text(expected, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
