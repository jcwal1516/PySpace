#!/usr/bin/env python3
"""Record resolved direct dependency versions and license metadata."""

from __future__ import annotations

import argparse
import json
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

GROUPS = ("runtime", "community", "dev", "docs")


def _license(distribution: metadata.Distribution) -> str:
    expression = distribution.metadata["License-Expression"]
    if expression:
        return expression
    classifiers = [
        value.removeprefix("License :: OSI Approved :: ")
        for value in distribution.metadata.get_all("Classifier", [])
        if value.startswith("License :: OSI Approved :: ")
    ]
    if classifiers:
        return "; ".join(classifiers)
    value = (distribution.metadata["License"] or "Unknown").strip()
    return value if len(value) <= 160 else "See installed distribution license files"


def _declared_groups(pyproject_path: Path) -> dict[str, list[str]]:
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = document["project"]
    optional = project.get("optional-dependencies", {})
    return {
        "runtime": list(project.get("dependencies", [])),
        "community": list(optional.get("community", [])),
        "dev": list(optional.get("dev", [])),
        "docs": list(optional.get("docs", [])),
    }


def build_inventory(pyproject_path: Path) -> dict[str, Any]:
    """Build a deterministic direct-dependency inventory for this environment."""
    groups = _declared_groups(pyproject_path)
    entries: list[dict[str, Any]] = []
    for group in GROUPS:
        for declaration in groups[group]:
            requirement = Requirement(declaration)
            try:
                distribution = metadata.distribution(requirement.name)
            except metadata.PackageNotFoundError:
                entries.append(
                    {
                        "group": group,
                        "name": requirement.name,
                        "declared": str(requirement),
                        "installed_version": None,
                        "license": None,
                        "homepage": None,
                    }
                )
                continue
            project_urls = distribution.metadata.get_all("Project-URL", [])
            entries.append(
                {
                    "group": group,
                    "name": requirement.name,
                    "declared": str(requirement),
                    "installed_version": distribution.version,
                    "license": _license(distribution),
                    "homepage": distribution.metadata["Home-page"] or (project_urls[0] if project_urls else None),
                }
            )
    return {
        "schema": "pyspace-direct-dependencies",
        "schema_version": 1,
        "package_version": "0.1.0",
        "entries": entries,
    }


def validate_inventory(pyproject_path: Path, inventory_path: Path) -> list[str]:
    """Validate that a recorded release environment covers every direct declaration."""
    if not inventory_path.is_file():
        return [f"dependency inventory is missing: {inventory_path}"]
    try:
        inventory: Any = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"dependency inventory is invalid JSON: {error}"]
    if not isinstance(inventory, dict) or inventory.get("schema") != "pyspace-direct-dependencies":
        return ["dependency inventory has an unsupported schema"]
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        return ["dependency inventory entries must be a list"]

    expected = {
        (group, Requirement(declaration).name.casefold(), str(Requirement(declaration)))
        for group, declarations in _declared_groups(pyproject_path).items()
        for declaration in declarations
    }
    recorded: set[tuple[str, str, str]] = set()
    findings: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            findings.append("dependency inventory contains a non-object entry")
            continue
        group, name, declared = entry.get("group"), entry.get("name"), entry.get("declared")
        if not isinstance(group, str) or not isinstance(name, str) or not isinstance(declared, str):
            findings.append("dependency inventory entry lacks group, name, or declaration")
            continue
        recorded.add((group, name.casefold(), declared))
        if not entry.get("installed_version") or not entry.get("license"):
            findings.append(f"dependency inventory lacks resolved version or license: {name}")
    findings.extend(f"dependency inventory is missing declaration: {item}" for item in sorted(expected - recorded))
    findings.extend(f"dependency inventory has stale declaration: {item}" for item in sorted(recorded - expected))
    return findings


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=root / "pyproject.toml")
    parser.add_argument("--output", type=Path, default=root / "DEPENDENCY_INVENTORY.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        findings = validate_inventory(args.pyproject, args.output)
        if findings:
            for finding in findings:
                print(f"FAIL: {finding}")
            return 1
        print("Dependency inventory declarations and resolved metadata are complete.")
        return 0
    expected = json.dumps(build_inventory(args.pyproject), indent=2, sort_keys=True) + "\n"
    args.output.write_text(expected, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
