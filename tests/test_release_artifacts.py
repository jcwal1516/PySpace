from __future__ import annotations

import json
import runpy
import tomllib
from pathlib import Path

from scripts.check_kernel_coverage import kernel_coverage
from scripts.check_public_tree import audit_tree, candidate_files
from scripts.inspect_distribution import inspect_member_names
from scripts.write_artifact_manifest import build_manifest
from scripts.write_dependency_inventory import validate_inventory

import pyspace

ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_audit_rejects_prohibited_and_oversized_files(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text("value = 1\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "unsafe.pyc").write_bytes(b"compiled")
    (tmp_path / "patient.pkl").write_bytes(b"pickle")
    (tmp_path / "too-large.bin").write_bytes(b"x" * 33)

    findings = audit_tree(tmp_path, max_bytes=32)

    assert any("cache directory" in finding for finding in findings)
    assert any("prohibited extension" in finding for finding in findings)
    assert any("exceeds 32 bytes" in finding for finding in findings)


def test_artifact_manifest_is_deterministic_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text("stale", encoding="utf-8")

    first = build_manifest(tmp_path)
    second = build_manifest(tmp_path)

    assert first == second
    assert [entry["path"] for entry in first["files"]] == ["a.txt", "b.txt"]
    assert all(len(entry["sha256"]) == 64 for entry in first["files"])


def test_distribution_inspection_rejects_traversal_pickle_and_caches() -> None:
    findings = inspect_member_names(
        [
            "pyspace/__init__.py",
            "../outside.txt",
            "pyspace/model.pkl",
            "pyspace/__pycache__/module.pyc",
        ]
    )

    assert any("unsafe archive path" in finding for finding in findings)
    assert any("unsafe serialized object" in finding for finding in findings)
    assert any("cache or compiled file" in finding for finding in findings)


def test_release_tree_and_packaged_data_manifest_pass() -> None:
    assert audit_tree(ROOT) == []

    data_root = ROOT / "src" / "pyspace" / "data"
    manifest = json.loads((data_root / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    declared = {entry["path"] for entry in manifest["files"]}
    actual = {
        path.relative_to(data_root).as_posix()
        for path in candidate_files(ROOT)
        if data_root in path.parents and path.name not in {"__init__.py", "DATA_MANIFEST.json"}
    }

    assert declared == actual
    assert all(entry["contains_identifiers"] is False for entry in manifest["files"])
    assert all(entry["license"] == "Apache-2.0" for entry in manifest["files"])


def test_api_documentation_names_every_export() -> None:
    api_text = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    missing = [name for name in pyspace.__all__ if f"`{name}`" not in api_text]
    assert missing == []


def test_public_repository_metadata_uses_the_final_github_target() -> None:
    repository = "https://github.com/jcwal1516/PySpace"
    advisory = f"{repository}/security/advisories/new"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["urls"]["Repository"] == repository
    assert pyproject["project"]["urls"]["Issues"] == f"{repository}/issues"
    assert repository in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert repository in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert advisory in (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    for filename in ("SECURITY.md", "docs/security.md", "security_best_practices_report.md"):
        text = (ROOT / filename).read_text(encoding="utf-8").lower()
        assert "repository name is intentionally undecided" not in text
        assert "eventual public repository" not in text


def test_examples_execute_without_external_data(tmp_path: Path) -> None:
    for filename in ("table_workflow.py", "image_workflow.py", "safe_bundle.py"):
        namespace = runpy.run_path(str(ROOT / "examples" / filename))
        result = namespace["main"](tmp_path / filename.removesuffix(".py"))
        assert result


def test_kernel_coverage_combines_lines_and_branches_and_requires_every_file() -> None:
    report = {
        "files": {
            "a.py": {
                "summary": {
                    "covered_lines": 8,
                    "covered_branches": 1,
                    "num_statements": 8,
                    "num_branches": 2,
                }
            }
        }
    }

    assert kernel_coverage(report, ("a.py",)) == 90.0
    try:
        kernel_coverage(report, ("missing.py",))
    except ValueError as error:
        assert "missing kernel files" in str(error)
    else:
        raise AssertionError("missing kernel file was accepted")


def test_dependency_inventory_check_is_declaration_based_and_requires_resolution(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname="example"\nversion="1"\ndependencies=["numpy>=1"]\n',
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "pyspace-direct-dependencies",
                "entries": [
                    {
                        "group": "runtime",
                        "name": "numpy",
                        "declared": "numpy>=1",
                        "installed_version": "2.0",
                        "license": "BSD-3-Clause",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert validate_inventory(pyproject, inventory) == []
    document = json.loads(inventory.read_text(encoding="utf-8"))
    document["entries"][0]["license"] = None
    inventory.write_text(json.dumps(document), encoding="utf-8")
    assert any("resolved version or license" in finding for finding in validate_inventory(pyproject, inventory))
