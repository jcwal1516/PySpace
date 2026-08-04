"""Pinned SPACE-to-PySpace parity metadata and release checks."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

UPSTREAM_SPACE_COMMIT = "94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8"
PARITY_DATA_DIR = Path(__file__).parent / "data" / "parity"
CORE_ORACLE_DIR = PARITY_DATA_DIR / "oracles"
PINNED_NAMESPACE_DIR = PARITY_DATA_DIR / "upstream"
DEFAULT_R_REPO = PINNED_NAMESPACE_DIR

R_EXPORT_TO_PYTHON_SYMBOL: dict[str, str] = {
    "alpha_diversity": "pyspace.alpha_diversity",
    "beta_diversity": "pyspace.beta_diversity",
    "calc_vol": "pyspace.calc_vol",
    "calc_vols": "pyspace.calc_vols",
    "census_image": "pyspace.census_image",
    "census_table": "pyspace.census_table",
    "learn_pattern": "pyspace.learn_pattern",
    "load_image": "pyspace.load_image",
    "load_table": "pyspace.load_table",
    "make_palette": "pyspace.make_palette",
    "map_pattern": "pyspace.map_pattern",
    "measure_cisMI": "pyspace.measure_cisMI",
    "measure_transMI": "pyspace.measure_transMI",
    "merge_objects": "pyspace.merge_objects",
    "patch_3D": "pyspace.patch_3D",
    "plot_MI_radius": "pyspace.plot_MI_radius",
    "plot_MI_rank": "pyspace.plot_MI_rank",
    "plot_dist": "pyspace.plot_dist",
    "plot_image": "pyspace.plot_image",
    "plot_palette": "pyspace.plot_palette",
    "plot_table": "pyspace.plot_table",
    "standardize_censuses": "pyspace.standardize_censuses",
    "suggest_number": "pyspace.suggest_number",
    "suggest_radii": "pyspace.suggest_radii",
}


@dataclass(frozen=True)
class ParityCheck:
    name: str
    passed: bool
    detail: str


def cismi_provenance(*, allow_permutation_fallback: bool, patch_list_available: bool) -> dict[str, Any]:
    """Describe whether a cisMI null model preserves upstream SPACE parity."""
    if patch_list_available:
        return {
            "patch_list_used": True,
            "allow_permutation_fallback": bool(allow_permutation_fallback),
            "r_parity_mode": "strict",
            "null_model": "space_patch_randomization",
        }
    if not allow_permutation_fallback:
        raise ValueError("Strict cisMI parity requires a SPACE-style patch list")
    return {
        "patch_list_used": False,
        "allow_permutation_fallback": True,
        "r_parity_mode": "exploratory",
        "null_model": "within_column_permutation",
    }


def _read_r_namespace_exports(namespace_path: Path) -> list[str]:
    exports: list[str] = []
    for raw_line in namespace_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export(") and line.endswith(")"):
            exports.append(line[len("export(") : -1].strip())
    return sorted(set(exports))


def _resolve_symbol(path: str) -> Any:
    module_name, _, attribute = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_upstream_source(r_repo: Path) -> list[ParityCheck]:
    """Verify the pinned revision and reject a dirty Git checkout."""
    if (r_repo / ".git").exists():
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=r_repo, capture_output=True, text=True, check=False)
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=r_repo, capture_output=True, text=True, check=False
        )
        actual_revision = revision.stdout.strip()
        dirty_detail = status.stdout.strip()
        return [
            ParityCheck(
                "upstream::commit",
                revision.returncode == 0 and actual_revision == UPSTREAM_SPACE_COMMIT,
                actual_revision or revision.stderr.strip(),
            ),
            ParityCheck(
                "upstream::pristine",
                status.returncode == 0 and not dirty_detail,
                "clean" if status.returncode == 0 and not dirty_detail else dirty_detail or status.stderr.strip(),
            ),
        ]
    commit_path = r_repo / "COMMIT"
    commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.is_file() else "missing"
    return [ParityCheck("upstream::commit", commit == UPSTREAM_SPACE_COMMIT, commit)]


@contextlib.contextmanager
def pristine_upstream_checkout() -> Iterator[Path]:
    """Yield a temporary clean checkout of the exact upstream SPACE revision."""
    with tempfile.TemporaryDirectory(prefix="pyspace-r-upstream-") as temporary:
        checkout = Path(temporary) / "SPACE"
        clone = subprocess.run(
            ["git", "clone", "--quiet", "https://github.com/eschrom/SPACE.git", str(checkout)],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone.returncode != 0:
            raise RuntimeError(f"Unable to clone upstream SPACE: {clone.stderr.strip()}")
        checkout_command = subprocess.run(
            ["git", "checkout", "--quiet", UPSTREAM_SPACE_COMMIT],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=False,
        )
        if checkout_command.returncode != 0:
            raise RuntimeError(f"Unable to check out pinned SPACE commit: {checkout_command.stderr.strip()}")
        failures = [check for check in check_upstream_source(checkout) if not check.passed]
        if failures:
            raise RuntimeError(f"Upstream checkout verification failed: {failures}")
        yield checkout


def check_api_surface(r_repo: Path = DEFAULT_R_REPO) -> list[ParityCheck]:
    checks = check_upstream_source(r_repo)
    namespace_path = r_repo / "NAMESPACE"
    if not namespace_path.is_file():
        return [*checks, ParityCheck("r_namespace_present", False, f"Missing {namespace_path}")]

    exports = _read_r_namespace_exports(namespace_path)
    export_set = set(exports)
    mapped_set = set(R_EXPORT_TO_PYTHON_SYMBOL)
    missing = sorted(export_set - mapped_set)
    stale = sorted(mapped_set - export_set)
    checks.extend(
        [
            ParityCheck("r_exports_mapped", not missing, "ok" if not missing else f"Missing mappings: {missing}"),
            ParityCheck("no_stale_mappings", not stale, "ok" if not stale else f"Stale mappings: {stale}"),
        ]
    )
    for export_name in exports:
        symbol = R_EXPORT_TO_PYTHON_SYMBOL.get(export_name)
        if symbol is None:
            continue
        try:
            callable_symbol = callable(_resolve_symbol(symbol))
            detail = f"{symbol} is callable" if callable_symbol else f"{symbol} is not callable"
        except (AttributeError, ImportError) as exc:
            callable_symbol = False
            detail = f"{symbol} import failed: {exc}"
        checks.append(ParityCheck(f"symbol::{export_name}", callable_symbol, detail))
    return checks


def check_core_oracle_fixtures() -> list[ParityCheck]:
    metadata_path = CORE_ORACLE_DIR / "metadata.json"
    if not metadata_path.is_file():
        return [ParityCheck("core_oracle::metadata", False, f"Missing {metadata_path}")]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = [
        ParityCheck(
            "core_oracle::upstream_commit",
            metadata.get("upstream_commit") == UPSTREAM_SPACE_COMMIT,
            f"upstream_commit={metadata.get('upstream_commit')}",
        )
    ]
    fixtures = metadata.get("fixtures")
    if not isinstance(fixtures, dict):
        return [*checks, ParityCheck("core_oracle::manifest", False, "fixtures must be a checksum mapping")]
    for filename, expected_hash in fixtures.items():
        path = CORE_ORACLE_DIR / filename
        actual_hash = _sha256(path) if path.is_file() else "missing"
        checks.append(ParityCheck(f"core_oracle::{filename}", actual_hash == expected_hash, f"sha256={actual_hash}"))
    return checks


def run_parity_checks(r_repo: Path = DEFAULT_R_REPO) -> dict[str, Any]:
    checks = check_api_surface(r_repo) + check_core_oracle_fixtures()
    passed = sum(check.passed for check in checks)
    total = len(checks)
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "score": round(passed / total, 4) if total else 0.0,
        },
        "checks": [asdict(check) for check in checks],
    }


def format_text_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Parity score: {summary['passed']}/{summary['total']} ({summary['score'] * 100:.1f}%)",
        "",
    ]
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"[{status}] {check['name']} - {check['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pinned SPACE-to-PySpace parity checks")
    parser.add_argument("--r-repo", type=Path, default=DEFAULT_R_REPO)
    parser.add_argument("--live-upstream", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.live_upstream:
        with pristine_upstream_checkout() as checkout:
            report = run_parity_checks(checkout)
    else:
        report = run_parity_checks(args.r_repo)
    rendered = json.dumps(report, indent=2) + "\n" if args.json else format_text_report(report) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
