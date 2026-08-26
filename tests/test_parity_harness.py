import inspect
import json
from pathlib import Path

import pyspace
from pyspace.parity import (
    DEFAULT_R_REPO,
    R_EXPORT_TO_PYTHON_SYMBOL,
    UPSTREAM_SPACE_COMMIT,
    _read_r_namespace_exports,
    check_api_surface,
    check_core_oracle_fixtures,
    main,
)


def test_r_exports_have_explicit_mapping():
    namespace_path = DEFAULT_R_REPO / "NAMESPACE"
    if not namespace_path.exists():
        import pytest

        pytest.skip(f"R SPACE namespace not found at {namespace_path}")

    r_exports = _read_r_namespace_exports(namespace_path)
    missing_mappings = sorted(set(r_exports) - set(R_EXPORT_TO_PYTHON_SYMBOL))
    assert missing_mappings == []


def test_api_surface_checks_pass():
    checks = check_api_surface(r_repo=DEFAULT_R_REPO)
    failures = [check for check in checks if not check.passed]
    assert failures == []


def test_r_style_exports_are_available_on_top_level_package():
    namespace_path = DEFAULT_R_REPO / "NAMESPACE"
    if not namespace_path.exists():
        import pytest

        pytest.skip(f"R SPACE namespace not found at {namespace_path}")

    missing = [name for name in _read_r_namespace_exports(namespace_path) if not hasattr(pyspace, name)]
    assert missing == []


def test_r_compatibility_signatures_keep_upstream_required_inputs_required():
    required_parameters = {
        "census_image": {"images", "radii", "sample_size"},
        "load_table": {"in_file", "table_type", "img", "col_pal"},
        "map_pattern": {"covar_data", "region_bounds", "img", "census", "radius", "radii", "col_pal"},
        "measure_cisMI": {"census", "patch_list", "depth"},
        "merge_objects": {"prof_table", "img", "col_pal", "obj_table", "obj_groups"},
        "plot_palette": {"col_pal", "axis_label", "col_labels"},
    }

    for function_name, expected in required_parameters.items():
        signature = inspect.signature(getattr(pyspace, function_name))
        actual = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        }
        assert expected.issubset(actual), f"{function_name} made required R inputs optional: {expected - actual}"


def test_mi_filter_parameters_use_python_safe_r_names():
    for function_name in ("measure_cisMI", "measure_transMI"):
        parameters = inspect.signature(getattr(pyspace, function_name)).parameters
        assert {"all", "alo", "not_"}.issubset(parameters)
        assert {"all_vars", "alo_vars", "not_vars"}.isdisjoint(parameters)


def test_map_pattern_has_no_random_palette_fallback():
    assert "random_state" not in inspect.signature(pyspace.map_pattern).parameters


def test_static_namespace_manifest_is_pinned():
    assert (DEFAULT_R_REPO / "COMMIT").read_text().strip() == UPSTREAM_SPACE_COMMIT


def test_core_oracle_fixtures_are_pinned_and_present():
    checks = check_core_oracle_fixtures()
    failures = [check for check in checks if not check.passed]

    assert failures == []
    commit_checks = [check for check in checks if check.name == "core_oracle::upstream_commit"]
    assert commit_checks and commit_checks[0].detail == f"upstream_commit={UPSTREAM_SPACE_COMMIT}"


def test_parity_cli_writes_machine_readable_report(tmp_path: Path):
    output = tmp_path / "parity.json"

    assert main(["--json", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["failed"] == 0
