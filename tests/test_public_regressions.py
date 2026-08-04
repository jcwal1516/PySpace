from __future__ import annotations

import inspect
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest


def _coordinate_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "object_id": [1, 1, 2, 2],
            "marker": [0.1, 0.2, 0.8, 0.9],
        }
    )


def test_table_validation_does_not_require_census_radius() -> None:
    from pyspace.io.validation import validate_inputs

    result = validate_inputs(_coordinate_table(), data_type="table")

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["stats"]["row_count"] == 4


def test_pipeline_keeps_dataframe_and_passes_it_to_census(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyspace.pipeline as pipeline_module

    table = _coordinate_table()
    observed: dict[str, object] = {}

    def fake_census_coordinates(table_data: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        observed["table_data"] = table_data
        observed["kwargs"] = kwargs
        return {"neighborhoods": [], "metadata": {}}

    monkeypatch.setattr(pipeline_module, "census_coordinates", fake_census_coordinates)

    pipeline = pipeline_module.SpacePipeline()
    pipeline.load_table(table, validate=True)
    pipeline.census(radii=[1.0], n_neighborhoods=1)

    pd.testing.assert_frame_equal(cast(pd.DataFrame, observed["table_data"]), table)


def test_pipeline_safe_bundle_round_trip(tmp_path: Path) -> None:
    from pyspace.pipeline import SpacePipeline

    pipeline = SpacePipeline()
    pipeline.load_table(_coordinate_table())
    pipeline.set_parameters(radii=[1.0], variables=["marker"])
    destination = tmp_path / "analysis.pyspace"

    pipeline.save_results(destination)
    restored = SpacePipeline.load_results(destination)

    assert restored.state.input_type == "table"
    pd.testing.assert_frame_equal(restored.state.input_data, _coordinate_table())
    assert restored.state.radii == [1.0]


@pytest.mark.parametrize("suffix", [".xlsx", ".parquet"])
def test_supported_table_formats_are_dispatched_explicitly(tmp_path: Path, suffix: str) -> None:
    from pyspace.io.validation import validate_inputs

    table = _coordinate_table()
    path = tmp_path / f"cells{suffix}"
    if suffix == ".xlsx":
        table.to_excel(path, index=False)
    else:
        table.to_parquet(path, index=False)

    result = validate_inputs(path, data_type="table")

    assert result["valid"] is True


def test_unknown_table_extension_is_rejected_before_csv_parsing(tmp_path: Path) -> None:
    from pyspace.io.validation import validate_inputs

    path = tmp_path / "cells.unknown"
    path.write_bytes(b"not a comma separated table")

    with pytest.raises(ValueError, match="Unsupported input format"):
        validate_inputs(path)


def test_network_result_repr_handles_missing_modularity() -> None:
    from pyspace.core.networks import NetworkResult

    assert repr(NetworkResult()) == "NetworkResult(nodes=0, edges=0, communities=0, modularity=N/A)"


def test_network_result_repr_counts_distinct_communities() -> None:
    from pyspace.core.networks import NetworkResult

    result = NetworkResult()
    result.communities = {"partition": {0: 0, 1: 0, 2: 1}}

    assert "communities=2" in repr(result)


def test_result_types_do_not_expose_unpopulated_speculative_fields() -> None:
    from pyspace.core.census_models import CensusResult, Neighborhood
    from pyspace.core.networks import NetworkResult
    from pyspace.core.pattern_models import PatternResult

    removed = {
        Neighborhood: {"volume", "surface_area", "connectivity_type"},
        CensusResult: {"variable_classification"},
        NetworkResult: {"significance_tests", "variables"},
        PatternResult: {"focal_group", "reference_group"},
    }

    for result_type, field_names in removed.items():
        assert field_names.isdisjoint(inspect.signature(result_type).parameters)


def test_python_apis_do_not_expose_single_option_knobs_or_duplicate_aliases() -> None:
    from pyspace import load_example_data, make_palette
    from pyspace.core.census_sampling import create_neighborhoods
    from pyspace.core.networks import calculate_network_modularity
    from pyspace.core.pattern_models import SelfOrganizingMap

    removed: list[tuple[Callable[..., object], set[str]]] = [
        (SelfOrganizingMap, {"input_dim", "distance_metric"}),
        (calculate_network_modularity, {"null_model"}),
        (create_neighborhoods, {"merge_cell_types"}),
        (load_example_data, {"dataset"}),
        (make_palette, {"seed"}),
    ]

    for callable_object, parameter_names in removed:
        assert parameter_names.isdisjoint(inspect.signature(callable_object).parameters)


def test_silhouette_uses_distance_matrix() -> None:
    from pyspace.core.networks import _calculate_silhouette_score

    distances = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 1.1, 1.0],
            [1.0, 1.1, 0.0, 0.1],
            [1.1, 1.0, 0.1, 0.0],
        ]
    )

    score = _calculate_silhouette_score(distances, {0: 0, 1: 0, 2: 1, 3: 1})

    assert np.isfinite(score)
    assert score > 0.8


def test_cli_refuses_pickle_input(tmp_path: Path) -> None:
    from pyspace.cli import load_data

    path = tmp_path / "unsafe.pkl"
    path.write_bytes(b"not loaded")

    with pytest.raises(ValueError, match="pickle"):
        load_data(str(path))


def test_import_is_quiet_and_does_not_create_files(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import pyspace"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
