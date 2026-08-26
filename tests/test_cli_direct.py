from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pyspace.cli import (
    _csv_frame,
    _parse_sample_size,
    load_data,
    main,
    parse_comma_separated,
    parse_radii,
    save_results,
    setup_argument_parser,
)
from pyspace.core.census_models import CensusResult
from pyspace.serialization import save_result


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "y": np.zeros(8),
            "object_id": [1, 1, 2, 2, 1, 2, 1, 2],
            "marker": np.linspace(0, 1, 8),
        }
    )


def test_cli_parsers_cover_valid_and_invalid_inputs() -> None:
    assert parse_comma_separated(" a, b ,, ") == ["a", "b"]
    assert parse_comma_separated(None) == []
    assert parse_radii("1,2.5") == [1.0, 2.5]
    assert _parse_sample_size("4", 2) == 4
    assert _parse_sample_size("4,5", 2) == [4, 5]
    with pytest.raises(SystemExit):
        setup_argument_parser().parse_args(
            ["analyze", "input.pyspace", "--method", "patterns", "--output", "out.pyspace"]
        )
    with pytest.raises(ValueError, match="Invalid radii"):
        parse_radii("not-a-number")
    with pytest.raises(ValueError, match="finite positive"):
        parse_radii("0")
    with pytest.raises(ValueError, match="Invalid sample size"):
        _parse_sample_size("x", 1)
    with pytest.raises(ValueError, match="positive integers"):
        _parse_sample_size("0", 1)
    with pytest.raises(ValueError, match="one count"):
        _parse_sample_size("1,2,3", 2)


def test_cli_main_table_and_image_census_paths(tmp_path: Path) -> None:
    table = tmp_path / "cells.csv"
    _cells().to_csv(table, index=False)
    table_output = tmp_path / "table.pyspace"

    table_status = main(
        [
            "--verbose",
            "census",
            str(table),
            "--radii",
            "1.5",
            "--variables",
            "marker",
            "--sample-size",
            "3",
            "--output",
            str(table_output),
        ]
    )

    image_path = tmp_path / "objects.png"
    Image.fromarray(np.array([[0, 1, 1], [0, 1, 2], [0, 2, 2]], dtype=np.uint8)).save(image_path)
    image_output = tmp_path / "image.json"
    image_status = main(
        [
            "census",
            str(image_path),
            "--radii",
            "1",
            "--sample-size",
            "1",
            "--format",
            "json",
            "--output",
            str(image_output),
        ]
    )

    assert table_status == 0
    assert image_status == 0
    assert table_output.is_dir()
    assert image_output.is_file()


def test_cli_convert_safe_formats_and_refuses_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.tsv"
    _cells().to_csv(source, sep="\t", index=False)
    json_output = tmp_path / "source.json"
    assert main(["convert", str(source), "--format", "json", "--output", str(json_output)]) == 0

    bundle = tmp_path / "source.pyspace"
    assert main(["convert", str(source), "--format", "bundle", "--output", str(bundle)]) == 0
    csv_output = tmp_path / "source.csv"
    assert main(["convert", str(bundle), "--format", "csv", "--output", str(csv_output)]) == 0

    assert main(["convert", str(source), "--format", "json", "--output", str(json_output)]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_cli_npz_json_and_bundle_loading_and_saving(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    array = np.arange(4).reshape(2, 2)
    npz_path = tmp_path / "arrays.npz"
    save_results({"matrix": array}, npz_path, "npz", verbose=True)
    loaded_npz = load_data(npz_path, verbose=True)

    json_path = tmp_path / "payload.json"
    save_results({"array": array, "scalar": np.int64(3), "path": Path("relative")}, json_path, "json")
    loaded_json = load_data(json_path)

    bundle_path = tmp_path / "payload.pyspace"
    save_results(pd.DataFrame({"a": [1]}), bundle_path, "bundle")
    loaded_bundle = load_data(bundle_path)

    assert np.array_equal(loaded_npz["matrix"], array)
    assert loaded_json == {"array": [[0, 1], [2, 3]], "scalar": 3, "path": "relative"}
    pd.testing.assert_frame_equal(loaded_bundle, pd.DataFrame({"a": [1]}))
    assert "Saved:" in capsys.readouterr().out


def test_cli_csv_and_output_type_validation(tmp_path: Path) -> None:
    census = CensusResult([], {}, [], census=pd.DataFrame({"x": [1]}))
    pd.testing.assert_frame_equal(_csv_frame(census), pd.DataFrame({"x": [1]}))
    pd.testing.assert_frame_equal(_csv_frame({"only": pd.DataFrame({"x": [2]})}), pd.DataFrame({"x": [2]}))

    with pytest.raises(TypeError, match="CSV output"):
        _csv_frame({"left": pd.DataFrame(), "right": pd.DataFrame()})
    with pytest.raises(ValueError, match="pickle output"):
        save_results({}, tmp_path / "unsafe.pkl", "pickle")
    with pytest.raises(TypeError, match="NPZ output"):
        save_results({"not": "an array"}, tmp_path / "bad.npz", "npz")
    with pytest.raises(TypeError, match="object-dtype"):
        save_results(np.array([{"x": 1}], dtype=object), tmp_path / "object.npz", "npz")
    with pytest.raises(ValueError, match="Unsupported output"):
        save_results({}, tmp_path / "bad.out", "unknown")


def test_cli_plot_variants_from_safe_bundle(tmp_path: Path) -> None:
    payload = {
        "enrichment_scores": np.array([0.1, 0.5, 0.2]),
        "som_result": {"training_history": {"quantization_error": [1.0, 0.5, 0.25]}},
        "covariation_data": pd.DataFrame(
            {
                "variable": ["A", "A", "B", "B"],
                "position": [0, 1, 0, 1],
                "mean_abundance": [0.1, 0.2, 0.3, 0.4],
            }
        ),
    }
    bundle = save_result(payload, tmp_path / "pattern.pyspace")
    for plot_type in ("enrichment", "som", "covariation"):
        output = tmp_path / f"{plot_type}.png"
        assert main(["plot", str(bundle), "--type", plot_type, "--output", str(output), "--dpi", "72"]) == 0
        assert output.stat().st_size > 0


def test_cli_analyze_pattern_and_reports_input_and_operation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    census = pd.DataFrame(
        {
            "O1.1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "O1.2": [90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
            "X": np.arange(6),
            "Y": np.zeros(6),
            "Z": np.zeros(6),
            "Radius": np.repeat(1.0, 6),
        }
    )
    bundle = save_result(census, tmp_path / "census.pyspace")
    output = tmp_path / "patterns.pyspace"

    status = main(
        [
            "analyze",
            str(bundle),
            "--method",
            "som",
            "--variables",
            "O1.1,O1.2",
            "--iterations",
            "2",
            "--random-state",
            "3",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert main(["convert", str(tmp_path / "missing.csv"), "--format", "json", "--output", "x.json"]) == 2

    from pyspace import cli

    monkeypatch.setattr(cli, "handle_convert_command", lambda _args: (_ for _ in ()).throw(RuntimeError("boom")))
    source = tmp_path / "valid.csv"
    pd.DataFrame({"x": [1]}).to_csv(source, index=False)
    assert main(["convert", str(source), "--format", "json", "--output", str(tmp_path / "never.json")]) == 1
    captured = capsys.readouterr()
    assert "Input error:" in captured.err
    assert "Operation failed: boom" in captured.err


@dataclass
class _Payload:
    value: int


def test_cli_json_encoder_supports_dataclasses(tmp_path: Path) -> None:
    output = tmp_path / "dataclass.json"
    save_results(_Payload(4), output, "json")
    assert load_data(output) == {"value": 4}
