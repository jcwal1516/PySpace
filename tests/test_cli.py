from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from pyspace.serialization import load_result


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pyspace.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_convert_csv_to_safe_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "converted.pyspace"
    expected = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    expected.to_csv(source, index=False)

    completed = _run_cli("convert", str(source), "--format", "bundle", "--output", str(destination))

    assert completed.returncode == 0, completed.stderr
    pd.testing.assert_frame_equal(load_result(destination), expected)


def test_pickle_input_has_stable_input_error_exit(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.pkl"
    source.write_bytes(b"not a pickle")

    completed = _run_cli("convert", str(source), "--format", "json", "--output", str(tmp_path / "out.json"))

    assert completed.returncode == 2
    assert "pickle" in completed.stderr.lower()
    assert not (tmp_path / "out.json").exists()


def test_table_census_completes_to_safe_bundle(tmp_path: Path) -> None:
    source = tmp_path / "cells.csv"
    destination = tmp_path / "census.pyspace"
    pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "object_id": [1, 1, 2, 2, 1],
            "marker": [0.0, 0.2, 0.5, 0.8, 1.0],
        }
    ).to_csv(source, index=False)

    completed = _run_cli(
        "census",
        str(source),
        "--radii",
        "1",
        "--variables",
        "marker",
        "--sample-size",
        "3",
        "--format",
        "bundle",
        "--output",
        str(destination),
    )

    assert completed.returncode == 0, completed.stderr
    loaded = load_result(destination)
    assert isinstance(loaded, dict)
    assert loaded["metadata"]["radii"] == [1.0]
