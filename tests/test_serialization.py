from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_result_bundle_round_trip(tmp_path: Path) -> None:
    from pyspace.serialization import load_result, save_result

    payload = {
        "table": pd.DataFrame({"value": [1.0, 2.0]}),
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "nested": [True, None, {"created": datetime(2026, 8, 3, tzinfo=UTC)}],
    }
    destination = tmp_path / "analysis.pyspace"

    save_result(payload, destination)
    loaded = load_result(destination)

    pd.testing.assert_frame_equal(loaded["table"], payload["table"])
    np.testing.assert_array_equal(loaded["array"], payload["array"])
    assert loaded["nested"] == [True, None, {"created": "2026-08-03T00:00:00+00:00"}]


def test_result_bundle_rejects_object_arrays(tmp_path: Path) -> None:
    from pyspace.serialization import save_result

    with pytest.raises(TypeError, match="object-dtype"):
        save_result(np.array([{"secret": "not safe"}], dtype=object), tmp_path / "unsafe.pyspace")


def test_result_bundle_does_not_overwrite_existing_destination(tmp_path: Path) -> None:
    from pyspace.serialization import save_result

    destination = tmp_path / "analysis.pyspace"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_result({"value": 1}, destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_result_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    from pyspace.serialization import load_result

    destination = tmp_path / "malicious.pyspace"
    destination.mkdir()
    manifest = {
        "schema": "pyspace-result",
        "schema_version": 1,
        "payload": {"$dataframe": "../outside.csv"},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes result bundle"):
        load_result(destination)
