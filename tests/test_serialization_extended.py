from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from pyspace.serialization import load_result, save_result


def _bundle(tmp_path: Path, payload: object) -> Path:
    root = tmp_path / "result.pyspace"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"schema": "pyspace-result", "schema_version": 1, "payload": payload}),
        encoding="utf-8",
    )
    return root


def test_bundle_round_trip_special_scalars_paths_dates_and_tuples(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.pyspace"
    payload = {
        "nan": math.nan,
        "positive": math.inf,
        "negative": -math.inf,
        "path": Path("relative/file.csv"),
        "date": date(2026, 8, 3),
        "tuple": (np.int64(2), np.float64(3.5)),
    }

    save_result(payload, destination)
    loaded = load_result(destination)

    assert math.isnan(loaded["nan"])
    assert loaded["positive"] == math.inf
    assert loaded["negative"] == -math.inf
    assert loaded["path"] == Path("relative/file.csv")
    assert loaded["date"] == "2026-08-03"
    assert loaded["tuple"] == [2, 3.5]


def test_bundle_writer_rejects_unsupported_values_and_mapping_keys(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="string keys"):
        save_result({1: "value"}, tmp_path / "keys.pyspace")
    with pytest.raises(TypeError, match="Unsupported result value"):
        save_result({1, 2}, tmp_path / "set.pyspace")


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"$float": "not-a-float"}, "float tag"),
        ({"$path": 42}, "path tag"),
        ({"$datetime": []}, "datetime tag"),
        ({"$dataframe": 42}, "dataframe tag"),
        ({"$array": []}, "array tag"),
        ({"$dataclass": "example.Result", "fields": []}, "dataclass tag"),
    ],
)
def test_bundle_reader_rejects_malformed_tagged_values(tmp_path: Path, payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_result(_bundle(tmp_path, payload))


def test_bundle_reader_rejects_nonstandard_json_and_bad_manifest_shape(tmp_path: Path) -> None:
    nonstandard = tmp_path / "nan.pyspace"
    nonstandard.mkdir()
    (nonstandard / "manifest.json").write_text(
        '{"schema":"pyspace-result","schema_version":1,"payload":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-standard JSON"):
        load_result(nonstandard)

    root_array = tmp_path / "array.pyspace"
    root_array.mkdir()
    (root_array / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_result(root_array)

    missing_payload = tmp_path / "missing.pyspace"
    missing_payload.mkdir()
    (missing_payload / "manifest.json").write_text(
        json.dumps({"schema": "pyspace-result", "schema_version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="payload"):
        load_result(missing_payload)


def test_bundle_reader_reports_missing_and_wrong_schema_members(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        load_result(tmp_path / "missing.pyspace")

    root = tmp_path / "empty.pyspace"
    root.mkdir()
    with pytest.raises(ValueError, match="no manifest"):
        load_result(root)

    wrong_schema = _bundle(tmp_path / "wrong", {"value": 1})
    manifest = json.loads((wrong_schema / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema"] = "other"
    (wrong_schema / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_result(wrong_schema)

    wrong_version = _bundle(tmp_path / "version", {"value": 1})
    manifest = json.loads((wrong_version / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    (wrong_version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_result(wrong_version)


def test_bundle_reader_rejects_array_archive_symlink_outside_bundle(tmp_path: Path) -> None:
    outside = tmp_path / "outside.npz"
    np.savez(outside, array_0000=np.array([42]))
    bundle = _bundle(tmp_path / "linked", {"$array": "array_0000"})
    try:
        (bundle / "arrays.npz").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"filesystem does not permit test symlinks: {error}")

    with pytest.raises(ValueError, match="escapes result bundle"):
        load_result(bundle)
