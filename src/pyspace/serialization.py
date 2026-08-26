"""Safe, versioned serialization for PySpace analysis results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import pandas as pd

SCHEMA_NAME = "pyspace-result"
SCHEMA_VERSION = 1


class _Encoder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames: list[Path] = []
        self.arrays: dict[str, np.ndarray] = {}

    def encode(self, value: Any) -> Any:  # noqa: PLR0911 - explicit tagged-union dispatch keeps the schema clear.
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, (datetime, date)):
            return {"$datetime": value.isoformat()}
        if isinstance(value, Path):
            return {"$path": str(value)}
        if isinstance(value, pd.DataFrame):
            table_dir = self.root / "tables"
            table_dir.mkdir(exist_ok=True)
            relative = Path("tables") / f"table_{len(self.frames):04d}.csv"
            value.to_csv(self.root / relative, index=False)
            self.frames.append(relative)
            return {"$dataframe": relative.as_posix()}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("PySpace result bundles do not support object-dtype arrays")
            key = f"array_{len(self.arrays):04d}"
            self.arrays[key] = value
            return {"$array": key}
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "$dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {field.name: self.encode(getattr(value, field.name)) for field in fields(value)},
            }
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Result bundle mappings must use string keys")
            return {key: self.encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        raise TypeError(f"Unsupported result value: {type(value).__name__}")

    def finish(self) -> None:
        if self.arrays:
            np.savez_compressed(self.root / "arrays.npz", **cast(Any, self.arrays))


class _Decoder:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        arrays_path = self._resolve_member("arrays.npz")
        self.arrays = np.load(arrays_path, allow_pickle=False) if arrays_path.exists() else None

    def _resolve_member(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Bundle member escapes result bundle: {relative}")
        return candidate

    @staticmethod
    def _string_tag(value: dict[str, Any], tag: str) -> str:
        tagged_value = value[tag]
        if not isinstance(tagged_value, str):
            raise ValueError(f"{tag.removeprefix('$')} tag must contain a string")
        return tagged_value

    def decode(self, value: Any) -> Any:  # noqa: PLR0911 - explicit tagged-union dispatch keeps validation local.
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"$float"}:
            tagged_value = self._string_tag(value, "$float")
            if tagged_value not in {"nan", "inf", "-inf"}:
                raise ValueError("float tag must be 'nan', 'inf', or '-inf'")
            return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[tagged_value]
        if set(value) == {"$datetime"}:
            return self._string_tag(value, "$datetime")
        if set(value) == {"$path"}:
            return Path(self._string_tag(value, "$path"))
        if set(value) == {"$dataframe"}:
            relative = self._string_tag(value, "$dataframe")
            path = self._resolve_member(relative)
            relative_path = Path(relative)
            if relative_path.suffix.lower() != ".csv" or relative_path.parts[:1] != ("tables",):
                raise ValueError("dataframe tag must reference a CSV file in tables/")
            if not path.is_file():
                raise ValueError(f"Missing result bundle table: {relative}")
            return pd.read_csv(path)
        if set(value) == {"$array"}:
            key = self._string_tag(value, "$array")
            if self.arrays is None or key not in self.arrays.files:
                raise ValueError(f"Missing result bundle array: {key}")
            return np.asarray(self.arrays[key])
        if set(value) == {"$dataclass", "fields"}:
            self._string_tag(value, "$dataclass")
            dataclass_fields = value["fields"]
            if not isinstance(dataclass_fields, dict) or not all(isinstance(key, str) for key in dataclass_fields):
                raise ValueError("dataclass tag fields must be a JSON object with string keys")
            return {key: self.decode(item) for key, item in dataclass_fields.items()}
        return {key: self.decode(item) for key, item in value.items()}

    def close(self) -> None:
        if self.arrays is not None:
            self.arrays.close()


def save_result(result: Any, destination: str | Path) -> Path:
    """Write a result bundle without executing or pickling user-controlled objects."""
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"Result bundle destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".pyspace-result-", dir=destination_path.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        encoder = _Encoder(bundle)
        payload = encoder.encode(result)
        encoder.finish()
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "payload": payload,
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        bundle.replace(destination_path)
    return destination_path


def load_result(source: str | Path) -> Any:
    """Load and validate a PySpace result bundle."""
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"Result bundle is not a directory: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Result bundle has no manifest: {manifest_path}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Result bundle contains non-standard JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid result bundle JSON: {error.msg}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Result bundle manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA_NAME:
        raise ValueError("Unsupported result bundle schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported result bundle version: {manifest.get('schema_version')}")
    if "payload" not in manifest:
        raise ValueError("Result bundle manifest has no payload")

    decoder = _Decoder(root)
    try:
        return decoder.decode(manifest["payload"])
    finally:
        decoder.close()


__all__ = ["SCHEMA_NAME", "SCHEMA_VERSION", "load_result", "save_result"]
