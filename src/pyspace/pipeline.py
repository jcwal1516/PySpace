"""Small stateful orchestrator over PySpace's canonical analysis functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core.census import CensusResult, census_coordinates, census_image
from .core.pattern_learning import learn_pattern_result
from .core.pattern_mapping import map_pattern
from .core.pattern_models import PatternResult
from .core.r_measure_cismi import measure_cisMI
from .core.r_measure_transmi import measure_transMI
from .io.table_loader import read_table
from .io.validation import validate_inputs
from .serialization import load_result, save_result

ProgressCallback = Callable[[str, float], None]


def _now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class PipelineState:
    """Serializable inputs, parameters, results, and provenance for one workflow."""

    input_path: str | None = None
    input_data: pd.DataFrame | np.ndarray | None = None
    input_type: str | None = None
    input_metadata: dict[str, Any] = field(default_factory=dict)
    radii: list[float] | None = None
    n_neighborhoods: int | list[int] | None = None
    variables: list[str] | None = None
    census_results: CensusResult | dict[str, Any] | None = None
    mi_results: dict[str, Any] = field(default_factory=dict)
    pattern_results: PatternResult | dict[str, Any] | None = None
    mapping_results: dict[str, Any] | np.ndarray | tuple[np.ndarray, list[str]] | None = None
    processing_history: list[dict[str, Any]] = field(default_factory=list)
    creation_time: datetime = field(default_factory=_now)
    last_modified: datetime = field(default_factory=_now)

    def record(self, operation: str, parameters: Mapping[str, Any], summary: Mapping[str, Any] | None = None) -> None:
        """Record an observable pipeline operation."""
        timestamp = _now()
        self.processing_history.append(
            {
                "operation": operation,
                "timestamp": timestamp.isoformat(),
                "parameters": dict(parameters),
                "summary": dict(summary or {}),
            }
        )
        self.last_modified = timestamp

    def as_payload(self) -> dict[str, Any]:
        """Return the stable payload stored in a result bundle."""
        return {
            "kind": "pipeline",
            "input_path": self.input_path,
            "input_data": self.input_data,
            "input_type": self.input_type,
            "input_metadata": self.input_metadata,
            "radii": self.radii,
            "n_neighborhoods": self.n_neighborhoods,
            "variables": self.variables,
            "census_results": self.census_results,
            "mi_results": self.mi_results,
            "pattern_results": self.pattern_results,
            "mapping_results": self.mapping_results,
            "processing_history": self.processing_history,
            "creation_time": self.creation_time,
            "last_modified": self.last_modified,
        }


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise TypeError(f"Expected an ISO datetime, got {type(value).__name__}")


class SpacePipeline:
    """Python-friendly workflow that delegates scientific work to canonical functions."""

    def __init__(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.state = PipelineState()

    def _progress(self, operation: str, fraction: float) -> None:
        if self.progress_callback is not None:
            self.progress_callback(operation, fraction)

    def load_image(self, image_path: str | Path, *, validate: bool = True) -> SpacePipeline:
        """Register an image input without reading or mutating it."""
        path = Path(image_path).expanduser().resolve()
        if validate:
            validation = validate_inputs(path, data_type="image")
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            self.state.input_metadata = dict(validation.get("metadata", {}))
        self.state.input_path = str(path)
        self.state.input_data = None
        self.state.input_type = "image"
        self.state.record("load_image", {"path": str(path), "validate": validate})
        return self

    def load_table(self, table: str | Path | pd.DataFrame, *, validate: bool = True) -> SpacePipeline:
        """Register a coordinate table, preserving DataFrame inputs exactly."""
        if isinstance(table, pd.DataFrame):
            frame = table.copy(deep=True)
            source: str | None = None
        else:
            path = Path(table).expanduser().resolve()
            validation = validate_inputs(path, data_type="table")
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            frame = read_table(path)
            source = str(path)
        if validate:
            validation = validate_inputs(frame, data_type="table")
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            self.state.input_metadata = dict(validation.get("metadata", {}))
        self.state.input_path = source
        self.state.input_data = frame
        self.state.input_type = "table"
        self.state.record("load_table", {"source": source or "DataFrame", "validate": validate})
        return self

    def set_parameters(
        self,
        *,
        radii: list[float] | tuple[float, ...] | None = None,
        n_neighborhoods: int | list[int] | None = None,
        variables: list[str] | tuple[str, ...] | None = None,
    ) -> SpacePipeline:
        """Set validated parameters used by subsequent steps."""
        if radii is not None:
            normalized = [float(radius) for radius in radii]
            if not normalized or any(radius <= 0 for radius in normalized):
                raise ValueError("radii must contain positive values")
            self.state.radii = normalized
        if n_neighborhoods is not None:
            counts = [n_neighborhoods] if isinstance(n_neighborhoods, int) else n_neighborhoods
            if not counts or any(int(count) <= 0 for count in counts):
                raise ValueError("n_neighborhoods must contain positive values")
            self.state.n_neighborhoods = n_neighborhoods
        if variables is not None:
            normalized_variables = [str(variable) for variable in variables]
            if not normalized_variables:
                raise ValueError("variables cannot be empty")
            self.state.variables = normalized_variables
        self.state.record(
            "set_parameters",
            {
                "radii": self.state.radii,
                "n_neighborhoods": self.state.n_neighborhoods,
                "variables": self.state.variables,
            },
        )
        return self

    def census(
        self,
        *,
        radii: list[float] | None = None,
        n_neighborhoods: int | list[int] | None = None,
        variables: list[str] | None = None,
        **kwargs: Any,
    ) -> CensusResult | dict[str, Any]:
        """Run census collection for the registered input."""
        self.set_parameters(radii=radii, n_neighborhoods=n_neighborhoods, variables=variables)
        if self.state.radii is None:
            raise ValueError("Set radii explicitly before census")
        selected_radii = self.state.radii
        self._progress("census", 0.0)
        if self.state.input_type == "table" and isinstance(self.state.input_data, pd.DataFrame):
            result = census_coordinates(
                self.state.input_data,
                radii=selected_radii,
                variables=self.state.variables,
                sample_size=self.state.n_neighborhoods,
                **kwargs,
            )
        elif self.state.input_type == "image" and self.state.input_path:
            if self.state.variables is not None:
                raise ValueError("Image census measures all variables; filter the resulting census explicitly")
            if self.state.n_neighborhoods is None:
                raise ValueError("Set n_neighborhoods explicitly before image census")
            result = census_image(
                self.state.input_path,
                radii=selected_radii,
                sample_size=self.state.n_neighborhoods,
                **kwargs,
            )
        else:
            raise RuntimeError("Load an image or coordinate table before running census")
        self.state.census_results = result
        self.state.radii = selected_radii
        count = len(result.get("neighborhoods", [])) if isinstance(result, dict) else len(result.neighborhoods)
        self.state.record("census", {}, {"neighborhoods": count})
        self._progress("census", 1.0)
        return result

    def measure_cisMI(
        self,
        *,
        depth: int = 3,
        bootstraps: int = 100,
        random_plan: list[pd.DataFrame] | None = None,
        allow_permutation_fallback: bool = False,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Measure within-sample mutual information using the SPACE-compatible core."""
        census_frame, patch_list = self._census_frame_and_patches()
        result = measure_cisMI(
            census=census_frame,
            patch_list=patch_list,
            depth=depth,
            bootstraps=bootstraps,
            random_censuses=random_plan,
            allow_permutation_fallback=allow_permutation_fallback,
            **kwargs,
        )
        self.state.mi_results["cisMI"] = result
        self.state.record("measure_cisMI", {"depth": depth, "bootstraps": bootstraps})
        return result

    def measure_transMI(
        self,
        censuses: list[pd.DataFrame],
        groups: pd.DataFrame,
        *,
        depth: int = 3,
        radii: list[float] | str = "all",
        bootstraps: int = 100,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Measure between-group mutual information using the SPACE-compatible core."""
        result = measure_transMI(
            censuses=censuses,
            groups=groups,
            depth=depth,
            radii=radii,
            bootstraps=bootstraps,
            **kwargs,
        )
        self.state.mi_results["transMI"] = result
        self.state.record("measure_transMI", {"depth": depth, "bootstraps": bootstraps})
        return result

    def learn_patterns(
        self,
        col_pal: Mapping[str, Sequence[str]],
        *,
        variables: list[str] | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Learn patterns from the current census."""
        selected_variables = variables or self.state.variables
        if not selected_variables:
            raise ValueError("variables are required for pattern learning")
        census_frame, _ = self._census_frame_and_patches()
        radius = float(kwargs.pop("radius", (self.state.radii or [census_frame["Radius"].iloc[0]])[0]))
        result = learn_pattern_result(
            census_frame,
            selected_variables,
            radius,
            col_pal,
            random_state=random_state,
            **kwargs,
        )
        self.state.pattern_results = result
        self.state.record("learn_patterns", {"variables": selected_variables, "random_state": random_state})
        return result

    def map_patterns(
        self,
        region_bounds: list[list[float]],
        image: np.ndarray | list[np.ndarray] | Mapping[str, np.ndarray],
        radii: Mapping[str | float, list[float]],
        col_pal: list[str],
        **kwargs: Any,
    ) -> tuple[np.ndarray, list[str]]:
        """Back-project learned patterns onto an image grid."""
        if not isinstance(self.state.pattern_results, PatternResult):
            raise RuntimeError("Run learn_patterns before mapping patterns")
        census_frame, _ = self._census_frame_and_patches()
        radius = float(kwargs.pop("radius", (self.state.radii or [census_frame["Radius"].iloc[0]])[0]))
        result = map_pattern(
            self.state.pattern_results.covariation_data,
            region_bounds,
            image,
            census_frame,
            radius,
            radii,
            col_pal,
            **kwargs,
        )
        self.state.mapping_results = result
        self.state.record("map_patterns", {"radius": radius, "region_bounds": region_bounds})
        return result

    def _census_frame_and_patches(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | None]:
        result = self.state.census_results
        if isinstance(result, CensusResult):
            if result.census is None:
                raise ValueError("The census result does not contain an exported census table")
            return result.census, result.patch_list
        if isinstance(result, dict):
            frame = result.get("census")
            if isinstance(frame, pd.DataFrame):
                patches = result.get("patch_list")
                return frame, patches if isinstance(patches, dict) else None
        raise RuntimeError("Run census before this analysis step")

    def get_results(self) -> dict[str, Any]:
        """Return a snapshot suitable for inspection or safe serialization."""
        return self.state.as_payload()

    def save_results(self, destination: str | Path) -> Path:
        """Save this workflow as a versioned, non-pickle result bundle."""
        return save_result(self.state.as_payload(), destination)

    @classmethod
    def load_results(cls, source: str | Path) -> SpacePipeline:
        """Load a workflow from a validated result bundle."""
        payload = load_result(source)
        if not isinstance(payload, dict) or payload.get("kind") != "pipeline":
            raise ValueError("Result bundle does not contain a PySpace pipeline")
        pipeline = cls()
        pipeline.state = PipelineState(
            input_path=payload.get("input_path"),
            input_data=payload.get("input_data"),
            input_type=payload.get("input_type"),
            input_metadata=dict(payload.get("input_metadata") or {}),
            radii=payload.get("radii"),
            n_neighborhoods=payload.get("n_neighborhoods"),
            variables=payload.get("variables"),
            census_results=payload.get("census_results"),
            mi_results=dict(payload.get("mi_results") or {}),
            pattern_results=payload.get("pattern_results"),
            mapping_results=payload.get("mapping_results"),
            processing_history=list(payload.get("processing_history") or []),
            creation_time=_parse_datetime(payload["creation_time"]),
            last_modified=_parse_datetime(payload["last_modified"]),
        )
        return pipeline


__all__ = ["PipelineState", "SpacePipeline"]
