from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import pyspace.pipeline as pipeline_module
from pyspace import SpacePipeline
from pyspace.core.census_models import CensusResult
from pyspace.serialization import save_result


def _coordinate_table(size: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(size, dtype=float),
            "y": np.tile([0.0, 1.0], size // 2),
            "object_id": np.tile([1, 2, 1], int(np.ceil(size / 3)))[:size],
            "marker": np.linspace(0, 1, size),
        }
    )


def _pattern_census() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "O1.1": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "S1.1": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "X": [1, 1, 1, 2, 2, 2],
            "Y": [1, 2, 3, 1, 2, 3],
            "Z": [1] * 6,
            "Radius": [1.0] * 6,
        }
    )


def test_pipeline_table_auto_parameters_census_progress_and_bundle(tmp_path: Path) -> None:
    progress: list[tuple[str, float]] = []
    pipeline = SpacePipeline(progress_callback=lambda operation, value: progress.append((operation, value)))
    pipeline.load_table(_coordinate_table())
    pipeline.census(radii=[1.5], n_neighborhoods=3, variables=["marker"], random_state=4)
    payload = pipeline.get_results()
    bundle = pipeline.save_results(tmp_path / "workflow.pyspace")
    restored = SpacePipeline.load_results(bundle)

    assert progress == [("census", 0.0), ("census", 1.0)]
    assert payload["kind"] == "pipeline"
    assert restored.state.input_type == "table"
    assert isinstance(restored.state.input_data, pd.DataFrame)
    assert restored.state.processing_history[-1]["operation"] == "census"


def test_pipeline_image_registration_optimization_and_census(tmp_path: Path) -> None:
    image_path = tmp_path / "objects.png"
    Image.fromarray(np.array([[0, 1, 1], [0, 1, 2], [0, 2, 2]], dtype=np.uint8)).save(image_path)
    pipeline = SpacePipeline().load_image(image_path)

    with pytest.raises(ValueError, match="n_neighborhoods"):
        pipeline.census(radii=[1.0])
    result = pipeline.census(radii=[1.0], n_neighborhoods=1, random_state=3)

    assert pipeline.state.input_type == "image"
    assert pipeline.state.input_data is None
    assert isinstance(result, CensusResult)
    assert len(result.neighborhoods) == 1


def test_pipeline_delegates_mi_pattern_and_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _pattern_census()
    patches = {"O1": pd.DataFrame({"Area": [1], "O1": [1], "Nbhd": [1]})}
    pipeline = SpacePipeline()
    pipeline.state.radii = [1.0]
    pipeline.state.variables = ["O1.1", "S1.1"]
    pipeline.state.census_results = CensusResult([], {}, ["O1.1", "S1.1"], patch_list=patches, census=frame)

    captured: dict[str, Any] = {}

    def fake_cismi(**kwargs: Any) -> dict[str, pd.DataFrame]:
        captured["cismi"] = kwargs
        return {"1.0": pd.DataFrame({"CisMI": [0.25]})}

    def fake_transmi(**kwargs: Any) -> dict[str, pd.DataFrame]:
        captured["transmi"] = kwargs
        return {"1.0": pd.DataFrame({"TransMI": [0.5]})}

    monkeypatch.setattr(pipeline_module, "measure_cisMI", fake_cismi)
    monkeypatch.setattr(pipeline_module, "measure_transMI", fake_transmi)
    cis = pipeline.measure_cisMI(depth=2, bootstraps=2, random_plan=[frame], allow_permutation_fallback=True)
    trans = pipeline.measure_transMI([frame, frame], pd.DataFrame({"group": ["A", "B"]}), depth=2, bootstraps=2)
    pattern = pipeline.learn_patterns({"O1": ["red"], "S1": ["blue"]}, random_state=3, som_reps=2, smooth_window=2)

    monkeypatch.setattr(
        pipeline_module,
        "map_pattern",
        lambda *_args, **_kwargs: (np.ones((2, 2, 1, 1), dtype=int), ["red"]),
    )
    mapped, palette = pipeline.map_patterns([[0.0, 100.0]], np.ones((2, 2, 1, 1)), {1.0: [1, 1, 1]}, ["red"])

    assert cis["1.0"].loc[0, "CisMI"] == 0.25
    assert trans["1.0"].loc[0, "TransMI"] == 0.5
    assert captured["cismi"]["patch_list"] is patches
    assert pattern.covariation_data.shape[0] == 12
    assert mapped.shape == (2, 2, 1, 1)
    assert palette == ["red"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"radii": []}, "radii"),
        ({"radii": [-1]}, "radii"),
        ({"n_neighborhoods": 0}, "n_neighborhoods"),
        ({"variables": []}, "variables"),
    ],
)
def test_pipeline_parameter_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SpacePipeline().set_parameters(**kwargs)


def test_pipeline_state_errors_and_bundle_validation(tmp_path: Path) -> None:
    pipeline = SpacePipeline()
    with pytest.raises(RuntimeError, match="before running census"):
        pipeline.census(radii=[1], n_neighborhoods=1)
    with pytest.raises(RuntimeError, match="Run census"):
        pipeline.measure_cisMI()
    with pytest.raises(ValueError, match="variables"):
        pipeline.learn_patterns({})
    with pytest.raises(RuntimeError, match="Run learn_patterns"):
        pipeline.map_patterns([], np.zeros((1, 1)), {}, [])

    not_pipeline = save_result({"kind": "other"}, tmp_path / "other.pyspace")
    with pytest.raises(ValueError, match="does not contain"):
        SpacePipeline.load_results(not_pipeline)
