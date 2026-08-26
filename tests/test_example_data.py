import pandas as pd
import pytest

from pyspace import load_example_data
from pyspace.core.r_measure_cismi import measure_cisMI
from pyspace.io.validation import validate_inputs


def test_load_example_data_shapes_and_metadata():
    bundle = load_example_data()
    census = bundle["census"]
    metadata = bundle["metadata"]

    assert census.shape == (40, 7)
    assert metadata["dataset"] == "synthetic_tissue"
    assert metadata["radii_pixels"] == [10.0, 20.0]
    assert "no human or clinical" in metadata["description"]


def test_validate_inputs_reports_radius_counts():
    bundle = load_example_data()
    report = validate_inputs(bundle["census"])

    assert report["valid"]
    assert report["stats"]["row_count"] == 40
    assert report["stats"]["radius_counts"][10.0] == 20


def test_measure_cismi_requires_patch_list_or_explicit_fallback():
    census = pd.DataFrame(
        {
            "O1.1": [50.0, 60.0, 40.0],
            "O1.2": [50.0, 40.0, 60.0],
            "Radius": [10.0, 10.0, 10.0],
        }
    )
    with pytest.raises(ValueError):
        measure_cisMI(census, patch_list=None, depth=2, radii=[10.0], bootstraps=1, max_bins=5)

    with pytest.warns(RuntimeWarning):
        fallback = measure_cisMI(
            census,
            patch_list=None,
            depth=2,
            radii=[10.0],
            bootstraps=1,
            max_bins=5,
            allow_permutation_fallback=True,
        )
    assert isinstance(fallback, dict)
