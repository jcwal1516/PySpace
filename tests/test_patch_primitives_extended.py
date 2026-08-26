from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyspace.core.patch_measurements import (
    eligible_seed_coordinates,
    image_group,
    isodata_threshold,
    measure_image_neighborhood,
    normalize_images,
    normalize_os_pairs,
    scalar_thresholds,
)
from pyspace.core.patch_summary import random_census, summarize_patches


def test_image_normalization_handles_supported_dimensions_and_types() -> None:
    arrays, image_types = normalize_images(
        {
            "O1": np.ones((2, 3), dtype=np.uint8),
            "O2": np.ones((2, 3, 1), dtype=np.uint8),
            "O3": np.ones((2, 3, 1, 1), dtype=np.uint8),
            "S4": np.ones((2, 3), dtype=np.uint8),
            "S5": np.ones((2, 3, 2), dtype=np.uint8),
            "S6": np.ones((2, 3, 1, 2), dtype=np.uint8),
        }
    )

    assert image_group("O12") == ("O", 12)
    assert arrays["O1"].shape == (2, 3, 1, 1)
    assert arrays["O2"].shape == (2, 3, 1, 1)
    assert arrays["O3"].shape == (2, 3, 1, 1)
    assert arrays["S4"].shape == (2, 3, 1, 1)
    assert arrays["S5"].shape == (2, 3, 1, 2)
    assert arrays["S6"].shape == (2, 3, 1, 2)
    assert image_types == {
        "O1": "object",
        "O2": "object",
        "O3": "object",
        "S4": "scalar",
        "S5": "scalar",
        "S6": "scalar",
    }


@pytest.mark.parametrize(
    "images, match",
    [
        ({}, "non-empty"),
        ({"bad": np.ones((2, 2))}, "must match"),
        ({"O1": np.array([])}, "non-empty"),
        ({"O1": np.ones((2,))}, "2D, 3D, or 4D"),
        ({"O1": np.full((2, 2), "x")}, "numeric"),
        ({"O1": np.array([[np.nan]])}, "finite"),
        ({"O1": np.ones((2, 2, 1, 2))}, "exactly one channel"),
        ({"O1": np.ones((2, 2)), "S1": np.ones((3, 2))}, "identical"),
    ],
)
def test_image_normalization_rejects_malformed_inputs(images: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        normalize_images(images)  # type: ignore[arg-type]


def test_isodata_threshold_and_scalar_thresholds_cover_histogram_cases() -> None:
    assert isodata_threshold(np.full((2, 2), 7)) == 7
    assert isodata_threshold(np.array([0, 10])) == 1
    assert 1 <= isodata_threshold(np.array([1, 2, 3, 200, 255])) <= 255

    images = {
        "O1": np.ones((2, 2, 1, 1), dtype=np.uint8),
        "S2": np.stack([np.full((2, 2, 1), 3), np.full((2, 2, 1), 9)], axis=-1),
    }
    assert scalar_thresholds(images, {"O1": "object", "S2": "scalar"}) == {"O1": None, "S2": [3, 9]}

    for invalid in (np.array([-1]), np.array([256]), np.array([1.5])):
        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            isodata_threshold(invalid)


def test_seed_eligibility_honors_allowed_object_ids_and_scalar_channels() -> None:
    objects = np.array([[1, 2], [0, 2]], dtype=np.uint8)[:, :, None, None]
    scalar = np.zeros((2, 2, 1, 2), dtype=np.uint8)
    scalar[0, 0, 0, 0] = 10
    scalar[1, 0, 0, 1] = 20
    images = {"O1": objects, "S2": scalar, "S3": scalar.copy()}
    types = {"O1": "object", "S2": "scalar", "S3": "scalar"}
    thresholds = {"O1": None, "S2": [5, 15], "S3": None}

    coordinates = eligible_seed_coordinates(
        images,
        types,
        thresholds,
        os_pairs=None,
        background_value=0,
        allowed_seed_values={"O1": [2], "S2": [2]},
    )
    assert coordinates.tolist() == [[0, 1, 0], [1, 0, 0], [1, 1, 0]]

    none_allowed = eligible_seed_coordinates(
        {"O1": objects},
        {"O1": "object"},
        {"O1": None},
        os_pairs=None,
        background_value=0,
        allowed_seed_values={"O1": []},
    )
    assert none_allowed.size == 0

    with pytest.raises(ValueError, match="outside 1..2"):
        eligible_seed_coordinates(
            {"S2": scalar},
            {"S2": "scalar"},
            {"S2": [5, 15]},
            os_pairs=None,
            background_value=0,
            allowed_seed_values={"S2": [3]},
        )


def test_object_scalar_pair_normalization_labels_and_validates_matrices() -> None:
    objects = np.array([[0, 2], [5, 5]], dtype=np.uint8)[:, :, None, None]
    scalar = np.zeros((2, 2, 1, 2), dtype=np.uint8)
    images = {"O1": objects, "S1": scalar}

    assert normalize_os_pairs(None, images) is None
    normalized = normalize_os_pairs({"O1": np.array([[1, 0], [0, 1]])}, images)
    assert normalized is not None
    assert normalized["O1"].index.tolist() == ["O1.2", "O1.5"]
    assert normalized["O1"].columns.tolist() == ["S1.1", "S1.2"]

    original = pd.DataFrame([[1]], index=["custom"], columns=["S1.1"])
    copied = normalize_os_pairs({"O1": original}, images)
    assert copied is not None and copied["O1"] is not original

    with pytest.raises(ValueError, match="requires images"):
        normalize_os_pairs({"O2": np.array([[1]])}, images)
    with pytest.raises(ValueError, match="two-dimensional"):
        normalize_os_pairs({"O1": np.array([1, 0])}, images)
    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        normalize_os_pairs({"O1": np.array([[1]])}, images)


def test_neighborhood_measurement_covers_object_scalar_and_scalar_only_groups() -> None:
    objects = np.array([[1, 1, 0], [1, 2, 2], [0, 2, 2]], dtype=np.uint8)[:, :, None, None]
    linked = np.arange(9, dtype=np.uint8).reshape(3, 3, 1, 1)
    independent = np.array([[0, 10, 0], [10, 10, 0], [0, 0, 0]], dtype=np.uint8)[:, :, None, None]

    patches, included = measure_image_neighborhood(
        {"O1": objects, "S1": linked, "S2": independent},
        center=np.array([1, 1, 0]),
        radius=np.array([2, 2, 1]),
        thresholds={"O1": None, "S1": [4], "S2": [5]},
        background_value=0,
    )

    assert included.shape == (3, 3, 1)
    assert set(patches) == {"O1", "S2.1"}
    assert set(patches["O1"].columns) == {"Area", "O1", "S1.1"}
    assert patches["O1"]["Area"].sum() == 7
    assert patches["S2.1"]["Area"].sum() == int(included.sum())

    empty, _ = measure_image_neighborhood(
        {"O1": np.zeros((2, 2, 1, 1), dtype=np.uint8)},
        center=np.array([0, 0, 0]),
        radius=np.ones(3),
        thresholds={"O1": None},
        background_value=0,
    )
    assert empty["O1"].to_dict("records") == [{"Area": 0, "O1": 0}]

    with pytest.raises(RuntimeError, match="Missing thresholds"):
        measure_image_neighborhood(
            {"S2": independent},
            center=np.array([1, 1, 0]),
            radius=np.ones(3),
            thresholds={"S2": None},
            background_value=0,
        )


def test_patch_summary_scalar_empty_and_validation_paths() -> None:
    scalar = pd.DataFrame({"Area": [2, 1, 1], "S2.10": [8.0, 3.0, 5.0], "S2.2": [4.0, 6.0, 1.0], "Nbhd": [1, 1, 2]})
    summary = summarize_patches({"S2": scalar})
    assert summary.columns.tolist() == ["S2.2", "S2.10"]
    assert summary.loc[0, "S2.10"] == pytest.approx(11 / 3)
    assert summarize_patches({}).columns.tolist() == ["Nbhd"]

    linked_only = summarize_patches(
        {"S1.1": scalar.rename(columns={"S2.10": "S1.1", "S2.2": "S1.2"})},
        {"O1": pd.DataFrame([[1]], index=["O1.1"], columns=["S1.1"])},
    )
    assert linked_only.empty

    with pytest.raises(TypeError, match="DataFrame"):
        summarize_patches({"O1": np.ones((2, 2))})
    with pytest.raises(ValueError, match="Nbhd.*Area"):
        summarize_patches({"O1": pd.DataFrame({"Area": [1]})})
    with pytest.raises(ValueError, match="cannot be missing"):
        summarize_patches({"O1": pd.DataFrame({"Area": [np.nan], "Nbhd": [1]})})
    with pytest.raises(ValueError, match="multiple object"):
        summarize_patches({"O1": pd.DataFrame({"Area": [1], "Nbhd": [1], "O1": [1], "O2": [1]})})
    with pytest.raises(ValueError, match="two-dimensional"):
        summarize_patches({}, {"O1": np.array([1])})


def test_random_census_slow_path_is_seeded_and_preserves_neighborhoods() -> None:
    patches = {
        "O1": pd.DataFrame(
            {
                "Area": [2, 1, 3, 2],
                "O1": [1, 2, 1, 2],
                "S1.1": [4.0, 8.0, 12.0, 10.0],
                "Nbhd": [1, 1, 2, 2],
            }
        )
    }
    osp = {"O1": pd.DataFrame([[1], [1]], index=["O1.1", "O1.2"], columns=["S1.1"])}

    first = random_census(patches, osp, rng=np.random.default_rng(42))
    second = random_census(patches, osp, rng=np.random.default_rng(42))
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (2, 4)
    np.testing.assert_allclose(first[["O1.1", "O1.2"]].sum(axis=1), 100.0)

    empty = {"O1": pd.DataFrame({"Area": [0], "O1": [0], "Nbhd": [1]})}
    assert random_census(empty, rng=np.random.default_rng(1)).shape == (1, 0)
    with pytest.raises(ValueError, match="Nbhd.*Area"):
        random_census({"O1": pd.DataFrame({"Area": [1]})})


def test_random_census_unit_area_fast_path_reuses_validated_cache() -> None:
    size = 10_000
    frame = pd.DataFrame(
        {
            "Area": np.ones(size),
            "O1": np.tile([1, 2], size // 2),
            "S1.1": np.tile([2.0, 4.0], size // 2),
            "Nbhd": np.repeat(np.arange(1, 101), 100),
        }
    )
    osp = {"O1": pd.DataFrame([[1], [1]], index=["O1.1", "O1.2"], columns=["S1.1"])}

    first = random_census({"O1": frame}, osp, rng=np.random.default_rng(7))
    second = random_census({"O1": frame}, osp, rng=np.random.default_rng(8))

    assert first.shape == second.shape == (100, 4)
    assert "_pyspace_unit_area_random_cache" in frame.attrs
    np.testing.assert_allclose(first[["O1.1", "O1.2"]].sum(axis=1), 100.0)
