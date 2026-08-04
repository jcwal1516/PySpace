from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import pyspace
from pyspace.core.census import export_multi_image_census_dataframe
from pyspace.core.patch_measurements import eligible_seed_coordinates, normalize_images, normalize_os_pairs
from pyspace.core.patch_summary import summarize_patches

from .tutorial_parity_helpers import (
    apply_object_permutation,
    coordinate_overlap_stats,
    distribution_gap,
    infer_object_permutation,
    structure_diff,
)


def test_structure_diff_reports_missing_and_extra_columns():
    left = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    right = pd.DataFrame({"A": [1], "B": [2], "D": [4]})

    diff = structure_diff(left, right)

    assert diff["left_shape"] == [1, 3]
    assert diff["right_shape"] == [1, 3]
    assert diff["shape_equal"] is True
    assert diff["column_order_equal"] is False
    assert diff["missing_in_left"] == ["D"]
    assert diff["extra_in_left"] == ["C"]


def test_coordinate_overlap_stats_counts_intersection_and_ratios():
    left = pd.DataFrame({"X": [1, 2, 3], "Y": [10, 20, 30], "Z": [0, 0, 0]})
    right = pd.DataFrame({"X": [2, 3, 4], "Y": [20, 30, 40], "Z": [0, 0, 0]})

    stats = coordinate_overlap_stats(left, right)

    assert stats["left_unique"] == 3
    assert stats["right_unique"] == 3
    assert stats["intersection"] == 2
    assert stats["overlap_vs_left"] == 2 / 3
    assert stats["overlap_vs_right"] == 2 / 3


def test_distribution_gap_summarizes_mean_and_std_differences():
    left = pd.DataFrame(
        {
            "O1.1": [0.0, 10.0, 20.0],
            "O1.2": [100.0, 90.0, 80.0],
            "X": [1, 2, 3],
        }
    )
    right = pd.DataFrame(
        {
            "O1.1": [0.0, 5.0, 10.0],
            "O1.2": [100.0, 95.0, 90.0],
            "X": [1, 2, 3],
        }
    )

    gap = distribution_gap(left, right)

    assert gap["n_columns"] == 2
    assert gap["avg_abs_mean_diff"] > 0
    assert gap["avg_abs_std_diff"] > 0
    assert len(gap["top_abs_mean_diffs"]) == 2


def test_infer_and_apply_object_permutation_updates_linked_columns():
    right = pd.DataFrame(
        {
            "O2.1": [90.0, 85.0],
            "O2.2": [10.0, 15.0],
            "O2.1_S1.1": [9.0, 8.5],
            "O2.2_S1.1": [1.0, 1.5],
        }
    )
    # Left is label-swapped relative to right.
    left = pd.DataFrame(
        {
            "O2.1": [10.0, 15.0],
            "O2.2": [90.0, 85.0],
            "O2.1_S1.1": [1.0, 1.5],
            "O2.2_S1.1": [9.0, 8.5],
        }
    )

    mapping = infer_object_permutation(left, right, prefix="O2")
    assert mapping == {1: 2, 2: 1}

    relabeled = apply_object_permutation(left, prefix="O2", mapping=mapping)

    pd.testing.assert_series_equal(relabeled["O2.1"], right["O2.1"], check_names=False)
    pd.testing.assert_series_equal(relabeled["O2.2"], right["O2.2"], check_names=False)
    pd.testing.assert_series_equal(relabeled["O2.1_S1.1"], right["O2.1_S1.1"], check_names=False)
    pd.testing.assert_series_equal(relabeled["O2.2_S1.1"], right["O2.2_S1.1"], check_names=False)


def test_census_image_applies_object_remap_before_measuring_abundance():
    obj = np.array(
        [
            [7, 7, 8],
            [7, 8, 8],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = pyspace.census_image(
        {"O1": obj},
        radii=[2.0],
        sample_size=[1],
        seed_points=np.array([[1, 0]]),
        object_remap={"O1": {7: 8, 8: 8}},
    )
    census = export_multi_image_census_dataframe(
        result.neighborhoods, result.variables, include_coordinates=True, r_compatible=True
    )

    assert result.metadata["object_remap_applied_to_labels"] is True
    assert "O1.8" in census.columns
    assert "O1.7" not in census.columns
    assert census.loc[0, "O1.8"] == 100.0


def test_census_image_requires_the_upstream_sample_size_parameter() -> None:
    with pytest.raises(TypeError, match="sample_size"):
        cast(Any, pyspace.census_image)({"O1": np.ones((2, 2), dtype=np.uint8)}, radii=[1.0])


def test_census_image_cores_preserve_order_and_results() -> None:
    image = {"O1": np.array([[1, 1, 0], [1, 2, 2], [0, 2, 2]], dtype=np.uint8)}
    seed_points = np.array([[0, 0], [1, 1]])

    serial = pyspace.census_image(image, radii=[1.0], sample_size=[2], seed_points=seed_points, cores=1)
    threaded = pyspace.census_image(image, radii=[1.0], sample_size=[2], seed_points=seed_points, cores=2)

    assert serial.census is not None and threaded.census is not None
    pd.testing.assert_frame_equal(serial.census, threaded.census)
    assert serial.metadata["cores"] == 1
    assert threaded.metadata["cores"] == 2


def test_census_image_patch_list_matches_exported_census_after_object_remap():
    obj = np.array(
        [
            [7, 7, 8],
            [7, 8, 8],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = pyspace.census_image(
        {"O1": obj},
        radii=[2.0],
        sample_size=[1],
        seed_points=np.array([[1, 0]]),
        object_remap={"O1": {7: 8, 8: 8}},
    )
    census = export_multi_image_census_dataframe(
        result.neighborhoods, result.variables, include_coordinates=True, r_compatible=True
    )
    assert result.patch_list is not None
    patch_list = result.patch_list
    patch_census = summarize_patches({"O1": patch_list["O1"].drop(columns=["Radius"])})

    assert patch_census.loc[0, "O1.8"] == census.loc[0, "O1.8"]


def test_census_image_measures_linked_scalars_with_numpy_link_table():
    obj = np.ones((3, 3), dtype=np.uint8)
    scalar = np.zeros((3, 3, 2), dtype=np.uint8)
    scalar[:, :, 0] = 10
    scalar[:, :, 1] = 20

    result = pyspace.census_image(
        {"O1": obj, "S1": scalar},
        radii=[2.0],
        sample_size=[1],
        seed_points=np.array([[1, 1]]),
        os_pairs={"O1": np.array([[1, 1]])},
    )
    census = export_multi_image_census_dataframe(
        result.neighborhoods, result.variables, include_coordinates=True, r_compatible=True
    )

    assert census.loc[0, "O1.1_S1.1"] == 10.0
    assert census.loc[0, "O1.1_S1.2"] == 20.0


def test_census_image_uses_one_seed_sequence_prefix_across_radii():
    obj = np.ones((5, 5), dtype=np.uint8)

    result = pyspace.census_image(
        {"O1": obj},
        radii=[1.0, 2.0],
        sample_size=[5, 3],
        random_state=11,
    )

    centers_by_radius = {
        radius: [tuple(nbhd.center) for nbhd in result.neighborhoods if nbhd.radius == radius] for radius in (1.0, 2.0)
    }

    assert centers_by_radius[2.0] == centers_by_radius[1.0][:3]


def test_r_space_seed_eligibility_excludes_linked_scalar_only_pixels():
    loaded_images = {
        "O1": np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        "S1": np.array(
            [
                [[0], [0], [0]],
                [[0], [0], [0]],
                [[0], [0], [255]],
            ],
            dtype=np.uint8,
        ),
        "S3": np.array(
            [
                [[0], [0], [0]],
                [[0], [200], [0]],
                [[0], [0], [0]],
            ],
            dtype=np.uint8,
        ),
    }
    bin_thresholds = {"O1": None, "S1": [128], "S3": [128]}
    os_pairs = {
        "O1": pd.DataFrame([[1]], index=pd.Index(["O1.1"]), columns=pd.Index(["S1.1"])),
    }

    images, image_types = normalize_images(loaded_images)
    pairs = normalize_os_pairs(os_pairs, images)
    eligible = eligible_seed_coordinates(images, image_types, bin_thresholds, pairs, 0)[:, :2]

    assert eligible.tolist() == [[0, 0], [1, 1]]


def test_permutation_fallback_provenance_is_explicitly_non_parity():
    from pyspace.parity import cismi_provenance

    provenance = cismi_provenance(
        allow_permutation_fallback=True,
        patch_list_available=False,
    )

    assert provenance["r_parity_mode"] == "exploratory"
    assert provenance["null_model"] == "within_column_permutation"
