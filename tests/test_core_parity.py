import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyspace.core.census import census_table, standardize_censuses
from pyspace.core.distributions import (
    r_round_column,
    smooth_dist,
    total_comp_bins,
)
from pyspace.core.operations import patch_3D
from pyspace.core.patch_summary import summarize_patches
from pyspace.core.r_measure_cismi import measure_cisMI
from pyspace.core.r_measure_transmi import measure_transMI
from pyspace.parity import CORE_ORACLE_DIR, PARITY_DATA_DIR, UPSTREAM_SPACE_COMMIT
from tests.parity_cases import _assert_frame_equal_canonical, transmi_inputs


def _oracle_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(Path(CORE_ORACLE_DIR) / name)


def test_core_oracle_metadata_is_pinned_to_target_space_commit():
    metadata = json.loads((Path(CORE_ORACLE_DIR) / "metadata.json").read_text())

    assert metadata["upstream_commit"] == UPSTREAM_SPACE_COMMIT
    assert metadata["numeric_tolerance"] == 1e-10


def test_census_table_uses_the_pinned_r_contract_by_default() -> None:
    ordinary_coordinates = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "object_id": [1, 2]})

    with pytest.raises(ValueError, match="Object table"):
        census_table(ordinary_coordinates, radii=[1.0], sample_size=[1])


def test_total_comp_bins_uses_max_over_variable_subsets_for_custom_ranges():
    assert (
        total_comp_bins(
            dimension=2,
            bins_per_var=3,
            min_per_var=[80, 0, 0],
            max_per_var=[100, 50, 100],
        )
        == 7
    )


def test_round_column_returns_original_values_when_range_is_constant():
    values = np.array([4.25, 4.25, 4.25])

    np.testing.assert_array_equal(
        r_round_column(values, col_min=4.25, col_max=4.25, bin_num=5, return_bin_values=False),
        values,
    )


def test_smooth_dist_matches_r_chao_jost_counts_and_expand_grid_order():
    joint_dist = pd.DataFrame({"O1.1": [0.0, 1.0], "freq": [1, 2]})
    min_max = pd.DataFrame([[0.0], [3.0]], columns=pd.Index(["O1.1"]))

    smoothed = smooth_dist(joint_dist, bin_num=4, min_max=min_max, full_dist=True)
    assert isinstance(smoothed, pd.DataFrame)
    oracle = _oracle_csv("smooth_dist_1d.csv")

    _assert_frame_equal_canonical(smoothed.reset_index(drop=True), oracle)

    joint_2d = pd.DataFrame({"O1.1": [0.0], "S1.1": [10.0], "freq": [1]})
    min_max_2d = pd.DataFrame([[0.0, 10.0], [1.0, 30.0]], columns=pd.Index(["O1.1", "S1.1"]))
    grid = smooth_dist(joint_2d, bin_num=[2, 3], min_max=min_max_2d, full_dist=True)
    assert isinstance(grid, pd.DataFrame)

    assert grid[["O1.1", "S1.1"]].to_numpy().tolist() == [
        [0.0, 10.0],
        [1.0, 10.0],
        [0.0, 20.0],
        [1.0, 20.0],
        [0.0, 30.0],
        [1.0, 30.0],
    ]


def test_cismi_binds_depth_outputs_with_missing_variable_cells_as_na():
    base = np.tile(np.array([80.0, 20.0, 60.0, 40.0]), 10)
    census = pd.DataFrame(
        {
            "O1.1": base,
            "O1.2": 100.0 - base,
            "S1.1": np.linspace(1.0, 4.0, len(base)),
            "Radius": 10.0,
        }
    )

    with pytest.warns(RuntimeWarning):
        result = measure_cisMI(
            census,
            patch_list=None,
            depth=2,
            radii=[10.0],
            bootstraps=2,
            max_bins=5,
            allow_permutation_fallback=True,
        )

    radius_df = result["10"]
    depth_one_rows = radius_df[radius_df["VB"].isna()]

    assert len(depth_one_rows) == 3


def test_cismi_uses_explicit_random_plan_without_mutating_global_rng():
    census = pd.DataFrame(
        {
            "O1.1": [0.0, 25.0, 50.0, 75.0, 100.0] * 4,
            "O1.2": [100.0, 75.0, 50.0, 25.0, 0.0] * 4,
            "Radius": 10.0,
        }
    )
    random_plan = [
        census.drop(columns="Radius").iloc[::-1].reset_index(drop=True),
        census.drop(columns="Radius").sample(frac=1.0, random_state=7).reset_index(drop=True),
    ]
    np.random.seed(2026)
    expected_global_draw = np.random.random(4)
    np.random.seed(2026)

    result = measure_cisMI(
        census,
        patch_list=None,
        depth=2,
        radii=[10.0],
        bootstraps=2,
        random_censuses={"10": random_plan},
    )

    np.testing.assert_array_equal(np.random.random(4), expected_global_draw)
    assert list(result) == ["10"]


def test_cismi_explicit_random_plan_is_parallel_equivalent_and_validates_cores() -> None:
    census = pd.DataFrame(
        {
            "O1.1": [0.0, 25.0, 50.0, 75.0, 100.0] * 4,
            "O1.2": [100.0, 75.0, 50.0, 25.0, 0.0] * 4,
            "Radius": 10.0,
        }
    )
    plan = [census.drop(columns="Radius").iloc[::-1].reset_index(drop=True)] * 2

    serial = measure_cisMI(census, patch_list=None, depth=2, radii=[10.0], bootstraps=2, random_censuses=plan, cores=1)
    threaded = measure_cisMI(
        census, patch_list=None, depth=2, radii=[10.0], bootstraps=2, random_censuses=plan, cores=2
    )

    pd.testing.assert_frame_equal(serial["10"], threaded["10"])
    with pytest.raises(ValueError, match="cores"):
        measure_cisMI(census, patch_list=None, depth=2, radii=[10.0], bootstraps=2, random_censuses=plan, cores=0)


def test_transmi_returns_radius_to_final_dataframe_without_internal_pair_mi_columns():
    base = np.tile(np.array([0.0, 100.0, 50.0, 25.0]), 10)
    censuses = []
    for offset in (0.0, 5.0, 20.0, 25.0):
        o11 = np.clip(base + offset, 0.0, 100.0)
        censuses.append(
            pd.DataFrame(
                {
                    "O1.1": o11,
                    "O1.2": 100.0 - o11,
                    "S1.1": np.linspace(0.0, 1.0, len(o11)) + offset,
                    "Radius": 10.0,
                }
            )
        )
    groups = pd.DataFrame({"Status": ["A", "A", "B", "B"]})

    result = measure_transMI(
        censuses=censuses,
        groups=groups,
        depth=2,
        radii=[10.0],
        bootstraps=2,
        max_bins=5,
        cores=1,
    )

    assert set(result) == {"10"}
    radius_df = result["10"]
    assert isinstance(radius_df, pd.DataFrame)
    assert not any(col.startswith("MI_Im") for col in radius_df.columns)
    assert {"VA", "VB", "TransMI_Status", "Zscore_Status", "Pvalue_Status", "Padjust_Status"}.issubset(
        radius_df.columns
    )
    assert len(radius_df) == 6
    assert radius_df["VB"].isna().sum() == 3


def test_transmi_explicit_permutations_are_parallel_equivalent_and_rng_local():
    base = np.tile(np.array([0.0, 100.0, 50.0, 25.0]), 5)
    censuses = [
        pd.DataFrame({"O1.1": base + offset, "O1.2": 100.0 - base, "Radius": 10.0}) for offset in (0.0, 1.0, 10.0, 11.0)
    ]
    groups = pd.DataFrame({"Status": ["A", "A", "B", "B"]})
    permutations = [np.array([0, 3, 2, 5, 1, 4]), np.array([2, 1, 5, 3, 0, 4])]
    np.random.seed(2027)
    expected_global_draw = np.random.random(4)
    np.random.seed(2027)

    serial = measure_transMI(
        censuses,
        groups,
        depth=2,
        radii=[10.0],
        bootstraps=2,
        cores=1,
        permutation_indices=permutations,
    )
    for backend in ("thread", "process"):
        parallel = measure_transMI(
            censuses,
            groups,
            depth=2,
            radii=[10.0],
            bootstraps=2,
            cores=2,
            permutation_indices=permutations,
            parallel_backend=backend,
        )
        pd.testing.assert_frame_equal(serial["10"], parallel["10"])
    np.testing.assert_array_equal(np.random.random(4), expected_global_draw)


def test_transmi_matches_the_pinned_non_degenerate_r_oracle() -> None:
    censuses, groups = transmi_inputs()
    random_plan = json.loads((PARITY_DATA_DIR / "random_plans.json").read_text(encoding="utf-8"))
    permutations = [np.asarray(step, dtype=int) for step in random_plan["transmi_pair_permutation_steps"]]

    result = measure_transMI(
        censuses,
        groups,
        depth=2,
        radii=[1.1],
        bootstraps=2,
        max_bins=5,
        cores=1,
        permutation_indices=permutations,
    )

    _assert_frame_equal_canonical(result["1.1"], _oracle_csv("transmi_1.1.csv"))


def test_census_table_r_style_call_exposes_census_and_radius_patch_list():
    object_table = pd.DataFrame(
        {
            "X": [0.0, 1.0, 4.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0],
            "Object": [9, 1, 2],
        }
    )
    extra_objects = pd.DataFrame(
        {
            "X": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "Y": [0.0] * 8,
            "Z": [0.0] * 8,
            "Object": [3, 4, 5, 6, 7, 8, 10, 11],
        }
    )
    object_table = pd.concat([object_table, extra_objects], ignore_index=True)

    result = census_table(
        object_table,
        radii=[1.1],
        sample_size=[1],
        seed_points=[9],
    )

    assert result.patch_list is not None
    assert result.census is not None
    assert set(result.patch_list) == {"1.1"}
    assert list(result.patch_list["1.1"]) == ["O1"]

    expected = _oracle_csv("census_table_r_style.csv")
    _assert_frame_equal_canonical(result.census.reset_index(drop=True), expected)


def test_standardize_censuses_matches_r_column_union_and_natural_order():
    left = pd.DataFrame({"O1.10": [10.0], "O1.1": [1.0], "X": [2.0], "Radius": [5.0]})
    right = pd.DataFrame({"O1.2": [2.0], "S1.1": [3.0], "Y": [4.0], "Z": [0.0], "Radius": [5.0]})

    standardized = standardize_censuses([left, right])

    expected_columns = ["O1.1", "O1.2", "O1.10", "S1.1", "X", "Y", "Z", "Radius"]
    assert [list(frame.columns) for frame in standardized] == [expected_columns, expected_columns]
    assert standardized[0].loc[0, "O1.2"] == 0
    assert standardized[1].loc[0, "X"] == 0
    assert list(left.columns) == ["O1.10", "O1.1", "X", "Radius"]


def test_summarize_patches_matches_committed_r_oracle_fixture():
    patches = {
        "O1": pd.DataFrame(
            {
                "Area": [2, 1, 0, 3, 1],
                "O1": [1, 2, 0, 1, 2],
                "S1.1": [10, 9, 0, 12, 1],
                "Nbhd": [1, 1, 2, 3, 3],
            }
        )
    }
    osp = {"O1": pd.DataFrame([[1], [0]], index=pd.Index(["O1.1", "O1.2"]), columns=pd.Index(["S1.1"]))}

    actual = summarize_patches(patches, osp)
    expected = _oracle_csv("summarize_patches.csv")

    _assert_frame_equal_canonical(actual, expected)


def test_patch_3d_matches_space_component_ranking_and_sizes() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True  # SPACE connects corner-touching voxels.
    mask[3, 0, 0] = True

    result = patch_3D(mask)

    assert set(result) == {"index", "size"}
    np.testing.assert_array_equal(result["size"][mask], np.array([2, 2, 1]))
    np.testing.assert_array_equal(result["index"][mask], np.array([1, 1, 2]))


def test_patch_3d_rejects_non_boolean_or_non_3d_masks() -> None:
    with pytest.raises(ValueError, match="3D boolean"):
        patch_3D(np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="3D boolean"):
        patch_3D(np.ones((2, 2, 2), dtype=np.uint8))
