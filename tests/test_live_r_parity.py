from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyspace.core.census import census_table
from pyspace.core.distributions import build_dist, r_round_column, smooth_dist, total_comp_bins
from pyspace.core.operations import calc_vol, calc_vols
from pyspace.core.patch_summary import random_census, summarize_patches
from pyspace.core.r_measure_cismi import _r_entropy, measure_cisMI
from pyspace.core.r_measure_transmi import measure_transMI
from pyspace.parity import PARITY_DATA_DIR, UPSTREAM_SPACE_COMMIT, check_upstream_source, pristine_upstream_checkout
from tests.parity_cases import transmi_inputs

ROOT = Path(__file__).resolve().parents[1]
R_ORACLE_SCRIPT = ROOT / "scripts" / "generate_core_parity_oracles.R"
pytestmark = pytest.mark.live_r


@pytest.fixture(scope="module")
def r_repo() -> Iterator[Path]:
    configured = os.environ.get("SPACE_R_REPO")
    if configured:
        path = Path(configured)
        failures = [check for check in check_upstream_source(path) if not check.passed]
        if failures:
            pytest.fail(f"SPACE_R_REPO must be a pristine pinned checkout: {failures}")
        yield path
        return
    with pristine_upstream_checkout() as checkout:
        yield checkout


def _assert_frame_equal_canonical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_index_equal(left.columns, right.columns)
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _object_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X": [0.0, 1.0, 4.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "Y": [0.0] * 11,
            "Z": [0.0] * 11,
            "Object": [9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
        }
    )


def _patches() -> dict[str, pd.DataFrame]:
    return {
        "O1": pd.DataFrame(
            {
                "Area": [2, 1, 0, 3, 1],
                "O1": [1, 2, 0, 1, 2],
                "S1.1": [10, 9, 0, 12, 1],
                "Nbhd": [1, 1, 2, 3, 3],
            }
        )
    }


def _osp() -> dict[str, pd.DataFrame]:
    return {
        "O1": pd.DataFrame(
            [[1], [0]],
            index=pd.Index(["O1.1", "O1.2"]),
            columns=pd.Index(["S1.1"]),
        )
    }


def _stable_random_patches() -> dict[str, pd.DataFrame]:
    return {
        "O1": pd.DataFrame(
            {
                "Area": [1, 2, 3, 1, 2, 3],
                "O1": [1, 1, 1, 1, 1, 1],
                "S1.1": [2, 4, 6, 2, 4, 6],
                "Nbhd": [1, 1, 1, 2, 2, 2],
            }
        )
    }


def _cismi_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    rows: list[dict[str, Any]] = []
    patch_rows_1: list[dict[str, Any]] = []
    patch_rows_2: list[dict[str, Any]] = []
    for nbhd in range(1, 41):
        rows.append({"O1.1": 100.0, "O2.1": 100.0, "X": 0.0, "Y": 0.0, "Z": 0.0, "Radius": 1.1})
        patch_rows_1.append({"Area": 1, "O1": 1, "Nbhd": nbhd})
        patch_rows_2.append({"Area": 1, "O2": 1, "Nbhd": nbhd})
    return pd.DataFrame(rows), {"1.1": {"O1": pd.DataFrame(patch_rows_1), "O2": pd.DataFrame(patch_rows_2)}}


def _generate_live_oracle(tmp_path: Path, r_repo: Path) -> Path:
    out_dir = tmp_path / "r-oracle"
    result = subprocess.run(
        ["Rscript", str(R_ORACLE_SCRIPT), "--r-repo", str(r_repo), "--out-dir", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"R oracle failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert out_dir.exists(), f"R oracle did not create {out_dir}"
    return out_dir


def test_live_r_oracle_matches_python_core_outputs(tmp_path: Path, r_repo: Path) -> None:
    oracle_dir = _generate_live_oracle(tmp_path, r_repo)

    metadata = json.loads((oracle_dir / "metadata.json").read_text())
    assert metadata["upstream_commit"] == UPSTREAM_SPACE_COMMIT
    assert metadata["numeric_tolerance"] == 1e-10
    assert metadata["fixtures"]
    for filename, expected_hash in metadata["fixtures"].items():
        assert hashlib.sha256((oracle_dir / filename).read_bytes()).hexdigest() == expected_hash

    utils = json.loads((oracle_dir / "utils.json").read_text())
    assert calc_vol([2, 2, 1], dims=[20, 20, 5]) == utils["calc_vol_ellipsoid"]
    assert calc_vols([[1, 1, 1], [2, 2, 1]], dims=[20, 20, 5]) == utils["calc_vols"]
    assert total_comp_bins(2, 3, [80, 0, 0], [100, 50, 100]) == utils["total_comp_bins_custom"]
    assert np.isclose(_r_entropy(np.array([[1, 1], [1, 2], [2, 1], [2, 2], [2, 2]]), 2), utils["entropy_2d"])

    round_expected = _read_csv(oracle_dir / "round_column.csv")
    round_actual = pd.DataFrame(
        {
            "bin_id": r_round_column(
                np.array([0.0, 1.0, 2.9, 4.1, 5.0]),
                col_min=0.0,
                col_max=5.0,
                bin_num=4,
                return_bin_values=False,
            ),
            "bin_value": r_round_column(
                np.array([0.0, 1.0, 2.9, 4.1, 5.0]),
                col_min=0.0,
                col_max=5.0,
                bin_num=4,
                return_bin_values=True,
            ),
        }
    )
    _assert_frame_equal_canonical(round_actual, round_expected)

    census = pd.DataFrame({"O1.1": [0, 0, 1, 1, 1], "S1.1": [10, 20, 10, 20, 20]})
    _assert_frame_equal_canonical(
        build_dist(census, ["O1.1", "S1.1"], "all"), _read_csv(oracle_dir / "build_dist_2d.csv")
    )

    joint_2d = pd.DataFrame({"O1.1": [0.0, 1.0], "S1.1": [10.0, 20.0], "freq": [1, 2]})
    min_max_2d = pd.DataFrame([[0.0, 10.0], [1.0, 30.0]], columns=pd.Index(["O1.1", "S1.1"]))
    smooth_full = smooth_dist(joint_2d, bin_num=[2, 3], min_max=min_max_2d, full_dist=True)
    assert isinstance(smooth_full, pd.DataFrame)
    _assert_frame_equal_canonical(smooth_full, _read_csv(oracle_dir / "smooth_dist_2d.csv"))

    smooth_freq = smooth_dist(joint_2d, bin_num=[2, 3], min_max=min_max_2d, full_dist=False)
    np.testing.assert_allclose(
        smooth_freq, _read_csv(oracle_dir / "smooth_dist_2d_freq.csv")["freq"].to_numpy(), atol=1e-10
    )

    _assert_frame_equal_canonical(
        summarize_patches(_patches(), _osp()), _read_csv(oracle_dir / "summarize_patches.csv")
    )

    np.random.seed(1)
    _assert_frame_equal_canonical(
        random_census(_stable_random_patches(), _osp()),
        _read_csv(oracle_dir / "random_census.csv"),
    )

    census_result = census_table(_object_table(), radii=[1.1], sample_size=[1], seed_points=[9])
    assert census_result.census is not None
    _assert_frame_equal_canonical(census_result.census, _read_csv(oracle_dir / "census_table_r_style.csv"))


def test_live_r_oracle_matches_python_mi_outputs(tmp_path: Path, r_repo: Path) -> None:
    oracle_dir = _generate_live_oracle(tmp_path, r_repo)

    cismi_census, cismi_patch_list = _cismi_inputs()
    cismi_result = measure_cisMI(
        cismi_census, cismi_patch_list, depth=2, radii=[1.1], bootstraps=2, max_bins=5, cores=1
    )
    _assert_frame_equal_canonical(cismi_result["1.1"], _read_csv(oracle_dir / "cismi_1.1.csv"))

    transmi_censuses, groups = transmi_inputs()
    random_plan = json.loads((PARITY_DATA_DIR / "random_plans.json").read_text())
    permutation_steps = [np.asarray(step, dtype=int) for step in random_plan["transmi_pair_permutation_steps"]]
    transmi_result = measure_transMI(
        transmi_censuses,
        groups,
        depth=2,
        radii=[1.1],
        bootstraps=2,
        max_bins=5,
        cores=1,
        permutation_indices=permutation_steps,
    )
    _assert_frame_equal_canonical(transmi_result["1.1"], _read_csv(oracle_dir / "transmi_1.1.csv"))
