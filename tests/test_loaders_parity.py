from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from pyspace import load_image, load_table


def test_load_object_image_uses_r_column_major_palette_order(tmp_path: Path) -> None:
    black = np.array([0, 0, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8)
    image = np.array([[black, red], [blue, red]], dtype=np.uint8)
    path = tmp_path / "objects.tif"
    tifffile.imwrite(path, image, photometric="rgb")

    loaded, palette = load_image(path, "O", "black", num_chs=1)

    assert loaded.shape == (2, 2, 1, 1)
    assert palette == ["#0000FF", "#FF0000"]
    np.testing.assert_array_equal(loaded[:, :, 0, 0], np.array([[0, 2], [1, 2]]))


def test_load_scalar_image_interleaves_channels_within_z(tmp_path: Path) -> None:
    path = tmp_path / "scalars.tif"
    pages = [np.full((2, 3), value, dtype=np.uint8) for value in (1, 2, 3, 4)]
    with tifffile.TiffWriter(path) as writer:
        for page in pages:
            writer.write(page, photometric="minisblack")

    loaded = load_image(path, "S", bkgd_col=None, num_chs=2)

    assert loaded.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(loaded[0, 0], np.array([[1, 2], [3, 4]]))


def test_load_table_preserves_link_labels_and_adds_object_z(tmp_path: Path) -> None:
    link_path = tmp_path / "links.csv"
    pd.DataFrame({"S1.1": [1, 0]}, index=["O1.1", "O1.2"]).to_csv(link_path)
    object_path = tmp_path / "objects.csv"
    pd.DataFrame({"X": [2], "Y": [3], "Object": [1], "S1.1": [4.5]}).to_csv(object_path, index=False)

    link = load_table(link_path, "L", None, None)
    objects = load_table(object_path, "O", None, None)

    assert isinstance(link, pd.DataFrame)
    assert list(link.index) == ["O1.1", "O1.2"]
    assert list(link.columns) == ["S1.1"]
    pd.testing.assert_frame_equal(
        objects,
        pd.DataFrame({"X": [2], "Y": [3], "Z": [1], "Object": [1], "S1.1": [4.5]}),
    )


def test_profile_table_without_mapping_inputs_warns_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "profile.csv"
    pd.DataFrame({"Object": [2, 1], "Count": [4, 3]}).to_csv(path, index=False)

    with pytest.warns(RuntimeWarning, match="might not match"):
        result = load_table(path, "P", None, None)

    assert result["Object"].tolist() == [1, 2]


def test_loaders_reject_unknown_modes_and_extensions(tmp_path: Path) -> None:
    path = tmp_path / "table.unknown"
    path.write_text("X,Y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="table_type must be P, O, or L"):
        load_table(path, "X", None, None)
    with pytest.raises(ValueError, match="Unsupported table format"):
        load_table(path, "O", None, None)
