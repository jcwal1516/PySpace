from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from pyspace.core.census_sampling import create_neighborhoods, normalize_radii, normalize_sample_sizes
from pyspace.io.image_loader import _as_uint8, load_image, read_image_array
from pyspace.io.table_loader import (
    _profile_by_coordinates,
    _profile_by_counts,
    load_coordinate_table,
    load_table,
    read_table,
)


def test_read_table_dispatches_all_supported_formats_and_indexes_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame({"id": [10, 20], "x": [1.0, 2.0], "y": [3.0, 4.0]})
    paths = {
        ".csv": tmp_path / "table.csv",
        ".tsv": tmp_path / "table.tsv",
        ".txt": tmp_path / "table.txt",
        ".xlsx": tmp_path / "table.xlsx",
        ".parquet": tmp_path / "table.parquet",
    }
    frame.to_csv(paths[".csv"], index=False)
    frame.to_csv(paths[".tsv"], sep="\t", index=False)
    frame.to_csv(paths[".txt"], sep="\t", index=False)
    frame.to_excel(paths[".xlsx"], index=False)
    frame.to_parquet(paths[".parquet"], index=False)

    for suffix, path in paths.items():
        loaded = read_table(path)
        pd.testing.assert_frame_equal(loaded, frame, check_dtype=suffix not in {".xlsx"})
    indexed = read_table(paths[".parquet"], index_col=0)
    assert indexed.index.tolist() == [10, 20]


def test_coordinate_and_space_table_contracts(tmp_path: Path) -> None:
    coordinate_path = tmp_path / "coordinates.csv"
    pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0], "Object": [1, 2]}).to_csv(coordinate_path, index=False)
    coordinates = load_coordinate_table(coordinate_path)
    objects = load_table(coordinate_path, "O", None, None)

    link_path = tmp_path / "links.csv"
    pd.DataFrame({"S1.1": [1, 0]}, index=["O1.1", "O1.2"]).to_csv(link_path)
    links = load_table(link_path, "L", None, None)

    assert coordinates.shape == (2, 3)
    assert objects.columns[:3].tolist() == ["X", "Y", "Z"]
    assert links.index.tolist() == ["O1.1", "O1.2"]


def test_profile_table_matches_image_by_coordinates_and_counts(tmp_path: Path) -> None:
    labels = np.array([[[1], [2]], [[2], [2]]], dtype=int)
    palette = ["#ff0000", "#00ff00"]
    coordinate_profile = pd.DataFrame(
        {
            "Object": [9, 8],
            "Row": [1, 1],
            "Column": [1, 2],
            "Z": [1, 1],
            "Count": [1, 3],
            "Marker": [0.2, 0.8],
        }
    )
    coordinate_path = tmp_path / "coordinate-profile.csv"
    coordinate_profile.to_csv(coordinate_path, index=False)
    matched_coordinates = load_table(coordinate_path, "P", labels, palette)

    count_profile = pd.DataFrame({"Object": [2, 1], "Count": [3, 1], "Marker": [0.8, 0.2]})
    count_path = tmp_path / "count-profile.csv"
    count_profile.to_csv(count_path, index=False)
    matched_counts = load_table(count_path, "P", labels, palette)

    assert matched_coordinates["Object"].tolist() == [1, 2]
    assert matched_coordinates["Marker"].tolist() == [0.2, 0.8]
    assert matched_counts["Object"].tolist() == [1, 2]
    assert matched_counts["Marker"].tolist() == [0.2, 0.8]


def test_image_scalar_conversion_and_generic_png_read(tmp_path: Path) -> None:
    np.testing.assert_array_equal(_as_uint8(np.array([0.0, 0.5, 1.0])), [0, 128, 255])
    np.testing.assert_array_equal(_as_uint8(np.array([0, 65535], dtype=np.uint16)), [0, 255])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _as_uint8(np.array([2.0]))
    with pytest.raises(TypeError, match="dtype"):
        _as_uint8(np.array(["x"]))

    scalar_path = tmp_path / "scalar.tiff"
    tifffile.imwrite(scalar_path, np.array([[[0, 255], [128, 64]]], dtype=np.uint8), photometric="minisblack")
    scalar = load_image(scalar_path, "S", None, num_chs=1, keep_chs=[1])
    assert scalar.shape == (2, 2, 1, 1)

    png_path = tmp_path / "image.png"
    Image.fromarray(np.array([[1, 2], [3, 4]], dtype=np.uint8)).save(png_path)
    np.testing.assert_array_equal(read_image_array(png_path), [[1, 2], [3, 4]])


def test_neighborhood_sampling_handles_tables_images_paths_and_empty_inputs(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "object_id": [1, 2, 1],
            "marker": [0.1, 0.2, 0.3],
        }
    )
    table_result = create_neighborhoods(
        frame,
        [1.1],
        seed_points=np.array([[1.0, 0.0]]),
        pixel_to_micron_factor=2.0,
    )
    table_path = tmp_path / "cells.csv"
    frame.to_csv(table_path, index=False)
    path_result = create_neighborhoods(table_path, 1.1, max_neighborhoods=2, random_state=3)
    image_result = create_neighborhoods(np.array([[0, 1], [2, 0]]), 1.5, random_state=2)
    rgb_result = create_neighborhoods(np.dstack([np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))]), 1.0)
    volume_result = create_neighborhoods(np.ones((2, 2, 5)), 1.0, max_neighborhoods=1)
    empty_result = create_neighborhoods(np.zeros((2, 2), dtype=int), [1.0, 2.0])

    assert len(table_result[1.1]) == 1
    assert len(path_result[1.1]) == 2
    assert len(image_result[1.5]) == 2
    assert rgb_result[1.0]
    assert volume_result[1.0][0].is_3d
    assert empty_result == {1.0: [], 2.0: []}


def test_coordinate_census_has_an_explicit_non_parity_entry_point() -> None:
    from pyspace.core.table_census import census_coordinates

    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "object_id": [1, 2, 1], "marker": [1.0, 2.0, 3.0]}
    )

    result = census_coordinates(frame, radii=[1.0], sample_size=2, variables=["marker"], random_state=4)

    assert result.metadata["r_parity_mode"] == "non_parity_python_coordinates"
    assert result.census is not None
    assert result.census.columns.tolist() == ["O1.1", "O1.2", "marker", "X", "Y", "Z", "Radius"]


def test_loader_and_sampling_validation_errors(tmp_path: Path) -> None:
    bad_coordinates = tmp_path / "bad.csv"
    pd.DataFrame({"x": ["a"], "y": [1]}).to_csv(bad_coordinates, index=False)
    with pytest.raises(ValueError, match="numeric"):
        load_coordinate_table(bad_coordinates)
    with pytest.raises(ValueError, match="table_type"):
        load_table(bad_coordinates, "X", None, None)
    with pytest.raises(ValueError, match="finite positive"):
        normalize_radii([1.0, np.nan])
    with pytest.raises(ValueError, match="does not match"):
        normalize_sample_sizes([1], 2)
    with pytest.raises(ValueError, match="positive integers"):
        normalize_sample_sizes([0], 1)
    with pytest.raises(ValueError, match="max_neighborhoods"):
        create_neighborhoods(np.ones((2, 2)), 1.0, max_neighborhoods=-1)
    with pytest.raises(ValueError, match="shape"):
        create_neighborhoods(np.ones((2, 2)), 1.0, seed_points=np.ones((1, 3)))


def test_table_loader_reports_missing_files_and_invalid_coordinate_contracts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Table file not found"):
        read_table(tmp_path / "missing.csv")

    missing_axes = tmp_path / "missing-axes.csv"
    pd.DataFrame({"x": [1.0]}).to_csv(missing_axes, index=False)
    with pytest.raises(ValueError, match="requires X/Y"):
        load_coordinate_table(missing_axes)

    missing_value = tmp_path / "missing-value.csv"
    pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]}).to_csv(missing_value, index=False)
    with pytest.raises(ValueError, match="must not contain missing"):
        load_coordinate_table(missing_value)

    profile_without_object = tmp_path / "profile.csv"
    pd.DataFrame({"Count": [1]}).to_csv(profile_without_object, index=False)
    with pytest.raises(ValueError, match="Object column"):
        load_table(profile_without_object, "P", None, None)

    profile = tmp_path / "matched-profile.csv"
    pd.DataFrame({"Object": [1], "Count": [1]}).to_csv(profile, index=False)
    with pytest.raises(ValueError, match="both img and col_pal"):
        load_table(profile, "P", img=np.ones((1, 1), dtype=int), col_pal=None)


def test_profile_matching_helpers_reject_ambiguous_or_incomplete_mappings() -> None:
    labels = np.array([[1, 0], [0, 2]], dtype=int)
    palette = ["red", "blue"]

    with pytest.raises(ValueError, match="X/Y/Z or Row/Column/Z"):
        _profile_by_coordinates(pd.DataFrame({"Object": [1]}), labels, palette)
    with pytest.raises(ValueError, match="integer row index"):
        _profile_by_coordinates(
            pd.DataFrame(
                {"Object": [1, 2], "Row": [1, 2], "Column": [1, 2], "Z": [1, 1]},
                index=["first", "second"],
            ),
            labels[..., None],
            palette,
        )
    with pytest.raises(ValueError, match="identify every"):
        _profile_by_coordinates(
            pd.DataFrame({"Object": [1], "X": [1], "Y": [1], "Z": [1]}),
            np.array([[[1], [2]]], dtype=int),
            palette,
        )

    with pytest.raises(ValueError, match="non-unique counts"):
        _profile_by_counts(pd.DataFrame({"Count": [1, 1]}), labels, palette)
    with pytest.raises(ValueError, match="absent from profile"):
        _profile_by_counts(pd.DataFrame({"Count": [99, 98]}), labels, palette)

    disconnected = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=int)
    by_components = _profile_by_counts(
        pd.DataFrame({"Count": [2], "Marker": [3.0]}),
        disconnected,
        ["red"],
    )
    assert by_components.to_dict("records") == [{"Count": 2, "Marker": 3.0, "Object": 1}]


def test_image_loader_validates_paths_modes_pages_and_channels(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        read_image_array(tmp_path / "missing.tif")

    unsupported = tmp_path / "image.bmp"
    unsupported.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        read_image_array(unsupported)
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(unsupported, "O", None, 1)

    grayscale = tmp_path / "grayscale.tif"
    tifffile.imwrite(grayscale, np.ones((2, 2), dtype=np.uint8), photometric="minisblack")
    np.testing.assert_array_equal(read_image_array(grayscale), np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="img_type"):
        load_image(grayscale, "X", None, 1)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="positive integer"):
        load_image(grayscale, "S", None, 0)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        load_image(grayscale, "O", None, 1)

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_path = tmp_path / "rgb.tif"
    tifffile.imwrite(rgb_path, rgb, photometric="rgb")
    with pytest.raises(ValueError, match="Unknown background"):
        load_image(rgb_path, "O", "not-a-color", 1)
    with pytest.raises(ValueError, match="two-dimensional grayscale"):
        load_image(rgb_path, "S", None, 1)

    pages_path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(pages_path) as writer:
        for value in range(3):
            writer.write(np.full((2, 2), value, dtype=np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="divisible"):
        load_image(pages_path, "S", None, 2)
    with pytest.raises(ValueError, match="one-based channel"):
        load_image(pages_path, "S", None, 1, keep_chs=[])
    with pytest.raises(ValueError, match="one-based channel"):
        load_image(pages_path, "S", None, 1, keep_chs=[2])


def test_image_loader_rejects_nonfinite_float_and_mismatched_page_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite values"):
        _as_uint8(np.array([np.nan], dtype=float))

    mismatched = tmp_path / "mismatched.tif"
    with tifffile.TiffWriter(mismatched) as writer:
        writer.write(np.zeros((2, 2), dtype=np.uint8), photometric="minisblack")
        writer.write(np.zeros((3, 2), dtype=np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="same spatial dimensions"):
        load_image(mismatched, "S", None, 1)

    rgb_mismatched = tmp_path / "rgb-mismatched.tif"
    with tifffile.TiffWriter(rgb_mismatched) as writer:
        writer.write(np.zeros((2, 2, 3), dtype=np.uint8), photometric="rgb")
        writer.write(np.zeros((3, 2, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="same spatial dimensions"):
        load_image(rgb_mismatched, "O", None, 1)
