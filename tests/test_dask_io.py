"""Tests for lazy TIFF reading (src.utils.image.dask_io)."""

import os

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src.utils.image.dask_io import (
    _get_shape_dtype,
    _read_frame,
    imread_lazy,
    imread_lazy_with_paths,
)


pytestmark = pytest.mark.dask


@pytest.fixture
def tiff_dir(tmp_path):
    """Create a temp dir with 3 small uint16 TIFFs for testing."""
    h, w = 4, 6
    transform = from_bounds(0, 0, w, h, w, h)
    for i in range(3):
        path = tmp_path / f"frame_{i:03d}.tiff"
        data = np.arange(h * w, dtype=np.uint16).reshape(h, w) + (i * 100)
        with rio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=data.dtype,
            crs=CRS.from_epsg(32633),
            transform=transform,
        ) as dst:
            dst.write(data, 1)
    return tmp_path


@pytest.fixture
def unsorted_tiff_paths(tmp_path):
    """Create TIFFs whose file order differs from lexicographic order."""
    filenames = ["frame_010.tiff", "frame_002.tiff", "frame_001.tiff"]
    for value, name in enumerate(filenames):
        data = np.full((2, 3), value, dtype=np.uint16)
        with rio.open(
            tmp_path / name,
            "w",
            driver="GTiff",
            height=2,
            width=3,
            count=1,
            dtype=data.dtype,
            transform=from_bounds(0, 0, 3, 2, 3, 2),
        ) as dst:
            dst.write(data, 1)
    return [str(tmp_path / name) for name in filenames]


def test_get_shape_dtype(tiff_dir):
    first = sorted(tiff_dir.glob("*.tiff"))[0]
    shape, dtype = _get_shape_dtype(str(first))
    assert shape == (4, 6)
    assert dtype == np.uint16


def test_read_frame(tiff_dir):
    first = sorted(tiff_dir.glob("*.tiff"))[0]
    arr = _read_frame(str(first))
    assert arr.shape == (4, 6)
    assert arr.dtype == np.uint16
    with rio.open(first) as src:
        expected = src.read(1)
    np.testing.assert_array_equal(arr, expected)


def test_imread_lazy_returns_dask_array(tiff_dir):
    stack = imread_lazy(str(tiff_dir))
    assert stack.shape == (3, 4, 6)
    assert stack.dtype == np.uint16
    assert stack.chunks[0] == (1, 1, 1)


def test_imread_lazy_compute_matches_direct_read(tiff_dir):
    paths = sorted(tiff_dir.glob("*.tiff"))
    stack = imread_lazy([str(p) for p in paths])
    for i, path in enumerate(paths):
        computed = stack[i].compute()
        with rio.open(path) as src:
            direct = src.read(1)
        np.testing.assert_array_equal(computed, direct)


def test_imread_lazy_with_paths(tiff_dir):
    stack, paths = imread_lazy_with_paths(str(tiff_dir))
    assert stack.shape[0] == len(paths) == 3
    assert all(os.path.isfile(p) for p in paths)
    np.testing.assert_array_equal(stack[1].compute(), _read_frame(paths[1]))


def test_imread_lazy_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files found"):
        imread_lazy(str(tmp_path))


def test_imread_lazy_list_of_paths(tiff_dir):
    paths = sorted(tiff_dir.glob("*.tiff"))
    path_list = [str(p) for p in paths]
    stack = imread_lazy(path_list)
    assert stack.shape == (3, 4, 6)
    np.testing.assert_array_equal(stack[0].compute(), _read_frame(path_list[0]))


def test_imread_lazy_glob_matches_directory_order(tiff_dir):
    from_directory = imread_lazy(str(tiff_dir))
    from_glob = imread_lazy(str(tiff_dir / "*.tiff"))
    np.testing.assert_array_equal(from_directory.compute(), from_glob.compute())


def test_imread_lazy_with_paths_list_matches_directory_order(tiff_dir):
    directory_stack, directory_paths = imread_lazy_with_paths(str(tiff_dir))
    explicit_stack, explicit_paths = imread_lazy_with_paths(directory_paths)
    assert explicit_paths == directory_paths
    np.testing.assert_array_equal(directory_stack.compute(), explicit_stack.compute())


def test_imread_lazy_sort_false_preserves_input_order(unsorted_tiff_paths):
    stack = imread_lazy(unsorted_tiff_paths, sort=False)
    assert list(unsorted_tiff_paths) != sorted(unsorted_tiff_paths)
    for index, path in enumerate(unsorted_tiff_paths):
        np.testing.assert_array_equal(stack[index].compute(), _read_frame(path))


def test_imread_lazy_single_file_returns_single_frame_stack(tiff_dir):
    first = sorted(tiff_dir.glob("*.tiff"))[0]
    stack = imread_lazy(str(first))
    assert stack.shape == (1, 4, 6)
    np.testing.assert_array_equal(stack[0].compute(), _read_frame(str(first)))


def test_imread_lazy_slice_compute_matches_expected_frames(tiff_dir):
    paths = sorted(tiff_dir.glob("*.tiff"))
    stack = imread_lazy([str(path) for path in paths])
    batch = stack[1:3].compute()
    expected = np.stack([_read_frame(str(path)) for path in paths[1:3]], axis=0)
    np.testing.assert_array_equal(batch, expected)


def test_imread_lazy_invalid_path_type_raises():
    with pytest.raises(TypeError, match="string or list of strings"):
        imread_lazy(123)  # type: ignore[arg-type]


def test_imread_lazy_invalid_path_list_entry_raises():
    with pytest.raises(TypeError, match="list entries must be strings"):
        imread_lazy(["valid.tiff", 123])  # type: ignore[list-item]


def test_imread_lazy_unmatched_glob_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files found"):
        imread_lazy(str(tmp_path / "*.tiff"))


def test_imread_lazy_mixed_shapes_raise(tmp_path):
    first = tmp_path / "frame_a.tiff"
    second = tmp_path / "frame_b.tiff"

    with rio.open(
        first,
        "w",
        driver="GTiff",
        height=4,
        width=6,
        count=1,
        dtype=np.uint16,
        transform=from_bounds(0, 0, 6, 4, 6, 4),
    ) as dst:
        dst.write(np.ones((4, 6), dtype=np.uint16), 1)

    with rio.open(
        second,
        "w",
        driver="GTiff",
        height=5,
        width=6,
        count=1,
        dtype=np.uint16,
        transform=from_bounds(0, 0, 6, 5, 6, 5),
    ) as dst:
        dst.write(np.ones((5, 6), dtype=np.uint16), 1)

    with pytest.raises(ValueError, match="same shape"):
        imread_lazy(str(tmp_path))


def test_imread_lazy_mixed_dtypes_raise(tmp_path):
    first = tmp_path / "frame_u16.tiff"
    second = tmp_path / "frame_u8.tiff"

    with rio.open(
        first,
        "w",
        driver="GTiff",
        height=4,
        width=6,
        count=1,
        dtype=np.uint16,
        transform=from_bounds(0, 0, 6, 4, 6, 4),
    ) as dst:
        dst.write(np.ones((4, 6), dtype=np.uint16), 1)

    with rio.open(
        second,
        "w",
        driver="GTiff",
        height=4,
        width=6,
        count=1,
        dtype=np.uint8,
        transform=from_bounds(0, 0, 6, 4, 6, 4),
    ) as dst:
        dst.write(np.ones((4, 6), dtype=np.uint8), 1)

    with pytest.raises(ValueError, match="same dtype"):
        imread_lazy(str(tmp_path))
