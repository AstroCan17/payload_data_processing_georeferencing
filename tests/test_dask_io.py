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
