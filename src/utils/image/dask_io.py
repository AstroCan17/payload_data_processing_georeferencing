"""
Lazy image stack reading with Dask.

Reads many TIFFs as a single Dask array without loading all into memory.
Each image is one chunk; data are read only when computed (e.g. by index or
when passed to pipeline steps that trigger computation).
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple, Union

import dask.array as da
import numpy as np
import rasterio as rio
from dask import delayed


def _read_frame(path: str) -> np.ndarray:
    """Read one band from a TIFF with rasterio. Used inside delayed()."""
    with rio.open(path, "r") as src:
        arr = src.read(1)
    return np.asarray(arr, dtype=arr.dtype)


def _get_shape_dtype(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Infer shape and dtype from the first image (metadata only, no pixel read)."""
    with rio.open(path, "r") as src:
        h, w = src.height, src.width
        dtype = np.dtype(src.dtypes[0])
    return (h, w), dtype


def imread_lazy(
    path_or_pattern: Union[str, List[str]],
    sort: bool = True,
) -> da.Array:
    """
    Load a stack of images as a lazy Dask array (frame, height, width).

    Images are not read into memory until the result is computed (e.g. by
    indexing, .compute(), or passing to a Dask-aware pipeline). Each image
    is one chunk.

    Args:
        path_or_pattern: Directory path (with *.tiff), glob pattern
            (e.g. "data/input_frames/**/image_*.tiff"), or list of paths.
        sort: If True, sort paths lexicographically (default: True).

    Returns:
        Dask array with shape (n_frames, height, width) and dtype from the
        first image (e.g. uint16). Chunks: (1, height, width) per frame.

    Example:
        >>> stack = imread_lazy("data/.../ImageFrames/")
        >>> stack.shape
        (49, 2984, 9344)
        >>> frame_0 = stack[0].compute()
        >>> batch = stack[10:20].compute()
    """
    if isinstance(path_or_pattern, list):
        paths = [os.path.abspath(p) for p in path_or_pattern]
    else:
        path_or_pattern = os.path.normpath(path_or_pattern)
        if os.path.isdir(path_or_pattern):
            pattern = os.path.join(path_or_pattern, "*.tiff")
        else:
            pattern = path_or_pattern
        paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files found for: {path_or_pattern}")

    if sort:
        paths = sorted(paths)

    shape_2d, dtype = _get_shape_dtype(paths[0])
    frame_shape = shape_2d  # (H, W), single band

    delayed_frames = [delayed(_read_frame)(p) for p in paths]
    arrays = [
        da.from_delayed(d, shape=frame_shape, dtype=dtype)
        for d in delayed_frames
    ]
    stack = da.stack(arrays, axis=0)
    return stack


def imread_lazy_with_paths(
    path_or_pattern: Union[str, List[str]],
    sort: bool = True,
) -> Tuple[da.Array, List[str]]:
    """
    Same as imread_lazy but also returns the ordered list of file paths.

    Useful when you need to associate computed frames with filenames (e.g.
    for metadata, IMU lookup, or writing outputs).

    Returns:
        (dask_array, paths): stack shape (n_frames, H, W), list of paths.
    """
    if isinstance(path_or_pattern, list):
        paths = [os.path.abspath(p) for p in path_or_pattern]
    else:
        path_or_pattern = os.path.normpath(path_or_pattern)
        if os.path.isdir(path_or_pattern):
            pattern = os.path.join(path_or_pattern, "*.tiff")
        else:
            pattern = path_or_pattern
        paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files found for: {path_or_pattern}")

    if sort:
        paths = sorted(paths)

    stack = imread_lazy(paths, sort=False)
    return stack, paths
