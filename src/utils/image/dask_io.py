"""
Lazy image stack reading with Dask.

Reads many TIFFs as a single Dask array without loading all into memory.
Each image is one chunk; data are read only when computed (e.g. by index or
when passed to pipeline steps that trigger computation).
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List, Tuple, Union

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


def _resolve_paths(
    path_or_pattern: Union[str, List[str]],
    sort: bool = True,
) -> List[str]:
    """Resolve a directory, glob pattern, single path, or path list to files."""
    if isinstance(path_or_pattern, list):
        if not path_or_pattern:
            raise FileNotFoundError("No files found for: []")
        if not all(isinstance(path, str) for path in path_or_pattern):
            raise TypeError("path_or_pattern list entries must be strings")
        paths = [os.path.abspath(path) for path in path_or_pattern]
    elif isinstance(path_or_pattern, str):
        normalized = os.path.normpath(path_or_pattern)
        if os.path.isdir(normalized):
            pattern = os.path.join(normalized, "*.tiff")
            paths = glob.glob(pattern)
        elif os.path.isfile(normalized):
            paths = [os.path.abspath(normalized)]
        else:
            paths = glob.glob(normalized)
    else:
        raise TypeError("path_or_pattern must be a string or list of strings")

    if not paths:
        raise FileNotFoundError(f"No files found for: {path_or_pattern}")

    return sorted(paths) if sort else paths


def _validate_stack_metadata(paths: Iterable[str]) -> Tuple[Tuple[int, ...], np.dtype]:
    """Ensure every file in the stack has the same 2D shape and dtype."""
    iterator = iter(paths)
    first_path = next(iterator)
    expected_shape, expected_dtype = _get_shape_dtype(first_path)

    for path in iterator:
        shape, dtype = _get_shape_dtype(path)
        if shape != expected_shape:
            raise ValueError(
                "All files must have the same shape; "
                f"expected {expected_shape} but got {shape} for {path}"
            )
        if dtype != expected_dtype:
            raise ValueError(
                "All files must have the same dtype; "
                f"expected {expected_dtype} but got {dtype} for {path}"
            )

    return expected_shape, expected_dtype


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
    paths = _resolve_paths(path_or_pattern, sort=sort)
    shape_2d, dtype = _validate_stack_metadata(paths)
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
    paths = _resolve_paths(path_or_pattern, sort=sort)
    stack = imread_lazy(paths, sort=False)
    return stack, paths
