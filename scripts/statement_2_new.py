"""
This module contains functions for image processing and georeferencing operations.

Key functions:
- convert_to_8bit: Converts images to 8-bit with percentile-based contrast stretching
- match_histogram: Matches histogram of one image to a reference image
- sharpen_image: Applies sharpening filter to enhance image details
- apply_stretch: Performs contrast stretching using percentile-based clipping

The module provides utilities for:
- Image format conversion and enhancement
- Histogram matching and normalization
- Image sharpening and filtering
- Contrast adjustment and stretching

Dependencies:
- OpenCV (cv2) for image processing
- NumPy for array operations
- scikit-image for exposure adjustments
- rasterio for geospatial operations
"""

import gc
import glob
import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
from pyproj import Transformer
from rasterio import windows
from rasterio.crs import CRS
from rasterio.transform import Affine
from skimage import exposure
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("Georeferencing Logger")  # Logger


def convert_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Convert image to 8-bit with percentile-based contrast stretching.

    Args:
        img: Input image array

    Returns:
        8-bit image array with enhanced contrast
    """
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(
        img, in_range=(float(p2), float(p98)), out_range=(0.0, 255.0)
    ).astype("uint8")


def match_histogram(img1: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Match histogram of img1 to reference image.

    Args:
        img1: Source image to modify
        ref: Reference image to match

    Returns:
        Image with matched histogram
    """
    return exposure.match_histograms(img1, ref)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter to enhance image details.

    Args:
        img: Input image array

    Returns:
        Sharpened image array
    """
    img = img.astype("float64")
    sharpen_kernel = np.array([[0, -1.5, 0], [-1.5, 7, -1.5], [0, -1.5, 0]])

    img = cv2.filter2D(src=img, ddepth=-1, kernel=sharpen_kernel)
    img[img > 2**16 - 1] = 2**16 - 1
    img[img < 0] = 0
    img = img.astype("uint16")
    return img


def apply_stretch(img: np.ndarray) -> np.ndarray:
    """
    Apply contrast stretching using percentile-based clipping.

    Args:
        img: Input image array

    Returns:
        Contrast-enhanced 8-bit image
    """
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(
        img, in_range=(float(p2), float(p98)), out_range=(0.0, 255.0)
    ).astype("uint8")


def apply_clahe(channel: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Args:
        channel: Input image channel

    Returns:
        CLAHE-enhanced image channel
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(channel.astype("uint8"))


class Georeferencing:
    """Class for handling georeferencing operations on image data."""

    def __init__(
        self,
        input_path: str,
        channels: List[str],
        output_path: str,
        apply_lever_offset: bool = False,
    ):
        """
        Initialize georeferencing parameters.

        Args:
            input_path: Path to input images
            channels: List of channels to process
            output_path: Path for output files
            apply_lever_offset: Whether to apply sensor lever arm and mount center lever offset
        """
        self.input_path = input_path
        self.channels = channels
        self.output_path = os.path.join(output_path, "02_Statement_2")
        os.makedirs(self.output_path, exist_ok=True)
        self.raw_image_list = sorted(glob.glob(input_path + "/*.tiff"))

        # Image dimensions in pixels (can be provided by user or metadata)
        self.pix_num_x: int = 9344
        self.pix_num_y: int = 2984

        # Binning, focal length and pixel size (example values, units must be consistent)
        self.binning: int = 4
        self.focal_length: float = 457e-3  # meters (457 mm)
        self.pix_height: float = 3.2e-6  # pixel size in meters
        self.pix_width: float = 3.2e-6  # pixel size in meters

        # Mount center (INS mount) and sensor (camera) lever-arm values (meters)
        # Assumed to be defined in "body" coordinate system
        self.mount_center_lever: np.ndarray = np.array([0.000, 0.129, 0.103])
        self.sensor_lever_arm: np.ndarray = np.array([0.113, 0.128, -0.175])

        # Ground elevation (meters)
        self.ground_elv: float = 850.0
        self.min_height: Optional[float] = None

        # Apply lever offset if IMU coordinates are not relative to image center
        self.apply_lever_offset = apply_lever_offset
        self.out_folder_raw_georef = os.path.join(
            self.output_path, "01_ImageFrames_Georeferenced"
        )
        self.out_folder_seperated = os.path.join(self.output_path, "02_Band_Seperated")

        os.makedirs(self.out_folder_raw_georef, exist_ok=True)
        os.makedirs(self.out_folder_seperated, exist_ok=True)

        self.imu_path = "data/IMU_DataSheet.txt"
        self.df_imu: Optional[pd.DataFrame] = None

    def img_generator(self) -> Generator[Tuple[str, np.ndarray, CRS], None, None]:
        """
        Generator function to yield image data from files.

        Yields:
            Tuple containing:
                - Image name (str)
                - Image data (np.ndarray)
                - CRS information (CRS)
        """
        for img_path in self.raw_image_list:
            img_name = os.path.basename(str(img_path))
            with rio.open(img_path, "r") as f:
                img = f.read(1)
                crs = f.crs
                yield img_name, img, crs


# ... rest of the code ...
