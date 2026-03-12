"""
This module contains functions for georeferencing operations.

Key functions:
- read_imu_data: Reads IMU data from files
- calculate_rotation_matrix: Calculates rotation matrices from IMU data
- calculate_effective_gsd: Calculates effective ground sample distance
- process_raw_images: Processes raw images with georeferencing

The module provides utilities for:
- IMU data processing
- Coordinate transformations
- Georeferencing calculations
- Image processing with geospatial information

Dependencies:
- NumPy for array operations
- rasterio for geospatial operations
- pandas for data handling
- scipy for spatial transformations
"""

import glob
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.errors
from rasterio.transform import from_origin
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.image import apply_stretch, convert_to_8bit, sharpen_image

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("Georeferencing Logger")  # Logger


class Georeferencing:
    """Class for handling georeferencing operations"""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        output_epsg: int = 32632,
        binning: int = 4,
    ):
        """
        Initialize georeferencing parameters

        Args:
            input_path: Path to input images
            output_path: Path for output files
            output_epsg: Output coordinate system EPSG code
            binning: Binning factor for pixel aggregation
        """
        self.input_path = input_path
        self.output_path = os.path.join(
            output_path, "04_Georeferencing")
        os.makedirs(self.output_path, exist_ok=True)
        self.output_epsg = output_epsg
        self.binning = binning

        self.raw_image_list = sorted(
            glob.glob(
                os.path.join(
                    self.input_path,
                    "*.tiff")))
        if not self.raw_image_list:
            LOG.warning("No images found in input path")
            return

        self.name_first = os.path.basename(self.raw_image_list[0])
        self.name_first = self.name_first.split("_")[1:]
        self.name_first = "_".join(self.name_first)

    def read_imu_data(self, imu_file: str) -> pd.DataFrame:
        """
        Read IMU data from file.

        Args:
            imu_file: Path to IMU data file

        Returns:
            DataFrame containing IMU data
        """
        try:
            imu_data = pd.read_csv(imu_file, sep="\t", header=None)
            imu_data.columns = ["timestamp", "roll", "pitch", "yaw"]
            return imu_data
        except (pd.errors.EmptyDataError, IOError) as e:
            LOG.error("Error reading IMU data: %s", e)
            return pd.DataFrame()

    def calculate_rotation_matrix(
        self, roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """
        Calculate rotation matrix from Euler angles.

        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees

        Returns:
            3x3 rotation matrix
        """
        r = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True)
        return r.as_matrix()

    def calculate_effective_gsd(
        self, focal_length: float, altitude: float, pixel_size: float
    ) -> float:
        """
        Calculate effective ground sample distance.

        Args:
            focal_length: Camera focal length in mm
            altitude: Flight altitude in m
            pixel_size: Pixel size in mm

        Returns:
            Effective GSD in meters
        """
        return (altitude * pixel_size) / focal_length

    def process_raw_images(
        self,
        raw_img_list: List[str],
        output_path: str,
        output_epsg: int,
        binning: int = 4,
        imu_file: Optional[str] = None,
    ) -> None:
        """
        Process raw images with georeferencing.

        Args:
            raw_img_list: List of raw image paths
            output_path: Path for output files
            output_epsg: Output coordinate system EPSG code
            binning: Binning factor for pixel aggregation
            imu_file: Optional path to IMU data file
        """
        if not raw_img_list:
            LOG.warning("No images to process")
            return

        progress_bar = tqdm(
            total=len(raw_img_list),
            desc="Processing images",
            unit="image",
            position=0,
            leave=True,
        )

        # Read IMU data if available
        imu_data = None
        if imu_file:
            imu_data = self.read_imu_data(imu_file)

        for img_path in raw_img_list:
            try:
                with rio.open(img_path) as src:
                    # Read image data
                    image = src.read(1)
                    profile = src.profile.copy()

                    # Get image metadata
                    height, width = image.shape
                    transform = src.transform

                    # Apply binning if specified
                    if binning > 1:
                        image = image[::binning, ::binning]
                        height, width = image.shape
                        transform = from_origin(
                            transform[2],
                            transform[5],
                            transform[0] * binning,
                            -transform[4] * binning,
                        )

                    # Calculate georeferencing parameters
                    if imu_data is not None:
                        # Find closest IMU data point
                        timestamp = float(
                            os.path.basename(img_path).split("_")[0])
                        closest_idx = (
                            imu_data["timestamp"] -
                            timestamp).abs().idxmin()
                        # type: ignore
                        row = imu_data.iloc[closest_idx]

                        # Calculate rotation matrix
                        rotation_matrix = self.calculate_rotation_matrix(
                            row["roll"], row["pitch"], row["yaw"])

                        # Update transform with rotation
                        # This is a simplified example - actual implementation would depend on
                        # specific requirements and coordinate system
                        # transformations
                        transform = transform * rio.transform.Affine.rotation(  # type: ignore
                            np.deg2rad(row["yaw"])
                        )

                    # Update profile
                    profile.update(
                        height=height,
                        width=width,
                        transform=transform,
                        crs=f"EPSG:{output_epsg}",
                    )

                    # Write output file
                    output_file = os.path.join(
                        output_path, f"georef_{
                            os.path.basename(img_path)}")

                    with rio.open(output_file, "w", **profile) as dst:
                        dst.write(image, 1)

                    # Create visualization
                    vis_file = os.path.join(
                        output_path, f"georef_{
                            os.path.splitext(
                                os.path.basename(img_path))[0]}_8b.png", )
                    cv2.imwrite(  # type: ignore
                        vis_file, convert_to_8bit(
                            apply_stretch(sharpen_image(image)))
                    )

            except (rasterio.errors.RasterioIOError, rasterio.errors.DatasetAttributeError, OSError) as e:
                LOG.error(
                    "Error processing image %s: %s", img_path, e)

            progress_bar.update(1)

        progress_bar.close()

    def run_georeferencing(
            self,
            imu_file: Optional[str] = None) -> None:
        """
        Run georeferencing process on all images.

        Args:
            imu_file: Optional path to IMU data file
        """
        self.process_raw_images(
            self.raw_image_list,
            self.output_path,
            self.output_epsg,
            self.binning,
            imu_file,
        )
