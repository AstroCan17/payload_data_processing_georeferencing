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
from rasterio.crs import CRS  # type: ignore
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from skimage import exposure  # type: ignore
from skimage.exposure import rescale_intensity  # type: ignore
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("Georeferencing Logger")  # Logger


def convert_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Convert image to 8-bit with percentile-based contrast stretching

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
    Match histogram of img1 to reference image

    Args:
        img1: Source image to modify
        ref: Reference image to match

    Returns:
        Image with matched histogram
    """
    return exposure.match_histograms(img1, ref)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter to enhance image details

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
    Apply contrast stretching using percentile-based clipping

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
    Apply Contrast Limited Adaptive Histogram Equalization

    Args:
        channel: Input image channel

    Returns:
        CLAHE-enhanced image channel
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(channel.astype("uint8"))


class Georeferencing:
    """Class for handling georeferencing operations on image data"""

    def __init__(
        self,
        input_path: str,
        channels: List[str],
        output_path: str,
        apply_lever_offset: bool = False,
    ):
        """
        Initialize georeferencing parameters

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

    def img_generator(
        self,
    ) -> Generator[Tuple[str, np.ndarray, rio.crs.CRS], None, None]:
        """
        Generator function to yield image data from files

        Yields:
            Tuple containing:
                - Image name (str)
                - Image data (np.ndarray)
                - CRS information (rio.crs.CRS)
        """
        for img_path in self.raw_image_list:
            img_name = os.path.basename(str(img_path))
            with rio.open(img_path, "r") as f:
                img = f.read(1)
                crs = f.crs
                yield img_name, img, crs

    def read_imu_data(self) -> pd.DataFrame:
        """
        Read and parse IMU data from text file

        Returns:
            DataFrame containing parsed IMU data
        """
        # Fixed width column definitions
        col_specs = [
            (0, 12),  # Photo name
            (17, 28),  # Easting
            (28, 40),  # Northing
            (40, 49),  # Height
            (49, 57),  # Omega
            (57, 66),  # Phi
            (66, 74),  # Kappa
            (74, 83),  # Date format YY-MM-DD
            (83, 92),  # Time format HH:MM:SS
            (92, 105),  # Week Seconds
        ]

        columns = [
            "Photo name",
            "Easting",
            "Northing",
            "Height",
            "Omega",
            "Phi",
            "Kappa",
            "Date",
            "Time",
            "Week Seconds",
        ]

        ground_elevation = None
        local_coordinate_system = None
        selected_zone = None

        data = []
        with open(self.imu_path, "r") as f:
            # Find "Output of event data" section
            read_data = False
            for line in f:
                if "Used Ground elevation:" in line:
                    ground_elevation = float(line.split(":")[1].strip().split()[0])
                if "Local Coordinate System:" in line:
                    local_coordinate_system = f.readline().strip()
                    utm_zone = (
                        local_coordinate_system.split()[0]
                        + ""
                        + local_coordinate_system.split()[1]
                    ).strip()
                    if "North" in utm_zone:
                        o = "N"
                    else:
                        o = utm_zone.split()[1][0]
                if "Selected Zone:" in line:
                    selected_zone = int(line.split(":")[1].strip())
                    selected_zone = str(selected_zone)
                if "Used Ground elevation:" in line:
                    self.min_height = float(line.split(":")[1].split()[0])
                if "Output of event data" in line:
                    read_data = True
                    next(f)  # Skip header line
                    continue
                if read_data and line.startswith("image_"):
                    # Fixed width parsing
                    row = []
                    for start, end in col_specs:
                        row.append(line[start:end].strip())
                    data.append(row)

                    row.append(ground_elevation)
                    row.append(local_coordinate_system)
                    row.append(selected_zone)

        # Create DataFrame and convert types
        df = pd.DataFrame(
            data,
            columns=columns
            + ["Ground Elevation", "Local Coordinate System", "Selected Zone"],
        )
        numeric_cols = ["Easting", "Northing", "Height", "Omega", "Phi", "Kappa"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        df["Date"] = pd.to_datetime(df["Date"], format="%y-%m-%d").dt.date
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
        df["Week Seconds"] = pd.to_numeric(df["Week Seconds"], errors="coerce")
        df["Photo name"] = df["Photo name"].astype(str)
        df["Local Coordinate System"] = df["Local Coordinate System"].astype(str)
        df["Selected Zone"] = df["Selected Zone"].astype(str)

        return df

    def get_camera_params(self, img_name: str) -> Optional[Dict[str, Any]]:
        """
        Get camera parameters from IMU data for a given image.

        Args:
            img_name: Name of the image file without extension

        Returns:
            Dictionary containing camera parameters including position, orientation, and metadata.
            Returns None if the image is not found in IMU data.
        """
        img_name = img_name.split(".")[0]
        self.df_imu = self.read_imu_data()

        metadata = None
        for img_name_IMU in self.df_imu["Photo name"].values:
            if img_name == img_name_IMU:
                metadata = {
                    "Easting": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Easting"
                        ].values[0]
                    ),
                    "Northing": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Northing"
                        ].values[0]
                    ),
                    "Height": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Height"
                        ].values[0]
                    ),
                    "omega_X": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Omega"
                        ].values[0]
                    ),
                    "phi_Y": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Phi"
                        ].values[0]
                    ),
                    "kappa_Z": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Kappa"
                        ].values[0]
                    ),
                    "date": self.df_imu.loc[
                        self.df_imu["Photo name"] == img_name, "Date"
                    ].values[0],
                    "time": self.df_imu.loc[
                        self.df_imu["Photo name"] == img_name, "Time"
                    ].values[0],
                    "week_seconds": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Week Seconds"
                        ].values[0]
                    ),
                    "ground_elevation": float(
                        self.df_imu.loc[
                            self.df_imu["Photo name"] == img_name, "Ground Elevation"
                        ].values[0]
                    ),
                    "local_coordinate_system": self.df_imu.loc[
                        self.df_imu["Photo name"] == img_name, "Local Coordinate System"
                    ].values[0],
                    "Crs": f"+proj=utm +zone={self.df_imu.loc[self.df_imu['Photo name'] == img_name, 'Selected Zone'].values[0]} +ellps=WGS84 +units=m +no_defs",
                }
                gc.collect()
                return metadata
        return None

    def effective_gsd(self, flight_height: float) -> float:
        """
        Calculate effective Ground Sample Distance (GSD).

        Args:
            flight_height: Flight height above ground level in meters

        Returns:
            float: Calculated GSD value in meters/pixel
        """
        # Subtract ground elevation if flight_height is ellipsoidal height
        flight_height = flight_height - self.ground_elv
        # Basic GSD calculation: (Flight height * pixel size) / focal length
        return (flight_height * self.pix_height * self.binning) / self.focal_length

    def calc_rotation_matrix(
        self,
        omega_deg: float,
        phi_deg: float,
        kappa_deg: float,
        easting: float,
        northing: float,
        elv: float,
        img_height: int,
        img_width: int,
    ) -> Tuple[Affine, np.ndarray]:
        """
        Calculate rotation matrix and affine transform for image georeferencing.

        Args:
            omega_deg: Omega angle in degrees (rotation around X-axis)
            phi_deg: Phi angle in degrees (rotation around Y-axis)
            kappa_deg: Kappa angle in degrees (rotation around Z-axis)
            easting: Easting coordinate in UTM
            northing: Northing coordinate in UTM
            elv: Elevation in meters
            img_height: Image height in pixels
            img_width: Image width in pixels

        Returns:
            Tuple containing:
                - Affine transform matrix for 2D georeferencing
                - 3D rotation matrix (3x3 numpy array)
        """
        # Convert angles from degrees to radians
        phi = np.deg2rad(phi_deg)
        omega = np.deg2rad(omega_deg)
        kappa = np.deg2rad(kappa_deg)

        # R_x(phi)
        R_x = np.array(
            [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        )
        # R_y(omega)
        R_y = np.array(
            [
                [np.cos(omega), 0, np.sin(omega)],
                [0, 1, 0],
                [-np.sin(omega), 0, np.cos(omega)],
            ]
        )
        # R_z(kappa)
        R_z = np.array(
            [
                [np.cos(kappa), -np.sin(kappa), 0],
                [np.sin(kappa), np.cos(kappa), 0],
                [0, 0, 1],
            ]
        )
        # Multiplication order PAT-B: Rx * Ry * Rz
        R_3D = R_x @ R_y @ R_z

        # Apply lever offset if needed
        if self.apply_lever_offset:
            # Calculate net offset in body system
            total_lever = self.mount_center_lever + self.sensor_lever_arm
            # Transform to world (UTM) coordinates
            offset_world = R_3D @ total_lever
            # Update coordinates with offset
            easting += offset_world[0]
            northing += offset_world[1]
            elv += offset_world[2]

        # Calculate GSD
        GSD = self.effective_gsd(elv)

        # Image center pixels
        cx = img_width / 2.0
        cy = img_height / 2.0

        # Project camera y-axis to XY plane for yaw angle
        cam_y_world = R_3D[:, 1]
        yaw = np.arctan2(cam_y_world[1], cam_y_world[0])
        yaw_deg = np.rad2deg(yaw)

        # Apply to 2D rotation
        R_2D = Affine.rotation(yaw_deg)

        # Final translation (geographic placement)
        T_back = Affine.translation(easting, northing)
        # Move center pixel to origin
        T_to_center = Affine.translation(-cx, -cy)
        # Scale
        S_scale = Affine.scale(GSD, GSD)

        transform = T_back * R_2D * S_scale * T_to_center

        return transform, R_3D

    @staticmethod
    def get_latlon_bounds(geotiff_path: str) -> Tuple[float, float, float, float]:
        """
        Calculate the latitude-longitude bounds of a georeferenced TIFF.

        Args:
            geotiff_path: Path to the georeferenced TIFF file

        Returns:
            Tuple containing:
                - min_lat: Minimum latitude
                - max_lat: Maximum latitude
                - min_lon: Minimum longitude
                - max_lon: Maximum longitude
        """
        with rio.open(geotiff_path) as src:
            transform = src.transform  # Affine dönüşüm matrisi
            width = src.width
            height = src.height
            utm_crs = src.crs.to_string()  # Raster CRS'sini al (örneğin, "EPSG:32633")

            # UTM -> WGS84 (EPSG:4326) dönüşümü
            transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

            # Dört köşe noktasını hesapla
            corners_utm = [
                transform * (0, 0),  # Sol Üst (Min X, Max Y)
                transform * (width, 0),  # Sağ Üst (Max X, Max Y)
                transform * (0, height),  # Sol Alt (Min X, Min Y)
                transform * (width, height),  # Sağ Alt (Max X, Min Y)
            ]

            # UTM'den WGS84'e çevir
            corners_latlon = [transformer.transform(x, y) for x, y in corners_utm]

            # Bounding box için min-max hesapla
            min_lon = min(corner[0] for corner in corners_latlon)
            max_lon = max(corner[0] for corner in corners_latlon)
            min_lat = min(corner[1] for corner in corners_latlon)
            max_lat = max(corner[1] for corner in corners_latlon)

            return min_lat, max_lat, min_lon, max_lon

    def rescale_image(self, img_array: np.ndarray) -> np.ndarray:
        """Rescale image intensity to 8-bit range."""
        return rescale_intensity(
            img_array,
            # Convert tuple to string
            in_range=str((np.min(img_array), np.max(img_array))),
            out_range=str((0, 255)),
        )  # Convert tuple to string

    def run_georeferencing(
        self,
        raw_img_list: List[str],
        output_path: str,
        output_epsg: int = 4326,
        output_res: Optional[float] = None,
        output_dtype: str = "uint16",
        output_nodata: Optional[float] = None,
        output_compress: str = "DEFLATE",
        output_tiled: bool = True,
        output_bigtiff: bool = True,
        output_photometric: str = "RGB",
        output_predictor: int = 2,
        output_zlevel: int = 9,
        n_processes: Optional[int] = None,
    ) -> None:
        """
        Process and georeference a list of raw images.

        Args:
            raw_img_list: List of paths to raw images
            output_path: Base path for output files
            output_epsg: Output coordinate system EPSG code
            output_res: Output resolution in units of coordinate system
            output_dtype: Output data type (e.g., 'uint16')
            output_nodata: No data value for output raster
            output_compress: Compression method for output files
            output_tiled: Whether to create tiled output
            output_bigtiff: Whether to create BigTIFF format
            output_photometric: Photometric interpretation
            output_predictor: Predictor for compression
            output_zlevel: Compression level
            n_processes: Number of parallel processes to use

        Returns:
            None
        """
        num_images = len(raw_img_list)
        progress_bar = tqdm(
            total=num_images,
            desc=f"Georeferencing Raw Images",
            unit="image",
            position=0,
            leave=True,
        )

        for i, raw_image_path in enumerate(raw_img_list):
            with rio.open(raw_image_path) as src:
                img_array = src.read(1)
                dtype = src.dtypes[0]
                img_height = src.height
                img_width = src.width

            img_name = os.path.basename(raw_image_path)
            img_name = img_name.split(".")[0]
            # metadata: [easting, northing, elv, omega_deg, phi_deg, kappa_deg]
            metadata = self.get_camera_params(img_name)
            easting, northing, elv, omega_deg, phi_deg, kappa_deg = (
                metadata["Easting"],
                metadata["Northing"],
                metadata["Height"],
                metadata["omega_X"],
                metadata["phi_Y"],
                metadata["kappa_Z"],
            )

            transform, _ = self.calc_rotation_matrix(
                omega_deg,
                phi_deg,
                kappa_deg,
                easting,
                northing,
                elv,
                img_height,
                img_width,
            )

            transform = Affine.translation(-24, 5) * transform

            out_band_path = os.path.join(output_path, f"{img_name}_L1c.tiff")

            with rio.open(
                out_band_path,
                "w",
                driver="GTiff",
                height=img_height,
                width=img_width,
                count=1,
                dtype=dtype,
                crs=CRS.from_epsg(output_epsg),
                transform=transform,
            ) as dst:
                dst.write(img_array, 1)

            with rio.open(out_band_path) as src:
                img_array = src.read(1)
                _transform = src.transform

                dtype = src.dtypes[0]
                _crs = src.crs

            band_slices = {
                "red_edge_channels": (0, 113),
                "nir_channels": (114, 221),
                "blue_channels": (222, 329),
                "green_channels": (330, 437),
                "red_channels": (438, None),
            }

            for ch_name, (start_row, end_row) in band_slices.items():
                ch_data = img_array[start_row:end_row, :]
                ch_height, ch_width = ch_data.shape
                # row_offset = kesmeye başladığınız satır
                # Örneğin; sadece satır kesimi yapıyorsanız:
                row_offset = start_row  # band'ın başladığı satır indeksi

                win = Window(
                    col_off=0,
                    row_off=row_offset,
                    width=ch_width * self.binning,
                    height=ch_height * self.binning,
                )
                band_transform = window_transform(win, _transform)

                color = ch_name.split("_")[0]

                out_band_path = os.path.join(
                    output_path, ch_name, f"L1c_{color}_{img_name}.tiff"
                )
                os.makedirs(os.path.join(output_path, ch_name), exist_ok=True)
                with rio.open(
                    out_band_path,
                    "w",
                    driver="GTiff",
                    height=ch_height,
                    width=ch_width,
                    count=1,
                    dtype=dtype,
                    crs=_crs,
                    transform=band_transform,
                ) as dst_band:
                    dst_band.write(ch_data, 1)
            progress_bar.update(1)

        gc.collect()
        progress_bar.close()

    def separate_bands(
        self, img_path: str, output_path: str, band_names: List[str]
    ) -> List[str]:
        """
        Separate image bands into individual files.

        Args:
            img_path: Path to input multi-band image
            output_path: Base path for output files
            band_names: List of names for each band

        Returns:
            List of paths to created single-band images
        """
        # ... existing code ...

    def process_image_chunk(self, args: Tuple[str, str, List[str]]) -> List[str]:
        """
        Process a chunk of images for parallel processing.

        Args:
            args: Tuple containing:
                - input image path
                - output path
                - list of band names

        Returns:
            List of paths to processed single-band images
        """
        # ... existing code ...

    def process_images_parallel(
        self,
        img_list: List[str],
        output_path: str,
        band_names: List[str],
        n_processes: Optional[int] = None,
    ) -> List[str]:
        """
        Process multiple images in parallel.

        Args:
            img_list: List of input image paths
            output_path: Base path for output files
            band_names: List of band names
            n_processes: Number of parallel processes to use

        Returns:
            List of paths to all processed single-band images
        """
        # ... existing code ...


class mosaickRaster(Georeferencing):
    def __init__(self, input_path, channels, output_path):
        super().__init__(input_path, channels, output_path)
        self.channels = channels
        self.output_path = os.path.join(output_path, "02_Statement_2")
        os.makedirs(self.output_path, exist_ok=True)
        self.input_path = self.out_folder_seperated

        self.out_folder_mosaic = os.path.join(self.output_path, "03_Mosaic")
        self.out_folder_aligned = os.path.join(self.output_path, "04_Aligned")

        os.makedirs(self.out_folder_mosaic, exist_ok=True)
        os.makedirs(self.out_folder_aligned, exist_ok=True)

    def least_squares_shift(self, good_matches, kps1, kps2):
        """
        Eşleşen keypoint çiftleri için en küçük kareler yöntemiyle
        optimal translation (dx, dy) hesaplar.

        Args:
            good_matches: İyi eşleşmeler listesi (m)
            kps1: Birinci görüntünün keypoint'leri
            kps2: İkinci görüntünün keypoint'leri
            scale_factor: Eğer görüntüler resize edildiyse, orijinal ölçeğe döndürmek için (örneğin 2)

        Returns:
            dx, dy: Hesaplanan kayma miktarları
        """
        diffs = []
        for m in good_matches:
            pt1 = np.array(kps1[m.queryIdx].pt)
            pt2 = np.array(kps2[m.trainIdx].pt)
            diffs.append(pt2 - pt1)
        diffs = np.array(diffs)
        dx = diffs[:, 0].mean()
        dy = diffs[:, 1].mean()
        return dx, dy

    def feature_matching(self, img_ref, img_tar, dist, match_cond=bool):
        SCALE_FACTOR = 1  # Optimize feature detection resolution
        sift = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher()
        kp1, des1 = sift.detectAndCompute(
            cv2.resize(
                convert_to_8bit(apply_stretch(img_ref)),
                None,
                fx=SCALE_FACTOR,
                fy=SCALE_FACTOR,
            ),
            None,
        )
        kp2, des2 = sift.detectAndCompute(
            cv2.resize(
                convert_to_8bit(apply_stretch(img_tar)),
                None,
                fx=SCALE_FACTOR,
                fy=SCALE_FACTOR,
            ),
            None,
        )

        if des1 is not None and des2 is not None and len(des1) > 2 and len(des2) > 2:

            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[
                : int(len(matches) * dist)
            ]
            # Scale keypoints to original coordinates
            src_pts = (
                np.float32(
                    [kp2[m.trainIdx].pt for m in matches]
                    # *self.binning
                ).reshape(-1, 1, 2)
                / SCALE_FACTOR
            )
            dst_pts = (
                np.float32(
                    [kp1[m.queryIdx].pt for m in matches]
                    # *self.binning
                ).reshape(-1, 1, 2)
                / SCALE_FACTOR
            )

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            H = H / H[2, 2]
            dx = H[0, 2] / 4
            dy = H[1, 2] / 4
            shift_coord = (dx, dy)
            # shift_coord = (dx, dy)

            del src_pts, dst_pts, matches, kp1, des1, kp2, des2

            if match_cond:

                stitched_mosaic = self.mosaic_generation(img_ref, img_tar, H)

                gc.collect()
                return stitched_mosaic, shift_coord, H
            else:
                return _, shift_coord, H

    def inter_frame_registration(self, channel):
        SCALE_FACTOR = 0.5  # Optimize feature detection resolution

        # Initialize features detector with optimized parameters
        # Initialize features detector with optimized parameters

        sift = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher()

        raw_image_list = sorted(
            glob.glob(os.path.join(self.input_path, channel, "*.tiff"))
        )
        name_first = os.path.basename(raw_image_list[0])
        name_first = name_first.split("_")[1:]
        name_first = "_".join(name_first)
        metadata_first = self.get_camera_params(name_first)

        num_images = len(raw_image_list)
        progress_bar = tqdm(
            total=num_images,
            desc=f"Stitching and georeferencing {channel}",
            unit="image",
            position=0,
            leave=True,
        )

        # Initialize with first image
        first_img_path = raw_image_list.pop(0)

        with rio.open(first_img_path) as src:
            stitched_mosaic = src.read(1)

            # height_src, width_src = stitched_mosaic.shape
            # first_image = cv2.resize(first_image, (int(width/self.binning), int(height/self.binning)))
            # transform_src = src.transform
            profile_src = src.profile
            src_transform = src.transform

        progress_bar.update(1)
        while raw_image_list:
            img_path = raw_image_list.pop(0)
            # Memory optimized image loading
            with rio.open(img_path) as dst:
                image = dst.read(1)

            # Feature detection on scaled images
            # Keypoint detection with reduced number of features
            kp1, des1 = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_clahe(stitched_mosaic)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )
            kp2, des2 = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_clahe(image)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )

            if (
                des1 is not None
                and des2 is not None
                and len(des1) > 2
                and len(des2) > 2
            ):

                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)[
                    : int(len(matches) * 0.1)
                ]
                # Scale keypoints to original coordinates
                src_pts = (
                    np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    / SCALE_FACTOR
                )
                dst_pts = (
                    np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    / SCALE_FACTOR
                )

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                del src_pts, dst_pts, matches, kp1, des1, kp2, des2

                # Memory efficient stitching

                stitched_mosaic = self.mosaic_generation(stitched_mosaic, image, H)
                del H, _, image

            gc.collect()

            progress_bar.update(1)

        dst_height_mos, dst_width_mos = stitched_mosaic.shape

        raw_image_list = sorted(
            glob.glob(os.path.join(self.input_path, channel, "*.tiff"))
        )
        with rio.open(raw_image_list[0]) as src:

            first_img = src.read(1)
            first_height, first_width = first_img.shape
            transform = src.transform

        final_height, final_width = stitched_mosaic.shape

        row_diff = final_height - first_height

        del first_img, src

        # win = Window(col_off=-(final_center_pix[1]-first_center_pix[1]), row_off=-(final_center_pix[0]-first_center_pix[0]), width=final_width, height=final_height)
        # transform = window_transform(win, transform)

        win = Window(
            col_off=0, row_off=-row_diff, width=final_width * 4, height=first_height * 4
        )
        transform = window_transform(win, transform)

        # Artık "destination" düz bir GeoTIFF olarak kaydedilebilir
        with rio.open(
            os.path.join(self.out_folder_mosaic, f"L1c_{channel}_mosaic.tiff"),
            "w",
            driver="GTiff",
            height=dst_height_mos,
            width=dst_width_mos,
            count=1,
            dtype=stitched_mosaic.dtype,
            crs=profile_src["crs"],
            transform=transform,
        ) as dst:
            dst.write(stitched_mosaic, 1)

        cv2.imwrite(
            os.path.join(self.out_folder_mosaic, f"L1c_{channel}_mosaic_8b.png"),
            convert_to_8bit(apply_stretch(sharpen_image(stitched_mosaic))),
        )
        progress_bar.close()
        del stitched_mosaic

    def mosaic_generation(self, img1, img2, H):
        """
        function for warping/stitching two images using the homography matrix H

        Args:
            img1  : first image, or source image
            img2  : second image, that needs to be mapped to the frame of img1
            H     : homography matrix that maps img2 to img1

        Returns:
            output_img: img2 warped to img1 using H
        """

        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(
            -1, 1, 2
        )
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]
        ).reshape(-1, 1, 2)
        points_2 = cv2.perspectiveTransform(temp_points, H)
        points_concat = np.concatenate((points_1, points_2), axis=0)

        [x_min, y_min] = np.int32(points_concat.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(points_concat.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array(
            [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
        )

        output_img = cv2.warpPerspective(
            img2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
        )
        frame_size = output_img.shape
        new_image = img2.shape
        output_img[
            translation_dist[1] : rows1 + translation_dist[1],
            translation_dist[0] : cols1 + translation_dist[0],
        ] = img1

        origin_r = int(points_2[0][0][1])
        origin_c = int(points_2[0][0][0])

        # if the origin of projected image is out of bounds, then mapping to ()
        if origin_r < 0:
            origin_r = 0
        if origin_c < 0:
            origin_c = 0

        # Clipping the new image, if it's size is more than the frame
        if new_image[0] > frame_size[0] - origin_r:
            img2 = img2[0 : frame_size[0] - origin_r, :]

        if new_image[1] > frame_size[1] - origin_c:
            img2 = img2[:, 0 : frame_size[1] - origin_c]

        output_img[
            origin_r : new_image[0] + origin_r, origin_c : new_image[1] + origin_c
        ] = img2

        return output_img

    def run_parallel_processing(self):

        with Pool(cpu_count()) as pool:
            tqdm(
                pool.map(self.inter_frame_registration, self.channels),
                total=len(self.channels),
                desc="Mosaicking and Georeferencing",
                unit="channel",
            )

        pool.close()
        pool.join()
        gc.collect()


class tccGenerator(mosaickRaster):
    def __init__(self, input_path, channels, output_path, reference_channel_TCC):
        super().__init__(input_path, channels, output_path)
        self.input_path = self.out_folder_mosaic
        self.output_path = os.path.join(output_path, "02_Statement_2")
        os.makedirs(self.output_path, exist_ok=True)
        self.reference_channel_TCC = reference_channel_TCC
        self.out_folder_aligned = os.path.join(self.output_path, "04_Aligned")
        self.out_folder_TCC = os.path.join(self.output_path, "05_TCC")
        os.makedirs(self.out_folder_TCC, exist_ok=True)

    def align_georef(self):

        mosaicked_dir_list = []
        for channel in self.channels:
            input_dir = self.out_folder_mosaic
            img_mosaicked_path = os.path.join(input_dir, f"L1c_{channel}_mosaic.tiff")
            mosaicked_dir_list.append(img_mosaicked_path)
            if channel == "red_channels":
                red_path = img_mosaicked_path
            elif channel == "green_channels":
                green_path = img_mosaicked_path
            elif channel == "blue_channels":
                blue_path = img_mosaicked_path

        with rio.open(red_path) as src:
            red_img = src.read(1)
            red_transform = src.transform
            red_crs = src.crs

        red_8b = convert_to_8bit(red_img)

        with rio.open(green_path) as src:
            green_img = src.read(1)
            green_transform = src.transform
            green_crs = src.crs
        green_8b = convert_to_8bit(green_img)

        with rio.open(blue_path) as src:
            blue_img = src.read(1)
            blue_transform = src.transform
            blue_crs = src.crs
        blue_8b = convert_to_8bit(blue_img)

        _, shift_coord_r_g, _ = self.feature_matching(
            red_8b, green_8b, 0.45, match_cond=False
        )
        _, shift_coord_r_b, _ = self.feature_matching(
            red_8b, blue_8b, 0.45, match_cond=False
        )

        print(f"Shift for Red-Green: {shift_coord_r_g}")
        print(f"Shift for Red-Blue: {shift_coord_r_b}")

        # win_r_g = Window(col_off=-shift_coord_r_g[0]/2.5, row_off=-shift_coord_r_g[1]/2.5, width=green_8b.shape[1]*self.binning, height=green_8b.shape[0]*self.binning)
        # win_r_b = Window(col_off=-shift_coord_r_b[0]/2.5, row_off=-shift_coord_r_b[1]/2.5, width=blue_8b.shape[1]*self.binning ,  height=blue_8b.shape[0]*self.binning)
        win_r_g = Window(
            col_off=shift_coord_r_g[0] - 5,
            row_off=shift_coord_r_g[1] - 25,
            width=green_8b.shape[1] * self.binning,
            height=green_8b.shape[0] * self.binning,
        )
        win_r_b = Window(
            col_off=shift_coord_r_b[0] - 5,
            row_off=shift_coord_r_b[1] - 60,
            width=blue_8b.shape[1] * self.binning,
            height=blue_8b.shape[0] * self.binning,
        )

        green_transform = window_transform(win_r_g, green_transform)
        blue_transform = window_transform(win_r_b, blue_transform)

        green_transform = window_transform(win_r_g, green_transform)
        blue_transform = window_transform(win_r_b, blue_transform)

        with rio.open(
            os.path.join(self.out_folder_aligned, "green_shifted.tiff"),
            "w",
            driver="GTiff",
            height=green_img.shape[0],
            width=green_img.shape[1],
            count=1,
            dtype=green_img.dtype,
            crs=green_crs,
            transform=green_transform,
        ) as dst:
            dst.write(green_img, 1)

        with rio.open(
            os.path.join(self.out_folder_aligned, "blue_shifted.tiff"),
            "w",
            driver="GTiff",
            height=blue_img.shape[0],
            width=blue_img.shape[1],
            count=1,
            dtype=blue_img.dtype,
            crs=blue_crs,
            transform=blue_transform,
        ) as dst:
            dst.write(blue_img, 1)
        with rio.open(
            os.path.join(self.out_folder_aligned, "red_reference.tiff"),
            "w",
            driver="GTiff",
            height=red_img.shape[0],
            width=red_img.shape[1],
            count=1,
            dtype=red_img.dtype,
            crs=red_crs,
            transform=red_transform,
        ) as dst:
            dst.write(red_img, 1)
        gc.collect()

    def true_Color_Stack(self):
        progress_bar = tqdm(
            total=100, desc="True Color Composite Georeferencing", unit="%"
        )
        # img_list = sorted(glob.glob(os.path.join(self.out_folder_mosaic, "*.tiff")))

        progress_bar.update(0)
        for channel in self.channels:
            img_path = os.path.join(
                self.out_folder_mosaic, f"L1c_{channel}_mosaic.tiff"
            )
            with rio.open(img_path) as src:
                img = src.read(1)
                transform = src.transform
                crs = src.crs
                dtype = src.dtypes[0]
                height, width = img.shape
                profile = src.profile
            if channel == "red_channels":
                red_img = img
                red_transform = transform
                red_crs = crs
                red_profile = profile
            elif channel == "green_channels":
                green_img = img
                green_transform = transform
                green_crs = crs
                green_profile = profile
            elif channel == "blue_channels":
                blue_img = img
                blue_transform = transform
                blue_crs = crs
                blue_profile = profile
            else:
                pass

        # Referans kanalını belirleyelim:
        if self.reference_channel_TCC == "red_channels":
            ref_img = red_img
            ref_transform = red_transform
            ref_crs = red_crs
        elif self.reference_channel_TCC == "green_channels":
            ref_img = green_img
            ref_transform = green_transform
            ref_crs = green_crs
        elif self.reference_channel_TCC == "blue_channels":
            ref_img = blue_img
            ref_transform = blue_transform
            ref_crs = blue_crs
        else:
            pass

        _, _, H_1 = self.feature_matching(ref_img, red_img, 0.1, match_cond=False)
        _, _, H_2 = self.feature_matching(ref_img, green_img, 0.1, match_cond=False)
        _, _, H_3 = self.feature_matching(ref_img, blue_img, 0.1, match_cond=False)

        progress_bar.update(20)
        red_warped = cv2.warpPerspective(
            red_img, H_1, (ref_img.shape[1], ref_img.shape[0])
        )
        green_warped = cv2.warpPerspective(
            green_img, H_2, (ref_img.shape[1], ref_img.shape[0])
        )
        blue_warped = cv2.warpPerspective(
            blue_img, H_3, (ref_img.shape[1], ref_img.shape[0])
        )

        overlap_mask = (blue_warped > 0) & (green_warped > 0) & (red_warped > 0)
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Crop the channels to the overlapping area
        blue_channel_cropped = blue_warped[y_min : y_max + 1, x_min : x_max + 1]
        green_channel_cropped = green_warped[y_min : y_max + 1, x_min : x_max + 1]
        red_channel_cropped = red_warped[y_min : y_max + 1, x_min : x_max + 1]
        progress_bar.update(40)
        win = Window(
            col_off=-x_min + 25,
            row_off=-y_min - 60,
            width=ref_img.shape[1] * self.binning,
            height=ref_img.shape[0] * self.binning,
        )
        ref_transform = window_transform(win, ref_transform)

        color = self.reference_channel_TCC.split("_")[0]

        if color == "red":
            reference_channel = red_channel_cropped
        elif color == "green":
            reference_channel = green_channel_cropped
        elif color == "blue":
            reference_channel = blue_channel_cropped

        progress_bar.update(60)

        height, width = reference_channel.shape
        rgb_frame = np.zeros((height, width, 3), dtype=np.uint16)

        rgb_frame[:, :, 0] = sharpen_image(red_channel_cropped)
        rgb_frame[:, :, 1] = sharpen_image(blue_channel_cropped)
        rgb_frame[:, :, 2] = sharpen_image(green_channel_cropped)

        rgb_frame_8b = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_frame_8b[:, :, 0] = convert_to_8bit(apply_stretch(rgb_frame[:, :, 0]))
        rgb_frame_8b[:, :, 1] = convert_to_8bit(apply_stretch(rgb_frame[:, :, 1]))
        rgb_frame_8b[:, :, 2] = convert_to_8bit(apply_stretch(rgb_frame[:, :, 2]))
        progress_bar.update(80)

        with rio.open(
            os.path.join(self.out_folder_TCC, f"L1c_TCC_8b.tiff"),
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=rgb_frame_8b.dtype,
            crs=ref_crs,
            transform=ref_transform,
        ) as dst:
            dst.write(rgb_frame_8b[:, :, 0], 1)
            dst.write(rgb_frame_8b[:, :, 1], 2)
            dst.write(rgb_frame_8b[:, :, 2], 3)

        with rio.open(
            os.path.join(self.out_folder_TCC, f"L1c_TCC_16b.tiff"),
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=rgb_frame.dtype,
            crs=ref_crs,
            transform=ref_transform,
        ) as dst:
            dst.write(rgb_frame[:, :, 0], 1)
            dst.write(rgb_frame[:, :, 1], 2)
            dst.write(rgb_frame[:, :, 2], 3)

        cv2.imwrite(os.path.join(self.out_folder_TCC, f"L1c_TCC_8b.png"), rgb_frame_8b)
        cv2.imwrite(os.path.join(self.out_folder_TCC, f"L1c_TCC_16b.png"), rgb_frame)

        progress_bar.update(100)
        progress_bar.close()
        gc.collect()
