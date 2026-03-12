"""
This module contains functions for inter-frame registration operations.

Key functions:
- feature_matching: Matches features between two images
- least_squares_shift: Calculates optimal translation using least squares
- inter_frame_registration: Performs registration between consecutive frames
- mosaic_generation: Generates mosaics by combining aligned frames

The module provides utilities for:
- Feature detection and matching
- Homography calculation
- Image warping and stitching
- Mosaic generation

Dependencies:
- OpenCV (cv2) for image processing
- NumPy for array operations
- rasterio for geospatial operations
"""

# pylint: disable=no-member,not-callable,unsubscriptable-object,no-name-in-module,c-extension-no-member
# type: ignore[attr-defined, index, union-attr, operator, misc, no-any-return]
# mypy: ignore-errors

import gc
import glob
import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
import rasterio as rio
from cv2.typing import MatLike  # type: ignore
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm

from src.utils.image import (apply_clahe, apply_stretch, convert_to_8bit,
                             sharpen_image)

# Type aliases
NDArray = npt.NDArray[np.float32]
KeyPoint = Any  # OpenCV KeyPoint type
DMatch = Any    # OpenCV DMatch type
Mat = NDArray  # same as npt.NDArray[np.float32]; avoid double subscript for numpy compatibility
IndexParams = Dict[str, Union[int, float, bool, str]]
SearchParams = Dict[str, Union[int, float, bool, str]]

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(
    "Inter-Frame Registration Logger")  # Logger


class InterFrameRegistration:
    """Class for handling inter-frame registration operations"""

    def __init__(
        self,
        input_path: str,
        channels: List[str],
        output_path: str,
        binning: int = 4,
    ):
        """
        Initialize inter-frame registration parameters

        Args:
            input_path: Path to input band-separated images
            channels: List of channels to process
            output_path: Path for output files
            binning: Binning factor for pixel aggregation
        """
        self.input_path = input_path
        self.channels = channels
        self.output_path = os.path.join(
            output_path, "03_Inter_Frame_Registration")
        os.makedirs(self.output_path, exist_ok=True)
        self.binning = binning

        self.out_folder_mosaic = os.path.join(
            self.output_path, "01_Mosaic")
        os.makedirs(self.out_folder_mosaic, exist_ok=True)

    def least_squares_shift(
        self,
        good_matches: List[DMatch],
        kps1: List[KeyPoint],
        kps2: List[KeyPoint]
    ) -> Tuple[float, float]:
        """
        Calculates optimal translation (dx, dy) using least squares method
        for matched keypoint pairs.

        Args:
            good_matches: List of good matches
            kps1: Keypoints from first image
            kps2: Keypoints from second image

        Returns:
            dx, dy: Calculated shift amounts
        """
        diffs = []
        for m in good_matches:
            pt1 = np.array(
                kps1[m.queryIdx].pt, dtype=np.float32)
            pt2 = np.array(
                kps2[m.trainIdx].pt, dtype=np.float32)
            diffs.append(pt2 - pt1)
        diffs = np.array(diffs, dtype=np.float32)
        dx = float(diffs[:, 0].mean())
        dy = float(diffs[:, 1].mean())
        return dx, dy

    def feature_matching(
            self,
            img1: NDArray,
            img2: NDArray) -> NDArray:
        """
        Perform feature matching between two images using SIFT.

        Args:
            img1: First image
            img2: Second image

        Returns:
            homography: Homography matrix
        """
        # Convert images to uint8 if needed
        img1_uint8 = img1.astype(
            np.uint8) if img1.dtype != np.uint8 else img1
        img2_uint8 = img2.astype(
            np.uint8) if img2.dtype != np.uint8 else img2

        # Create SIFT object
        sift = cv2.SIFT_create()  # type: ignore

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1_uint8, None)
        kp2, des2 = sift.detectAndCompute(img2_uint8, None)

        if des1 is None or des2 is None or len(
                kp1) < 2 or len(kp2) < 2:
            logging.warning(
                "Not enough features found for matching")
            return np.eye(3, dtype=np.float32)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = {
            "algorithm": FLANN_INDEX_KDTREE,
            "trees": 5}
        search_params = {"checks": 50}

        flann = cv2.FlannBasedMatcher(
            index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches: List[DMatch] = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            logging.warning("Not enough good matches found")
            return np.eye(3, dtype=np.float32)
        # Get matched keypoints
        src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

        # Find homography
        homography_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0)

        if homography_matrix is None:
            logging.warning(
                "Could not find homography matrix")
            return np.eye(3, dtype=np.float32)

        return cast(NDArray, homography_matrix)

    def apply_transform(self,
                        img: NDArray,
                        transform_matrix: NDArray,
                        output_shape: Optional[Tuple[int,
                                                     int]] = None) -> NDArray:
        """
        Apply transformation matrix to image.

        Args:
            img: Input image
            transform_matrix: Transformation matrix
            output_shape: Optional output shape (height, width)

        Returns:
            Transformed image
        """
        if output_shape is None:
            output_shape = img.shape[:2]

        # Calculate new image bounds
        h, w = output_shape
        origin_r = max(0, int(transform_matrix[1, 2]))
        origin_c = max(0, int(transform_matrix[0, 2]))

        # Apply perspective transform
        warped = cv2.warpPerspective(
            img,
            transform_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )

        return warped

    def inter_frame_registration(self, channel):
        """
        Performs inter-frame registration for a specific channel.

        Args:
            channel: Channel name to process
        """
        SCALE_FACTOR = 0.5  # Optimize feature detection resolution

        # Initialize features detector with optimized
        # parameters
        sift = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher()

        raw_image_list = sorted(
            glob.glob(
                os.path.join(
                    self.input_path,
                    channel,
                    "*.tiff")))

        if not raw_image_list:
            LOG.warning(
                f"No images found for channel {channel}")
            return

        name_first = os.path.basename(raw_image_list[0])
        name_first = name_first.split("_")[1:]
        name_first = "_".join(name_first)

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
            profile_src = src.profile
            src_transform = src.transform

        progress_bar.update(1)
        while raw_image_list:
            img_path = raw_image_list.pop(0)
            # Memory optimized image loading
            with rio.open(img_path) as dst:
                image = dst.read(1)

            # Feature detection on scaled images
            # Keypoint detection with reduced number of
            # features
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
                    : int(len(matches) * 0.1)]
                # Scale keypoints to original coordinates
                src_pts = (
                    np.float32(
                        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    / SCALE_FACTOR
                )
                dst_pts = (
                    np.float32(
                        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    / SCALE_FACTOR
                )

                H, _ = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0)
                del src_pts, dst_pts, matches, kp1, des1, kp2, des2

                # Memory efficient stitching
                stitched_mosaic = self.mosaic_generation(
                    stitched_mosaic, image, H)
                del H, _, image

            gc.collect()
            progress_bar.update(1)

        dst_height_mos, dst_width_mos = stitched_mosaic.shape

        raw_image_list = sorted(
            glob.glob(
                os.path.join(
                    self.input_path,
                    channel,
                    "*.tiff")))
        with rio.open(raw_image_list[0]) as src:
            first_img = src.read(1)
            first_height, first_width = first_img.shape
            transform = src.transform

        final_height, final_width = stitched_mosaic.shape
        row_diff = final_height - first_height

        del first_img, src

        win = Window(
            col_off=0,
            row_off=-row_diff,
            width=final_width * 4,
            height=first_height * 4)
        transform = window_transform(win, transform)

        # Save the mosaic as a GeoTIFF
        with rio.open(
            os.path.join(
                self.out_folder_mosaic,
                f"L1c_{channel}_mosaic.tiff"),
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
            os.path.join(
                self.out_folder_mosaic,
                f"L1c_{channel}_mosaic_8b.png"),
            convert_to_8bit(
                apply_stretch(
                    sharpen_image(stitched_mosaic))),
        )
        progress_bar.close()
        del stitched_mosaic

    def mosaic_generation(self, img1, img2, H):
        """
        Function for warping/stitching two images using the homography matrix H.

        Args:
            img1: First image, or source image
            img2: Second image, that needs to be mapped to the frame of img1
            H: Homography matrix that maps img2 to img1

        Returns:
            output_img: img2 warped to img1 using H
        """
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        points_1 = np.array([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        temp_points = np.array([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]], dtype=np.float32).reshape(-1, 1, 2)
        points_2 = cv2.perspectiveTransform(temp_points, H)
        points_concat = np.concatenate(
            (points_1, points_2), axis=0)

        [x_min, y_min] = np.int32(
            points_concat.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(
            points_concat.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [
            0, 1, translation_dist[1]], [0, 0, 1]])

        output_img = cv2.warpPerspective(
            img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
        frame_size = output_img.shape
        new_image = img2.shape
        output_img[translation_dist[1]: rows1 +
                   translation_dist[1], translation_dist[0]: cols1 +
                   translation_dist[0], ] = img1

        origin_r = int(points_2[0][0][1])
        origin_c = int(points_2[0][0][0])

        # if the origin of projected image is out of bounds, then
        # mapping to ()
        if origin_r < 0:
            origin_r = 0
        if origin_c < 0:
            origin_c = 0

        # Clipping the new image, if it's size is more than
        # the frame
        if new_image[0] > frame_size[0] - origin_r:
            img2 = img2[0: frame_size[0] - origin_r, :]

        if new_image[1] > frame_size[1] - origin_c:
            img2 = img2[:, 0: frame_size[1] - origin_c]

        output_img[origin_r: new_image[0] + origin_r,
                   origin_c: new_image[1] + origin_c] = img2

        return output_img

    def run_parallel_processing(self):
        """
        Run inter-frame registration in parallel for all channels.
        """
        with Pool(cpu_count()) as pool:
            tqdm(
                pool.map(
                    self.inter_frame_registration,
                    self.channels),
                total=len(
                    self.channels),
                desc="Mosaicking and Georeferencing",
                unit="channel",
            )

        pool.close()
        pool.join()
        gc.collect()
