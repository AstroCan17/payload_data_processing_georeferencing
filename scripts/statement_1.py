"""
This module contains the code for the first statement of the project.
"""

import gc
import glob
import logging
import os
import sys
from multiprocessing import Pool, cpu_count

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import psutil
from skimage import exposure
from tqdm import tqdm


def log_memory_usage():
    """Logs the memory usage of the current process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # Byte'dan MB'a çevir
    LOG_frameRegistration.info(
        f"Current memory usage: {
            memory_usage_mb:.2f} MB"
    )


def convert_to_8bit(img):
    img = img.astype("float64")
    return (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255).astype("uint8")


def match_histogram(img1, ref):
    return exposure.match_histograms(img1, ref)


def sharpen_image(img):
    img = img.astype("float64")
    sharpen_kernel = np.array([[0, -1.5, 0], [-1.5, 7, -1.5], [0, -1.5, 0]])
    img[np.where(img < 0)] = 0
    img[np.where(img > 2**16 - 1)] = 2**16 - 1
    return (cv2.filter2D(img, -1, sharpen_kernel)).astype("uint16")


def apply_stretch(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(
        img, in_range=(p2, p98), out_range=(0, 255)
    ).astype("uint8")


def apply_clahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(channel)


logging.basicConfig(level=logging.INFO, stream=sys.stdout)

LOG_bandSegmentation = logging.getLogger("Band Segmentation")  # Logger
LOG_bandSegmentation.setLevel(logging.INFO)


class bandSegmentation:
    def __init__(self, input_path, channels, output_path):
        self.dataset_path = input_path
        self.channels = channels
        self.output_path = os.path.join(output_path, "01_Statement_1")
        os.makedirs(self.output_path, exist_ok=True)

        self.output_path_band_seg = os.path.join(
            self.output_path, "01_band_segmentation"
        )
        os.makedirs(self.output_path_band_seg, exist_ok=True)

        self.raw_image_list = sorted(
            glob.glob(os.path.join(input_path, "*.tiff")))
        self.num_images = len(self.raw_image_list)

    def band_segmentation_generator(self):

        for img_path in self.raw_image_list:
            img_path = img_path.replace(os.sep, "/")
            img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_array = img_array[:, :]
            img_name = img_path.split("/")[-1]
            # LOG_bandSegmentation.info(f"Processing image: {img_name}")

            bands = {
                "red_edge_channels": img_array[:113, :],
                "nir_channels": img_array[114:221, :],
                "blue_channels": img_array[222:329, :],
                "green_channels": img_array[330:437, :],
                "red_channels": img_array[438:, :],
            }

            yield img_name, bands

    def run_statement_1_1(self):
        progress_bar = tqdm(
            total=self.num_images,
            desc=f"Band Segmentation Processing...\n{'-' * 50}",
            unit="image",
        )

        for img_name, bands in self.band_segmentation_generator():
            for channel, band_img in bands.items():
                output_dir = os.path.join(self.output_path_band_seg, channel).replace(
                    os.sep, "/"
                )
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(
                    output_dir, f"{channel.split('_')[0]}_{img_name}"
                )
                cv2.imwrite(output_file, band_img)
                # Free memory
                del band_img

            progress_bar.update(1)
        progress_bar.close()

        LOG_bandSegmentation.info(f"Band segmentation completed")
        for channel in self.channels:
            LOG_bandSegmentation.info(
                f"{channel} saved to {
                    os.path.join(
                        self.output_path_band_seg,
                        channel).replace(
                        os.sep,
                        '/')}\n"
                + "-" * 50
            )


LOG_frameRegistration = logging.getLogger("frameRegistration")  # Logger
LOG_frameRegistration.setLevel(logging.INFO)


class frameRegistrationMosaic(bandSegmentation):
    def __init__(self, input_path, channels, output_paths):
        super().__init__(input_path, channels, output_paths)
        self.dataset_path = self.output_path_band_seg
        self.channels = channels
        self.output_path = os.path.join(output_paths, "01_Statement_1")
        os.makedirs(self.output_path, exist_ok=True)

        self.output_path_stitched = os.path.join(
            self.output_path, "02_frame_stitched")
        os.makedirs(self.output_path_stitched, exist_ok=True)

    def inter_frame_registration(self, channel):
        SCALE_FACTOR = 0.5  # Optimize feature detection resolution
        SIFT_CONTRAST_THRESHOLD = 0.2  # Increased threshold to reduce keypoints

        # Initialize features detector with optimized parameters
        # Initialize features detector with optimized parameters

        sift = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher()

        LOG_frameRegistration.info(
            "Method for feature detection: SIFT\n"
            + "Method for feature matching: FLANN"
        )

        LOG_frameRegistration.info("-" * 50 + f"\nProcessing {channel}")
        raw_image_list = sorted(
            glob.glob(os.path.join(self.dataset_path, channel, "*.tiff"))
        )
        num_images = len(raw_image_list)
        progress_bar = tqdm(
            total=num_images,
            desc=f"Processing {channel} | {log_memory_usage()}",
            unit="image",
            position=0,
            leave=True,
        )

        # Initialize with first image
        first_img_path = raw_image_list.pop(0)

        stitched_mosaic = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)

        LOG_frameRegistration.info(
            f"Initial mosaic size: {
                stitched_mosaic.shape}"
        )

        progress_bar.update(1)
        while raw_image_list:
            img_path = raw_image_list.pop(0)

            # Memory optimized image loading
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Feature detection on scaled images

            # Keypoint detection with reduced number of features
            kp1, des1 = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_stretch(stitched_mosaic)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )
            kp2, des2 = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_stretch(image)),
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
                del des1, des2

                # Scale keypoints to original coordinates
                src_pts = (
                    np.float32([kp2[m.trainIdx].pt for m in matches]
                               ).reshape(-1, 1, 2)
                    / SCALE_FACTOR
                )
                dst_pts = (
                    np.float32([kp1[m.queryIdx].pt for m in matches]
                               ).reshape(-1, 1, 2)
                    / SCALE_FACTOR
                )

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                del src_pts, dst_pts, matches, kp1, kp2

                # Memory efficient stitching

                stitched_mosaic = self.mosaic_generation(
                    stitched_mosaic, image, H)

                del H, _, image

            progress_bar.update(1)

        LOG_frameRegistration.info(
            f"{channel} stitching completed\n"
            + f"Final mosaic size: {stitched_mosaic.shape}\n"
            + f"Stitched mosaic saved to {
                os.path.join(
                    self.output_path,
                    f'{channel}_stitched.tiff')}\n"
            + "-" * 50
        )

        cv2.imwrite(
            os.path.join(self.output_path_stitched,
                         f"{channel}_stitched_16b.tiff"),
            stitched_mosaic,
        )
        cv2.imwrite(
            os.path.join(self.output_path_stitched,
                         f"{channel}_stitched_8b.tiff"),
            convert_to_8bit(apply_stretch(sharpen_image(stitched_mosaic))),
        )
        progress_bar.close()
        del stitched_mosaic
        gc.collect()

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
            translation_dist[1]: rows1 + translation_dist[1],
            translation_dist[0]: cols1 + translation_dist[0],
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
            img2 = img2[0: frame_size[0] - origin_r, :]

        if new_image[1] > frame_size[1] - origin_c:
            img2 = img2[:, 0: frame_size[1] - origin_c]

        output_img[
            origin_r: new_image[0] + origin_r, origin_c: new_image[1] + origin_c
        ] = img2

        return output_img

    def run_parallel_processing(self):

        with Pool(cpu_count()) as pool:
            pool.map(self.inter_frame_registration, self.channels)

        pool.close()
        pool.join()
        gc.collect()


LOG_trueColorComposite = logging.getLogger("trueColorComposite")  # Logger
LOG_trueColorComposite.setLevel(logging.INFO)


class trueColorComposite(frameRegistrationMosaic):
    def __init__(self, input_path, channels, output_path, reference_channel):
        super().__init__(input_path, channels, output_path)
        self.dataset_path = self.output_path_stitched
        self.channels = channels
        self.output_path_tcc = os.path.join(
            output_path, "01_Statement_1", "03_true_color_composite"
        )
        os.makedirs(self.output_path_tcc, exist_ok=True)
        self.reference_channel = reference_channel
        self.keypoints_method = "SIFT"
        self.matcher_method = "FLANN"

    def mask_image(self, img):
        stitched = cv2.copyMakeBorder(
            convert_to_8bit(img), 0, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
        )
        if len(stitched.shape) == 2:
            gray = stitched
        else:
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        # Apply GaussianBlur to reduce noise and improve contour detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            raise ValueError("No contours found in the image.")
        c = max(cnts, key=cv2.contourArea)
        # allocate memory for the mask which will contain the
        # rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        # create two copies of the mask: one to serve as our actual
        # minimum rectangular region and another to serve as a counter
        # for how many pixels need to be removed to form the minimum
        # rectangular region
        minRect = mask.copy()
        sub = mask.copy()
        # keep looping until there are no non-zero pixels left in the
        # subtracted image
        while cv2.countNonZero(sub) > 0:
            # erode the minimum rectangular mask and then subtract
            # the thresholded image from the minimum rectangular mask
            # so we can count if there are any non-zero pixels left
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

            cnts = cv2.findContours(
                minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = imutils.grab_contours(cnts)
            if not cnts:
                break
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            # use the bounding box coordinates to extract the our final
            # stitched image
            stitched = img[y: y + h, x: x + w]
        # plt.figure()
        # plt.subplot(121), plt.imshow(convert_to_8bit(img),cmap='gray'), plt.title('Original Image')
        # plt.subplot(122), plt.imshow(convert_to_8bit(stitched),cmap='gray'), plt.title('Masked Image')
        # plt.show()
        return stitched

    def true_color(self):
        progress_bar = tqdm(
            total=100, desc="True Color Composite Processing", unit="%")
        progress_bar.update(0)

        channel_paths = {
            "red_channels": os.path.join(
                self.dataset_path, "red_channels_stitched_16b.tiff"
            ).replace(os.sep, "/"),
            "green_channels": os.path.join(
                self.dataset_path, "green_channels_stitched_16b.tiff"
            ).replace(os.sep, "/"),
            "blue_channels": os.path.join(
                self.dataset_path, "blue_channels_stitched_16b.tiff"
            ).replace(os.sep, "/"),
        }

        red_channel = cv2.imread(
            channel_paths["red_channels"], cv2.IMREAD_UNCHANGED)
        green_channel = cv2.imread(
            channel_paths["green_channels"], cv2.IMREAD_UNCHANGED
        )
        blue_channel = cv2.imread(
            channel_paths["blue_channels"], cv2.IMREAD_UNCHANGED)

        # red_channel = red_channel[:, 41:2282]
        # green_channel = green_channel[:, 41:2282]
        # blue_channel = blue_channel[:, 41:2282]

        progress_bar.update(10)

        LOG_trueColorComposite.info(
            f"Shape of red channel: {red_channel.shape}\n"
            + f"Shape of green channel: {green_channel.shape}\n"
            + f"Shape of blue channel: {blue_channel.shape}"
        )

        reference_channel = cv2.imread(
            channel_paths[self.reference_channel], cv2.IMREAD_UNCHANGED
        )
        # reference_channel = self.mask_image(reference_channel)

        LOG_trueColorComposite.info(
            f"Reference channel: {self.reference_channel}\n"
            + f"Shape of reference channel: {reference_channel.shape}"
        )

        progress_bar.update(20)

        # Initialize features detector with optimized parameters
        if self.keypoints_method == "SIFT":
            sift = cv2.SIFT_create()
        else:
            sift = cv2.ORB_create(nfeatures=2000)

        if self.matcher_method == "FLANN":
            matcher = cv2.FlannBasedMatcher()

            def matcher_func(des1, des2, matcher):
                matches = matcher.knnMatch(des1, des2, k=2)
                # matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches)*0.45)]
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.45 * n.distance:
                        good_matches.append(m)
                return good_matches

        else:
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

            def matcher_func(des1, des2, matcher):
                matches = matcher.knnMatch(des1, des2, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.45 * n.distance:
                        good_matches.append(m)
                return good_matches

        SCALE_FACTOR = 1
        # Keypoint detection with reduced number of features
        with Pool(cpu_count()) as pool:
            kp_ref, des_ref = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_stretch(reference_channel)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )
            kp_blue, des_blue = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_stretch(blue_channel)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )
            kp_green, des_green = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_stretch(green_channel)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )
            kp_red, des_red = sift.detectAndCompute(
                cv2.resize(
                    convert_to_8bit(apply_stretch(red_channel)),
                    None,
                    fx=SCALE_FACTOR,
                    fy=SCALE_FACTOR,
                ),
                None,
            )
            progress_bar.update(40)
        pool.close()
        pool.join()

        # kp_ref, des_ref     = sift.detectAndCompute(cv2.resize(convert_to_8bit(apply_stretch(reference_channel)), None, fx=SCALE_FACTOR, fy=SCALE_FACTOR), None)
        # kp_blue, des_blue   = sift.detectAndCompute(cv2.resize(convert_to_8bit(apply_stretch(blue_channel)), None, fx=SCALE_FACTOR, fy=SCALE_FACTOR), None)
        # kp_green, des_green = sift.detectAndCompute(cv2.resize(convert_to_8bit(apply_stretch(green_channel)), None, fx=SCALE_FACTOR, fy=SCALE_FACTOR), None)
        # kp_red, des_red     = sift.detectAndCompute(cv2.resize(convert_to_8bit(apply_stretch(red_channel)), None, fx=SCALE_FACTOR, fy=SCALE_FACTOR), None)

        # progress_bar.update(40)

        # matches_blue = matcher.match(des_ref, des_blue)
        # matches_blue = sorted(matches_blue, key=lambda x: x.distance)[:int(len(matches_blue) * 0.5)]

        # matches_green = matcher.match(des_ref, des_green)
        # matches_green = sorted(matches_green, key=lambda x: x.distance)[:int(len(matches_green) * 0.5)]

        # matches_red = matcher.match(des_ref, des_red)
        # matches_red = sorted(matches_red, key=lambda x: x.distance)[:int(len(matches_red) * 0.5)]

        matches_blue = matcher_func(des_ref, des_blue, matcher)
        matches_green = matcher_func(des_ref, des_green, matcher)
        matches_red = matcher_func(des_ref, des_red, matcher)

        progress_bar.update(60)

        # Scale keypoints to original coordinates
        src_pts_blue = (
            np.float32([kp_blue[m.trainIdx].pt for m in matches_blue]
                       ).reshape(-1, 1, 2)
            / SCALE_FACTOR
        )
        dst_pts_blue = (
            np.float32([kp_ref[m.queryIdx].pt for m in matches_blue]
                       ).reshape(-1, 1, 2)
            / SCALE_FACTOR
        )

        src_pts_green = (
            np.float32([kp_green[m.trainIdx].pt for m in matches_green]).reshape(
                -1, 1, 2
            )
            / SCALE_FACTOR
        )
        dst_pts_green = (
            np.float32([kp_ref[m.queryIdx].pt for m in matches_green]
                       ).reshape(-1, 1, 2)
            / SCALE_FACTOR
        )

        src_pts_red = (
            np.float32([kp_red[m.trainIdx].pt for m in matches_red]
                       ).reshape(-1, 1, 2)
            / SCALE_FACTOR
        )
        dst_pts_red = (
            np.float32([kp_ref[m.queryIdx].pt for m in matches_red]
                       ).reshape(-1, 1, 2)
            / SCALE_FACTOR
        )

        H_blue, _ = cv2.findHomography(
            src_pts_blue, dst_pts_blue, cv2.RANSAC, 5.0)
        H_green, _ = cv2.findHomography(
            src_pts_green, dst_pts_green, cv2.RANSAC, 5.0)
        H_red, _ = cv2.findHomography(
            src_pts_red, dst_pts_red, cv2.RANSAC, 5.0)

        del (
            src_pts_blue,
            dst_pts_blue,
            src_pts_green,
            dst_pts_green,
            src_pts_red,
            dst_pts_red,
        )

        progress_bar.update(80)

        # Warp channels to reference channel
        blue_channel_warped = cv2.warpPerspective(
            blue_channel,
            H_blue,
            (reference_channel.shape[1], reference_channel.shape[0]),
        )
        green_channel_warped = cv2.warpPerspective(
            green_channel,
            H_green,
            (reference_channel.shape[1], reference_channel.shape[0]),
        )
        red_channel_warped = cv2.warpPerspective(
            red_channel, H_red, (reference_channel.shape[1],
                                 reference_channel.shape[0])
        )

        del blue_channel, green_channel, red_channel, H_blue, H_green, H_red
        # Find the overlapping area

        overlap_mask = (
            (blue_channel_warped > 0)
            & (green_channel_warped > 0)
            & (red_channel_warped > 0)
        )
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Crop the channels to the overlapping area
        blue_channel_cropped = blue_channel_warped[y_min: y_max +
                                                   1, x_min: x_max + 1]
        green_channel_cropped = green_channel_warped[
            y_min: y_max + 1, x_min: x_max + 1
        ]
        red_channel_cropped = red_channel_warped[y_min: y_max +
                                                 1, x_min: x_max + 1]

        height, width = blue_channel_cropped.shape
        rgb_frame = np.zeros((height, width, 3), dtype=np.uint16)

        rgb_frame[:, :, 0] = sharpen_image(red_channel_cropped)
        rgb_frame[:, :, 1] = sharpen_image(blue_channel_cropped)
        rgb_frame[:, :, 2] = sharpen_image(green_channel_cropped)

        rgb_frame_8b = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_frame_8b[:, :, 0] = convert_to_8bit(
            apply_stretch(rgb_frame[:, :, 0]))
        rgb_frame_8b[:, :, 1] = convert_to_8bit(
            apply_stretch(rgb_frame[:, :, 1]))
        rgb_frame_8b[:, :, 2] = convert_to_8bit(
            apply_stretch(rgb_frame[:, :, 2]))

        cv2.imwrite(
            os.path.join(self.output_path_tcc, "true_color_composite.tiff").replace(
                os.sep, "/"
            ),
            rgb_frame,
        )
        cv2.imwrite(
            os.path.join(self.output_path_tcc, "true_color_composite_8b.tiff").replace(
                os.sep, "/"
            ),
            rgb_frame_8b,
        )

        gc.collect()
        progress_bar.update(100)
        progress_bar.close()
