"""
This module contains functions for band segmentation operations.

Key functions:
- separate_bands: Separates a multi-band image into individual band files
- process_image_chunk: Processes a chunk of images for parallel processing
- process_images_parallel: Processes multiple images in parallel

The module provides utilities for:
- Band separation from multi-band images
- Parallel processing of band separation
- Band-specific file management

Dependencies:
- OpenCV (cv2) for image processing
- NumPy for array operations
- rasterio for geospatial operations
"""

import gc
import glob
import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("Band Segmentation Logger")  # Logger


class BandSegmentation:
    """Class for handling band segmentation operations on image data"""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        binning: int = 4,
    ):
        """
        Initialize band segmentation parameters

        Args:
            input_path: Path to input images
            output_path: Path for output files
            binning: Binning factor for pixel aggregation
        """
        self.input_path = input_path
        self.output_path = os.path.join(output_path, "02_Band_Segmentation")
        os.makedirs(self.output_path, exist_ok=True)
        self.raw_image_list = sorted(glob.glob(input_path + "/*.tiff"))
        self.binning = binning

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
        output_paths = []
        
        with rio.open(img_path) as src:
            img_array = src.read(1)
            transform = src.transform
            dtype = src.dtypes[0]
            crs = src.crs
            
            # Define band slices based on row ranges
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
                row_offset = start_row  # band's starting row index
                
                win = Window(
                    col_off=0,
                    row_off=row_offset,
                    width=ch_width * self.binning,
                    height=ch_height * self.binning,
                )
                band_transform = window_transform(win, transform)
                
                color = ch_name.split("_")[0]
                
                out_band_path = os.path.join(
                    output_path, ch_name, f"L1c_{color}_{os.path.basename(img_path)}"
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
                    crs=crs,
                    transform=band_transform,
                ) as dst_band:
                    dst_band.write(ch_data, 1)
                
                output_paths.append(out_band_path)
        
        return output_paths

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
        img_path, output_path, band_names = args
        return self.separate_bands(img_path, output_path, band_names)

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
        if n_processes is None:
            n_processes = cpu_count()
        
        # Prepare arguments for parallel processing
        args_list = [(img_path, output_path, band_names) for img_path in img_list]
        
        # Process images in parallel
        with Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_image_chunk, args_list),
                total=len(args_list),
                desc="Separating bands",
                unit="image"
            ))
        
        # Flatten results list
        all_output_paths = [path for sublist in results for path in sublist]
        
        return all_output_paths

    def run_band_segmentation(self) -> List[str]:
        """
        Run band segmentation on all input images.
        
        Returns:
            List of paths to all processed single-band images
        """
        band_names = [
            "red_edge_channels",
            "nir_channels",
            "blue_channels",
            "green_channels",
            "red_channels"
        ]
        
        return self.process_images_parallel(
            self.raw_image_list,
            self.output_path,
            band_names
        ) 