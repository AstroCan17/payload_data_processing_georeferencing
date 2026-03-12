"""
This module contains utility functions for image processing operations.

Key functions:
- convert_to_8bit: Converts image to 8-bit format
- apply_stretch: Applies contrast stretching to image
- apply_clahe: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
- sharpen_image: Applies sharpening filter to image

The module provides utilities for:
- Image format conversion
- Contrast enhancement
- Image sharpening
- Histogram equalization

Dependencies:
- OpenCV (cv2) for image processing
- NumPy for array operations
- scikit-image for exposure adjustments
"""

import cv2
import numpy as np
from skimage import exposure
from skimage.exposure import rescale_intensity


def convert_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Convert image to 8-bit with percentile-based contrast stretching

    Args:
        img: Input image array

    Returns:
        8-bit image array
    """
    # Get min and max values for percentile-based stretching
    p2, p98 = np.percentile(img, (2, 98))
    
    # Apply contrast stretching
    img_stretched = rescale_intensity(img, in_range=(p2, p98))
    
    # Convert to 8-bit
    img_8bit = (img_stretched * 255).astype(np.uint8)
    
    return img_8bit

def apply_stretch(img: np.ndarray) -> np.ndarray:
    """
    Apply contrast stretching to image

    Args:
        img: Input image array

    Returns:
        Contrast-stretched image array
    """
    # Get min and max values for percentile-based stretching
    p2, p98 = np.percentile(img, (2, 98))
    
    # Apply contrast stretching
    img_stretched = rescale_intensity(img, in_range=(p2, p98))
    
    return img_stretched

def apply_clahe(channel: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image channel

    Args:
        channel: Input image channel

    Returns:
        CLAHE-enhanced image channel
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(channel.astype("uint8"))

def match_histogram(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Match histogram of image to a reference image.

    Args:
        img: Input image array.
        ref: Reference image array.

    Returns:
        Image with histogram matched to ref.
    """
    return exposure.match_histograms(img, ref)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter to image

    Args:
        img: Input image array

    Returns:
        Sharpened image array
    """
    # Define sharpening kernel
    sharpen_kernel = np.array([[0, -1.5, 0], [-1.5, 7, -1.5], [0, -1.5, 0]])
    
    # Apply filter
    img = cv2.filter2D(src=img, ddepth=-1, kernel=sharpen_kernel)
    
    # Clip values to valid range
    img[img > 2**16 - 1] = 2**16 - 1
    img[img < 0] = 0
    img = img.astype("uint16")
    
    return img