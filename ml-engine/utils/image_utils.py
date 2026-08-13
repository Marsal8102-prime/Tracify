"""
Utils — Image I/O helpers.

Common image operations used across multiple modules:
loading, saving, resizing, and color space conversion.

All functions work with NumPy arrays in BGR format (OpenCV default).

Usage:
    from utils.image_utils import load_image, resize_image, bgr_to_rgb
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from utils.logger import get_logger

_logger = get_logger("tracify.image_utils")


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk as a BGR NumPy array.

    Args:
        path: Path to the image file.

    Returns:
        Image as np.ndarray in BGR format (H, W, C).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    _logger.debug(f"Loaded image: {path} ({image.shape[1]}x{image.shape[0]})")
    return image


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a BGR NumPy array to disk.

    Args:
        image: Image array in BGR format.
        path: Destination file path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save image: {path}")
    _logger.debug(f"Saved image: {path}")


def resize_image(
    image: np.ndarray,
    max_dimension: int,
) -> Tuple[np.ndarray, float]:
    """
    Resize an image so its longest edge <= max_dimension, preserving aspect ratio.

    Args:
        image: Input image (H, W, C).
        max_dimension: Maximum allowed size for the longest edge.

    Returns:
        Tuple of (resized_image, scale_factor).
        scale_factor can be used to map coordinates back to original size.
    """
    h, w = image.shape[:2]
    longest_edge = max(h, w)

    if longest_edge <= max_dimension:
        return image, 1.0

    scale = max_dimension / longest_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    _logger.debug(f"Resized: ({w}x{h}) → ({new_w}x{new_h}), scale={scale:.3f}")
    return resized, scale


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for low-light enhancement. Works on the luminance channel only
    to avoid color distortion.

    Args:
        image: Input BGR image.

    Returns:
        Enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def draw_detections(
    image: np.ndarray,
    bboxes: list,
    landmarks: list = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and optional landmarks on an image. Useful for
    visual debugging of the detection pipeline.

    Args:
        image: Input BGR image (will be copied, not modified in place).
        bboxes: List of [x1, y1, x2, y2] bounding boxes.
        landmarks: Optional list of (5, 2) landmark arrays.
        color: BGR color tuple for drawing.
        thickness: Line thickness.

    Returns:
        Copy of image with detections drawn.
    """
    canvas = image.copy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Draw landmarks if provided
        if landmarks is not None and i < len(landmarks):
            for point in landmarks[i]:
                px, py = int(point[0]), int(point[1])
                cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1)

    return canvas
