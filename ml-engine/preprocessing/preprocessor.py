"""
Preprocessing — Frame standardization before face detection.

Ensures consistent input to the detector regardless of source camera
resolution. Handles resizing (aspect-preserving), color normalization,
and optional histogram equalization for low-light CCTV feeds.

Key design: stores the scale factor so detection bounding boxes can be
mapped back to original frame coordinates downstream.

Usage:
    from preprocessing import FramePreprocessor
    from config import load_settings

    settings = load_settings()
    preprocessor = FramePreprocessor(settings.preprocessing)
    processed, meta = preprocessor.process(raw_frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from config.settings import PreprocessingSettings
from utils.logger import get_logger
from utils.timing import timed

_logger = get_logger("tracify.preprocessing")


@dataclass
class PreprocessedFrame:
    """
    Container for a preprocessed frame and its metadata.

    Attributes:
        frame: The preprocessed BGR image (resized, possibly equalized).
        original_shape: (H, W) of the original frame before preprocessing.
        scale_factor: Multiplier used during resize. To map detection
                      coordinates back to original: coord / scale_factor.
    """
    frame: np.ndarray
    original_shape: Tuple[int, int]
    scale_factor: float


class FramePreprocessor:
    """
    Standardizes raw camera frames before face detection.

    Pipeline:
      1. (Optional) Histogram equalization for low-light enhancement
      2. Resize to max_dimension while preserving aspect ratio
    """

    def __init__(self, config: PreprocessingSettings):
        """
        Args:
            config: PreprocessingSettings from the loaded configuration.
        """
        self._max_dim = config.max_dimension
        self._equalize = config.equalize_histogram
        _logger.info(
            f"Preprocessor initialized: max_dim={self._max_dim}, "
            f"equalize={self._equalize}"
        )

    @timed(name="preprocessing.process")
    def process(self, frame: np.ndarray) -> PreprocessedFrame:
        """
        Preprocess a raw camera frame.

        Args:
            frame: Raw BGR image from camera (H, W, 3).

        Returns:
            PreprocessedFrame with the processed image and metadata.
        """
        original_h, original_w = frame.shape[:2]
        result = frame

        # Step 1: Histogram equalization (optional, for low-light)
        if self._equalize:
            result = self._apply_clahe(result)

        # Step 2: Resize to max dimension
        result, scale = self._resize(result)

        return PreprocessedFrame(
            frame=result,
            original_shape=(original_h, original_w),
            scale_factor=scale,
        )

    def _resize(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize keeping aspect ratio so longest edge <= max_dimension."""
        h, w = frame.shape[:2]
        longest = max(h, w)

        if longest <= self._max_dim:
            return frame, 1.0

        scale = self._max_dim / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    @staticmethod
    def _apply_clahe(frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the luminance channel to enhance low-light images
        without distorting colors.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_ch)
        enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
