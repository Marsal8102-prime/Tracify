"""
Alignment — Abstract base class for face aligners.

Defines the contract that any face alignment implementation must follow.
Alignment transforms a detected face region into a standardized, pose-
normalized crop suitable for embedding extraction.

Usage:
    from alignment.base import BaseAligner
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from detection.base import DetectionResult


class BaseAligner(ABC):
    """
    Abstract base class for face alignment.

    Implementations take a raw frame and a DetectionResult (which includes
    5-point landmarks) and produce a normalized, aligned face crop.
    """

    @abstractmethod
    def align(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
    ) -> Optional[np.ndarray]:
        """
        Align and crop a single detected face from a frame.

        Args:
            frame: The original BGR image (H, W, 3).
            detection: A DetectionResult with landmarks from the detector.

        Returns:
            Aligned face crop as np.ndarray (output_h, output_w, 3) in BGR,
            or None if alignment cannot be performed (e.g., missing landmarks).
        """
        ...

    @property
    @abstractmethod
    def output_size(self) -> Tuple[int, int]:
        """Return the (width, height) of aligned face crops."""
        ...
