"""
Detection — Abstract base class and data structures for face detection.

Defines the DetectionResult dataclass and the BaseDetector interface.
Any face detector (SCRFD, RetinaFace, YOLO-Face) must implement
BaseDetector, ensuring the rest of the pipeline is model-agnostic.

Usage:
    from detection.base import BaseDetector, DetectionResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    """
    A single detected face.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates.
        confidence: Detection confidence score (0.0 - 1.0).
        landmarks: 5-point facial landmarks as (5, 2) array:
                   [left_eye, right_eye, nose, left_mouth, right_mouth].
                   None if the detector doesn't provide landmarks.
    """
    bbox: np.ndarray           # shape (4,) → [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[np.ndarray] = None  # shape (5, 2) or None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)

    def scale_to_original(self, scale_factor: float) -> "DetectionResult":
        """
        Map coordinates from preprocessed frame back to original resolution.

        Args:
            scale_factor: The scale_factor from PreprocessedFrame.

        Returns:
            New DetectionResult with coordinates in original frame space.
        """
        if scale_factor == 1.0:
            return self

        scaled_bbox = self.bbox / scale_factor
        scaled_landmarks = None
        if self.landmarks is not None:
            scaled_landmarks = self.landmarks / scale_factor

        return DetectionResult(
            bbox=scaled_bbox,
            confidence=self.confidence,
            landmarks=scaled_landmarks,
        )


class BaseDetector(ABC):
    """
    Abstract base class for face detectors.

    Implementations must provide:
      - load_model(): Initialize the detection model.
      - detect(frame): Run detection and return results.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model into memory. Called once at startup."""
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of DetectionResult, one per detected face.
            Empty list if no faces found.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded and is ready."""
        ...
