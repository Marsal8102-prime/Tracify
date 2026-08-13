"""
Registration — Face quality validation for registration samples.

Performs configurable quality checks on face images, detections,
aligned crops, and embeddings before accepting them for registration.

Returns structured rejection reasons rather than raising exceptions,
allowing the registration service to collect and report all issues.

Reuses the existing validate_embedding() from recognition.similarity
for embedding-level checks.

Usage:
    from registration.quality import FaceQualityValidator
    from config import load_settings

    settings = load_settings()
    validator = FaceQualityValidator(
        minimum_face_size=settings.registration.minimum_face_size,
        embedding_dimension=settings.embedding.dimension,
        enabled=settings.registration.quality_checks_enabled,
    )
    ok, reason = validator.validate_image(image)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from detection.base import DetectionResult
from utils.logger import get_logger

_logger = get_logger("tracify.registration.quality")


class FaceQualityValidator:
    """
    Validates face samples during registration.

    All check methods return (passed: bool, reason: str).
    When checks are disabled, all methods return (True, "checks disabled").
    """

    def __init__(
        self,
        minimum_face_size: int = 80,
        embedding_dimension: int = 512,
        enabled: bool = True,
    ):
        """
        Args:
            minimum_face_size: Minimum face bbox dimension (px).
            embedding_dimension: Expected embedding vector length.
            enabled: If False, all checks pass unconditionally.
        """
        self._min_face_size = minimum_face_size
        self._embedding_dim = embedding_dimension
        self._enabled = enabled

        _logger.info(
            f"FaceQualityValidator initialized: "
            f"min_face_size={self._min_face_size}, "
            f"embedding_dim={self._embedding_dim}, "
            f"enabled={self._enabled}"
        )

    # ── Image-level checks ──────────────────────────────────────────

    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """Check that the input image is a valid, non-empty BGR frame."""
        if not self._enabled:
            return True, "checks disabled"

        if image is None:
            return False, "image is None"

        if not isinstance(image, np.ndarray):
            return False, f"image is not a numpy array (got {type(image).__name__})"

        if image.size == 0:
            return False, "image is empty (zero pixels)"

        if image.ndim != 3 or image.shape[2] != 3:
            return False, f"image must be 3-channel BGR (got shape {image.shape})"

        return True, "valid image"

    # ── Detection-level checks ──────────────────────────────────────

    def validate_detection(
        self,
        detections: List[DetectionResult],
    ) -> Tuple[bool, str]:
        """
        Check that exactly one face was detected with valid landmarks
        and sufficient size.
        """
        if not self._enabled:
            return True, "checks disabled"

        if not detections:
            return False, "no faces detected in sample"

        if len(detections) > 1:
            return (
                False,
                f"multiple faces detected ({len(detections)}); "
                f"registration requires exactly one face",
            )

        face = detections[0]

        # Landmark checks
        if face.landmarks is None:
            return False, "face has no landmarks"

        if face.landmarks.shape != (5, 2):
            return (
                False,
                f"invalid landmark shape: expected (5, 2), "
                f"got {face.landmarks.shape}",
            )

        if not np.all(np.isfinite(face.landmarks)):
            return False, "landmarks contain NaN or Inf values"

        # Size check
        face_w = float(face.width)
        face_h = float(face.height)
        min_dim = min(face_w, face_h)
        if min_dim < self._min_face_size:
            return (
                False,
                f"face too small: {face_w:.0f}x{face_h:.0f}px "
                f"(minimum {self._min_face_size}px)",
            )

        return True, "valid detection"

    # ── Aligned-image checks ────────────────────────────────────────

    def validate_aligned(self, aligned: np.ndarray) -> Tuple[bool, str]:
        """Check that the aligned face crop is valid."""
        if not self._enabled:
            return True, "checks disabled"

        if aligned is None:
            return False, "alignment produced None (likely missing landmarks)"

        if not isinstance(aligned, np.ndarray):
            return False, "aligned face is not a numpy array"

        if aligned.size == 0:
            return False, "aligned face is empty"

        if aligned.ndim != 3 or aligned.shape[2] != 3:
            return (
                False,
                f"aligned face must be 3-channel (got shape {aligned.shape})",
            )

        return True, "valid aligned face"

    # ── Embedding-level checks ──────────────────────────────────────

    def validate_embedding(self, embedding: np.ndarray) -> Tuple[bool, str]:
        """
        Check that an embedding vector is valid for storage.

        Checks: type, shape, dimension, finiteness, non-zero.
        """
        if not self._enabled:
            return True, "checks disabled"

        if embedding is None:
            return False, "embedding is None"

        if not isinstance(embedding, np.ndarray):
            return (
                False,
                f"embedding is not a numpy array "
                f"(got {type(embedding).__name__})",
            )

        if embedding.ndim != 1:
            return (
                False,
                f"embedding must be 1-dimensional (got shape {embedding.shape})",
            )

        if embedding.shape[0] != self._embedding_dim:
            return (
                False,
                f"embedding dimension mismatch: "
                f"expected {self._embedding_dim}, got {embedding.shape[0]}",
            )

        if not np.all(np.isfinite(embedding)):
            return False, "embedding contains NaN or Inf values"

        if np.linalg.norm(embedding) < 1e-10:
            return False, "embedding is a zero vector"

        return True, "valid embedding"
