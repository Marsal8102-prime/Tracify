"""
Alignment — 5-point landmark affine face aligner.

Uses the standard ArcFace alignment approach: compute a similarity
transform from the detected 5-point facial landmarks to a canonical
reference template, then warp the face to a 112×112 crop.

The reference template is the one used by InsightFace/ArcFace, ensuring
that aligned crops are compatible with the pretrained recognition model.

Usage:
    from alignment.face_aligner import FaceAligner
    from config import load_settings

    settings = load_settings()
    aligner = FaceAligner(settings.alignment)
    aligned = aligner.align(frame, detection)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import AlignmentSettings
from detection.base import DetectionResult
from alignment.base import BaseAligner
from utils.logger import get_logger
from utils.timing import timed

_logger = get_logger("tracify.alignment")

# ── ArcFace canonical reference landmarks ───────────────────────────────
# These are the target positions for [left_eye, right_eye, nose,
# left_mouth_corner, right_mouth_corner] in a 112×112 image.
# Source: InsightFace alignment code (face_align.py).
_ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def _estimate_similarity_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """
    Estimate a 2×3 similarity transform matrix (rotation, scale, translation)
    from src landmarks to dst landmarks using least-squares.

    This is equivalent to skimage.transform.SimilarityTransform.estimate()
    but implemented with pure numpy to avoid an extra dependency.

    Args:
        src: Source landmarks, shape (N, 2).
        dst: Destination landmarks, shape (N, 2).

    Returns:
        2×3 affine transformation matrix.
    """
    num_points = src.shape[0]

    # Build the system of equations:
    #   [x' ]   [a -b tx] [x]
    #   [y' ] = [b  a ty] [y]
    #                      [1]
    # Rearranged into A @ params = b form.
    A = np.zeros((2 * num_points, 4), dtype=np.float64)
    b = np.zeros(2 * num_points, dtype=np.float64)

    for i in range(num_points):
        A[2 * i, 0] = src[i, 0]
        A[2 * i, 1] = -src[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 3] = 0.0
        b[2 * i] = dst[i, 0]

        A[2 * i + 1, 0] = src[i, 1]
        A[2 * i + 1, 1] = src[i, 0]
        A[2 * i + 1, 2] = 0.0
        A[2 * i + 1, 3] = 1.0
        b[2 * i + 1] = dst[i, 1]

    # Solve least-squares: params = [a, b, tx, ty]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, ty = params

    # Build 2×3 matrix
    M = np.array(
        [[a, -b_val, tx],
         [b_val, a, ty]],
        dtype=np.float64,
    )
    return M


class FaceAligner(BaseAligner):
    """
    Aligns detected faces using 5-point landmark similarity transform.

    Produces 112×112 (or configurable size) crops suitable for ArcFace
    embedding extraction.
    """

    def __init__(self, config: AlignmentSettings):
        """
        Args:
            config: AlignmentSettings from the loaded configuration.
        """
        self._output_w = config.output_size[0]
        self._output_h = config.output_size[1]

        # Scale the reference landmarks if output size differs from 112×112
        scale_x = self._output_w / 112.0
        scale_y = self._output_h / 112.0
        self._ref_landmarks = _ARCFACE_REF_LANDMARKS.copy()
        self._ref_landmarks[:, 0] *= scale_x
        self._ref_landmarks[:, 1] *= scale_y

        _logger.info(
            f"FaceAligner initialized: output_size=({self._output_w}x{self._output_h})"
        )

    @timed(name="alignment.align")
    def align(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
    ) -> Optional[np.ndarray]:
        """
        Align and crop a single detected face from a frame.

        Args:
            frame: The original BGR image (H, W, 3).
            detection: A DetectionResult with 5-point landmarks.

        Returns:
            Aligned face crop (output_h, output_w, 3) in BGR,
            or None if landmarks are missing or invalid.
        """
        # Validate landmarks
        if detection.landmarks is None:
            _logger.warning("Cannot align face: landmarks are None")
            return None

        landmarks = detection.landmarks
        if landmarks.shape != (5, 2):
            _logger.warning(
                f"Cannot align face: expected landmarks shape (5, 2), "
                f"got {landmarks.shape}"
            )
            return None

        # Check for NaN/Inf in landmarks
        if not np.all(np.isfinite(landmarks)):
            _logger.warning("Cannot align face: landmarks contain NaN or Inf")
            return None

        # Compute similarity transform
        try:
            M = _estimate_similarity_transform(
                src=landmarks.astype(np.float32),
                dst=self._ref_landmarks,
            )
        except np.linalg.LinAlgError:
            _logger.warning("Cannot align face: singular matrix in transform estimation")
            return None

        # Warp the frame to produce the aligned crop
        aligned = cv2.warpAffine(
            frame,
            M,
            (self._output_w, self._output_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Sanity check: ensure the result is valid
        if aligned is None or aligned.size == 0:
            _logger.warning("Alignment produced an empty image")
            return None

        _logger.debug(
            f"Face aligned: {aligned.shape[1]}x{aligned.shape[0]}, "
            f"confidence={detection.confidence:.3f}"
        )
        return aligned

    @property
    def output_size(self) -> Tuple[int, int]:
        return (self._output_w, self._output_h)
