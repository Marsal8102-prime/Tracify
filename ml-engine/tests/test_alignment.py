"""
Tests — Face alignment module.

Tests valid alignment, invalid/missing landmarks, correct output size,
and edge cases. All tests use synthetic data and do not require
a GPU or a pretrained model.
"""

import numpy as np
import pytest

from alignment.face_aligner import FaceAligner, _ARCFACE_REF_LANDMARKS
from alignment.base import BaseAligner
from config.settings import AlignmentSettings
from detection.base import DetectionResult


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def aligner() -> FaceAligner:
    """Create a FaceAligner with default 112×112 output."""
    config = AlignmentSettings(output_size=[112, 112], landmark_type="2d")
    return FaceAligner(config)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a synthetic 480×640 BGR image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def valid_detection() -> DetectionResult:
    """Create a DetectionResult with valid 5-point landmarks."""
    return DetectionResult(
        bbox=np.array([100.0, 80.0, 250.0, 280.0], dtype=np.float32),
        confidence=0.95,
        landmarks=np.array(
            [
                [140.0, 140.0],  # left eye
                [210.0, 140.0],  # right eye
                [175.0, 180.0],  # nose
                [145.0, 220.0],  # left mouth
                [205.0, 220.0],  # right mouth
            ],
            dtype=np.float32,
        ),
    )


# ── Test: valid alignment ───────────────────────────────────────────────

class TestFaceAlignerValidInput:
    """Tests for successful alignment with valid inputs."""

    def test_output_shape_112x112(self, aligner, sample_frame, valid_detection):
        """Aligned face should be exactly 112×112×3."""
        result = aligner.align(sample_frame, valid_detection)
        assert result is not None
        assert result.shape == (112, 112, 3)

    def test_output_dtype_uint8(self, aligner, sample_frame, valid_detection):
        """Output should preserve uint8 dtype."""
        result = aligner.align(sample_frame, valid_detection)
        assert result is not None
        assert result.dtype == np.uint8

    def test_output_not_all_zeros(self, aligner, sample_frame, valid_detection):
        """Result should contain actual image data, not be blank."""
        result = aligner.align(sample_frame, valid_detection)
        assert result is not None
        assert result.sum() > 0

    def test_custom_output_size(self, sample_frame, valid_detection):
        """Aligner should respect a custom output size."""
        config = AlignmentSettings(output_size=[224, 224], landmark_type="2d")
        custom_aligner = FaceAligner(config)
        result = custom_aligner.align(sample_frame, valid_detection)
        assert result is not None
        assert result.shape == (224, 224, 3)

    def test_output_size_property(self, aligner):
        """output_size property should return configured size."""
        assert aligner.output_size == (112, 112)

    def test_is_subclass_of_base_aligner(self):
        """FaceAligner must implement BaseAligner."""
        assert issubclass(FaceAligner, BaseAligner)


# ── Test: invalid / missing landmarks ───────────────────────────────────

class TestFaceAlignerInvalidInput:
    """Tests for graceful handling of invalid inputs."""

    def test_none_landmarks(self, aligner, sample_frame):
        """Should return None when landmarks are None."""
        detection = DetectionResult(
            bbox=np.array([100, 80, 250, 280], dtype=np.float32),
            confidence=0.9,
            landmarks=None,
        )
        result = aligner.align(sample_frame, detection)
        assert result is None

    def test_wrong_landmark_shape(self, aligner, sample_frame):
        """Should return None for non-(5,2) landmarks."""
        detection = DetectionResult(
            bbox=np.array([100, 80, 250, 280], dtype=np.float32),
            confidence=0.9,
            landmarks=np.array([[1, 2], [3, 4]], dtype=np.float32),  # (2, 2)
        )
        result = aligner.align(sample_frame, detection)
        assert result is None

    def test_nan_landmarks(self, aligner, sample_frame):
        """Should return None when landmarks contain NaN."""
        landmarks = np.array(
            [[float("nan"), 140], [210, 140], [175, 180], [145, 220], [205, 220]],
            dtype=np.float32,
        )
        detection = DetectionResult(
            bbox=np.array([100, 80, 250, 280], dtype=np.float32),
            confidence=0.9,
            landmarks=landmarks,
        )
        result = aligner.align(sample_frame, detection)
        assert result is None

    def test_inf_landmarks(self, aligner, sample_frame):
        """Should return None when landmarks contain Inf."""
        landmarks = np.array(
            [[float("inf"), 140], [210, 140], [175, 180], [145, 220], [205, 220]],
            dtype=np.float32,
        )
        detection = DetectionResult(
            bbox=np.array([100, 80, 250, 280], dtype=np.float32),
            confidence=0.9,
            landmarks=landmarks,
        )
        result = aligner.align(sample_frame, detection)
        assert result is None


# ── Test: reference landmarks ───────────────────────────────────────────

class TestReferenceLandmarks:
    """Tests for the canonical ArcFace reference template."""

    def test_reference_landmarks_shape(self):
        """Reference landmarks must be (5, 2)."""
        assert _ARCFACE_REF_LANDMARKS.shape == (5, 2)

    def test_reference_landmarks_dtype(self):
        """Reference landmarks must be float32."""
        assert _ARCFACE_REF_LANDMARKS.dtype == np.float32

    def test_reference_landmarks_within_112(self):
        """All reference landmark coords must be within [0, 112]."""
        assert np.all(_ARCFACE_REF_LANDMARKS >= 0)
        assert np.all(_ARCFACE_REF_LANDMARKS <= 112)
