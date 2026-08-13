"""
Tests — Face embedding module.

Tests embedding generation, dimension validation, L2 normalization,
and error handling. Unit tests use a mock embedder that doesn't
require a GPU or model download. Integration tests with the real
InsightFace model are separated by a pytest marker.
"""

import numpy as np
import pytest

from embedding.base import BaseEmbedder
from embedding.arcface_embedder import ArcFaceEmbedder
from config.settings import EmbeddingSettings


# ── Mock Embedder for unit tests ────────────────────────────────────────

class MockEmbedder(BaseEmbedder):
    """
    Lightweight mock that simulates ArcFace without loading a real model.
    Used for unit tests that verify interface behavior.
    """

    def __init__(self, dimension: int = 512):
        self._dimension = dimension
        self._loaded = False

    def load_model(self) -> None:
        self._loaded = True

    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Empty input")
        if aligned_face.ndim != 3 or aligned_face.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3), got {aligned_face.shape}")

        # Generate a deterministic embedding from the image
        rng = np.random.RandomState(int(aligned_face.sum()) % (2**31))
        raw = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(raw)
        if norm < 1e-10:
            return np.zeros(self._dimension, dtype=np.float32)
        return raw / norm

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimension(self) -> int:
        return self._dimension


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def embedder() -> MockEmbedder:
    """Create and load a mock embedder."""
    e = MockEmbedder(dimension=512)
    e.load_model()
    return e


@pytest.fixture
def aligned_face() -> np.ndarray:
    """Create a synthetic 112×112 aligned face."""
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)


# ── Test: embedding generation ──────────────────────────────────────────

class TestEmbeddingGeneration:
    """Tests for correct embedding output."""

    def test_output_shape_512(self, embedder, aligned_face):
        """Embedding should have 512 dimensions."""
        result = embedder.generate(aligned_face)
        assert result.shape == (512,)

    def test_output_dtype_float32(self, embedder, aligned_face):
        """Embedding should be float32."""
        result = embedder.generate(aligned_face)
        assert result.dtype == np.float32

    def test_l2_normalized(self, embedder, aligned_face):
        """Embedding should be L2-normalized (norm ≈ 1.0)."""
        result = embedder.generate(aligned_face)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self, embedder, aligned_face):
        """Same input should produce same embedding."""
        result1 = embedder.generate(aligned_face)
        result2 = embedder.generate(aligned_face)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_different_input_different_output(self, embedder):
        """Different faces should produce different embeddings."""
        face1 = np.ones((112, 112, 3), dtype=np.uint8) * 100
        face2 = np.ones((112, 112, 3), dtype=np.uint8) * 200
        result1 = embedder.generate(face1)
        result2 = embedder.generate(face2)
        assert not np.allclose(result1, result2)


# ── Test: dimension validation ──────────────────────────────────────────

class TestEmbeddingDimension:
    """Tests for embedding dimension configuration."""

    def test_dimension_property(self, embedder):
        """dimension property should return configured value."""
        assert embedder.dimension == 512

    def test_custom_dimension(self):
        """MockEmbedder should respect custom dimension."""
        e = MockEmbedder(dimension=256)
        e.load_model()
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = e.generate(face)
        assert result.shape == (256,)


# ── Test: error handling ────────────────────────────────────────────────

class TestEmbeddingErrors:
    """Tests for proper error handling."""

    def test_model_not_loaded(self):
        """Should raise RuntimeError if model not loaded."""
        e = MockEmbedder()
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            e.generate(face)

    def test_empty_input(self, embedder):
        """Should raise ValueError for empty input."""
        with pytest.raises(ValueError):
            embedder.generate(np.array([]))

    def test_wrong_channels(self, embedder):
        """Should raise ValueError for non-3-channel input."""
        face = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
        with pytest.raises(ValueError):
            embedder.generate(face)

    def test_is_loaded_before_load(self):
        """is_loaded should be False before load_model()."""
        e = MockEmbedder()
        assert not e.is_loaded

    def test_is_loaded_after_load(self):
        """is_loaded should be True after load_model()."""
        e = MockEmbedder()
        e.load_model()
        assert e.is_loaded


# ── Test: ArcFaceEmbedder instantiation (no model download) ────────────

class TestArcFaceEmbedderConfig:
    """Tests for ArcFaceEmbedder configuration (no model required)."""

    def test_instantiation(self):
        """Should instantiate without loading model."""
        config = EmbeddingSettings(
            backend="arcface",
            model_name="buffalo_l",
            dimension=512,
            provider="cpu",
        )
        embedder = ArcFaceEmbedder(config)
        assert not embedder.is_loaded
        assert embedder.dimension == 512

    def test_generate_before_load(self):
        """Should raise RuntimeError if generate called before load."""
        config = EmbeddingSettings()
        embedder = ArcFaceEmbedder(config)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.generate(face)

    def test_is_subclass_of_base(self):
        """ArcFaceEmbedder must implement BaseEmbedder."""
        assert issubclass(ArcFaceEmbedder, BaseEmbedder)
