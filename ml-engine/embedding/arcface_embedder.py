"""
Embedding — ArcFace face embedding generator via InsightFace.

Uses the recognition model from the InsightFace model pack (e.g., buffalo_l)
to extract 512-dimensional face embeddings from aligned face crops.

The model is loaded once via load_model() and reused for all subsequent
generate() calls. Embeddings are L2-normalized before returning.

Usage:
    from embedding.arcface_embedder import ArcFaceEmbedder
    from config import load_settings

    settings = load_settings()
    embedder = ArcFaceEmbedder(settings.embedding)
    embedder.load_model()
    vector = embedder.generate(aligned_face)  # shape (512,), L2-normalized
"""

from __future__ import annotations

import numpy as np

from config.settings import EmbeddingSettings
from embedding.base import BaseEmbedder
from utils.logger import get_logger
from utils.timing import timed

_logger = get_logger("tracify.embedding.arcface")


class ArcFaceEmbedder(BaseEmbedder):
    """
    Face embedding generator using InsightFace's ArcFace recognition model.

    The model is loaded once and reused. Embeddings are always
    L2-normalized before returning.
    """

    def __init__(self, config: EmbeddingSettings):
        """
        Args:
            config: EmbeddingSettings from the loaded configuration.
        """
        self._model_name = config.model_name
        self._dimension = config.dimension
        self._provider = config.provider.lower()
        self._rec_model = None  # InsightFace recognition model instance

        _logger.info(
            f"ArcFaceEmbedder configured: model={self._model_name}, "
            f"dimension={self._dimension}, provider={self._provider}"
        )

    def load_model(self) -> None:
        """
        Load the InsightFace recognition model.

        Uses the recognition module from the InsightFace model pack.
        The model is loaded once and reused for all generate() calls.
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace is not installed. Run: pip install insightface onnxruntime"
            )

        _logger.info(f"Loading ArcFace model from pack: {self._model_name}")

        # Load only the recognition module (not detection)
        app = FaceAnalysis(
            name=self._model_name,
            allowed_modules=["recognition"],
        )

        # ctx_id: 0 = GPU, -1 = CPU
        ctx_id = 0 if self._provider == "gpu" else -1
        app.prepare(ctx_id=ctx_id)

        # Extract the recognition model from the FaceAnalysis app
        if not app.models:
            raise RuntimeError(
                f"No recognition model found in pack '{self._model_name}'. "
                "Ensure the model pack includes a recognition model."
            )

        # The recognition model is the one with 'rec' task or the first available model
        self._rec_model = None
        for model in app.models.values():
            if hasattr(model, "get_feat") or hasattr(model, "get"):
                self._rec_model = model
                break

        if self._rec_model is None:
            raise RuntimeError(
                f"Could not find a recognition model with get_feat() "
                f"in pack '{self._model_name}'."
            )

        _logger.info(
            f"ArcFace model loaded successfully (provider={self._provider})"
        )

    @timed(name="embedding.generate")
    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Generate an L2-normalized embedding from an aligned face image.

        Args:
            aligned_face: Aligned BGR face image, typically (112, 112, 3).

        Returns:
            L2-normalized embedding vector, shape (dimension,).

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If the input image is invalid or produces wrong dimension.
        """
        if self._rec_model is None:
            raise RuntimeError(
                "Embedding model not loaded. Call load_model() first."
            )

        # Validate input
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Input face image is empty or None.")

        if aligned_face.ndim != 3 or aligned_face.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel image (H, W, 3), "
                f"got shape {aligned_face.shape}"
            )

        # Generate embedding using InsightFace's recognition model
        # The get_feat method expects a BGR image
        try:
            raw_embedding = self._rec_model.get_feat(aligned_face)
        except Exception as e:
            raise ValueError(
                f"Embedding inference failed: {e}"
            ) from e

        # InsightFace may return (1, dim) or (dim,) depending on the model version
        embedding = np.array(raw_embedding, dtype=np.float32).flatten()

        # Validate dimension
        if embedding.shape[0] != self._dimension:
            raise ValueError(
                f"Expected embedding dimension {self._dimension}, "
                f"got {embedding.shape[0]}. Check model configuration."
            )

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            _logger.warning("Embedding has near-zero norm, returning zero vector")
            return np.zeros(self._dimension, dtype=np.float32)

        embedding = embedding / norm

        _logger.debug(
            f"Embedding generated: dim={embedding.shape[0]}, "
            f"norm={np.linalg.norm(embedding):.4f}"
        )
        return embedding

    @property
    def is_loaded(self) -> bool:
        return self._rec_model is not None

    @property
    def dimension(self) -> int:
        return self._dimension
