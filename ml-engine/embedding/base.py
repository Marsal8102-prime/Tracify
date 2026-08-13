"""
Embedding — Abstract base class for face embedding generators.

Defines the contract for any face embedding model (ArcFace, FaceNet, etc.).
Implementations accept an aligned face crop and return a normalized
embedding vector.

Usage:
    from embedding.base import BaseEmbedder
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for face embedding extraction.

    Implementations must provide:
      - load_model(): Initialize the embedding model (called once).
      - generate(aligned_face): Produce a normalized embedding vector.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model into memory. Called once at startup."""
        ...

    @abstractmethod
    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Generate a normalized embedding from an aligned face image.

        Args:
            aligned_face: Aligned BGR face image (H, W, 3), typically 112×112.

        Returns:
            L2-normalized embedding vector as np.ndarray of shape (dimension,).

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If the input image is invalid.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded and is ready."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vector."""
        ...
