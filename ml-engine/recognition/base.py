"""
Recognition — Abstract base class for face recognizers.

Defines the contract for any recognition implementation. The recognizer
accepts a normalized query embedding and returns a structured
RecognitionResult indicating known/unknown status with similarity scores.

The abstraction decouples recognition logic from the storage backend,
allowing the system to swap between local files, PostgreSQL + pgvector,
FAISS, Milvus, or any other vector store without changing the
recognition API.

Usage:
    from recognition.base import BaseRecognizer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from recognition.result import RecognitionResult


class BaseRecognizer(ABC):
    """
    Abstract base class for face recognition engines.

    Implementations must provide:
      - recognize(query_embedding): Match a single query against the gallery.
      - load_gallery(): Load known embeddings into memory.
      - gallery_size: Number of embeddings currently loaded.
    """

    @abstractmethod
    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        """
        Match a query embedding against the known gallery.

        Args:
            query_embedding: L2-normalized face embedding, shape (dim,).

        Returns:
            RecognitionResult with KNOWN/UNKNOWN status, similarity score,
            and top-K candidates.

        Raises:
            TypeError: If the query is not a numpy array.
            ValueError: If the query has wrong dimension, contains NaN/Inf,
                or is a zero vector.
        """
        ...

    @abstractmethod
    def recognize_batch(
        self,
        query_embeddings: List[np.ndarray],
    ) -> List[RecognitionResult]:
        """
        Match multiple query embeddings against the known gallery.

        Args:
            query_embeddings: List of L2-normalized face embeddings.

        Returns:
            List of RecognitionResult, one per query.
        """
        ...

    @abstractmethod
    def load_gallery(self) -> int:
        """
        Load known embeddings from storage into the in-memory gallery.

        Should be called once at startup. Use refresh_gallery() to reload.

        Returns:
            Number of embeddings loaded.
        """
        ...

    @abstractmethod
    def refresh_gallery(self) -> int:
        """
        Reload the gallery from storage (cache invalidation).

        Returns:
            Number of embeddings loaded after refresh.
        """
        ...

    @property
    @abstractmethod
    def gallery_size(self) -> int:
        """Return the number of embeddings in the current gallery."""
        ...
