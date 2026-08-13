"""
Recognition — Embedding-based face recognizer with in-memory gallery.

Implements the full matching pipeline:
    1. Load known embeddings from storage into memory (gallery)
    2. Validate incoming query embedding
    3. Compute cosine similarity against all gallery entries (vectorized)
    4. Rank candidates by similarity (descending)
    5. Group by person_id and select the best match per person
    6. Apply configurable similarity threshold
    7. Return structured RecognitionResult (KNOWN or UNKNOWN)

The gallery is cached in memory as a contiguous NumPy matrix for
CCTV-speed batch comparison. Use refresh_gallery() to reload after
new registrations.

Usage:
    from recognition import EmbeddingRecognizer
    from storage import LocalEmbeddingStore
    from config import load_settings

    settings = load_settings()
    store = LocalEmbeddingStore(
        storage_dir=settings.storage.embeddings_dir,
        expected_dimension=settings.embedding.dimension,
    )
    recognizer = EmbeddingRecognizer(
        store=store,
        config=settings.recognition,
        expected_dimension=settings.embedding.dimension,
    )
    recognizer.load_gallery()
    result = recognizer.recognize(query_embedding)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import RecognitionSettings
from recognition.base import BaseRecognizer
from recognition.result import MatchCandidate, RecognitionResult, RecognitionStatus
from recognition.similarity import (
    cosine_similarity_batch,
    validate_embedding,
)
from storage.base import EmbeddingStore
from utils.logger import get_logger
from utils.timing import timed

_logger = get_logger("tracify.recognition.engine")


class EmbeddingRecognizer(BaseRecognizer):
    """
    Face recognizer using vectorized cosine similarity against an
    in-memory gallery of known embeddings.

    Supports:
      - Multiple embeddings per person (best match wins)
      - Configurable similarity threshold and top-K
      - Gallery caching for CCTV-speed recognition
      - Graceful handling of empty/corrupted gallery entries
    """

    def __init__(
        self,
        store: EmbeddingStore,
        config: RecognitionSettings,
        expected_dimension: int = 512,
    ):
        """
        Args:
            store: EmbeddingStore backend for loading known embeddings.
            config: RecognitionSettings with threshold, strategy, top_k.
            expected_dimension: Expected embedding dimensionality.
        """
        self._store = store
        self._threshold = config.similarity_threshold
        self._top_k = config.top_k
        self._expected_dim = expected_dimension

        # In-memory gallery (populated by load_gallery)
        self._gallery_matrix: Optional[np.ndarray] = None  # shape (N, dim)
        self._gallery_person_ids: List[str] = []
        self._gallery_embedding_ids: List[str] = []
        self._gallery_loaded: bool = False

        _logger.info(
            f"EmbeddingRecognizer initialized: threshold={self._threshold}, "
            f"top_k={self._top_k}, dimension={self._expected_dim}"
        )

    @timed(name="recognition.load_gallery")
    def load_gallery(self) -> int:
        """
        Load all known embeddings from storage into the in-memory gallery.

        Skips invalid or corrupted entries with a warning.

        Returns:
            Number of valid embeddings loaded.
        """
        records = self._store.get_all()

        embeddings: List[np.ndarray] = []
        person_ids: List[str] = []
        embedding_ids: List[str] = []

        # Track per-person embedding count for multi-embedding support
        person_counts: Dict[str, int] = {}

        for record in records:
            # Validate each embedding before including in gallery
            try:
                self._validate_gallery_embedding(record.embedding)
            except (TypeError, ValueError) as e:
                _logger.warning(
                    f"Skipping invalid embedding for '{record.person_id}': {e}"
                )
                continue

            # Generate a unique embedding_id for multi-embedding support
            count = person_counts.get(record.person_id, 0)
            embedding_id = (
                f"{record.person_id}:{count}" if count > 0
                else record.person_id
            )
            person_counts[record.person_id] = count + 1

            embeddings.append(record.embedding.astype(np.float32))
            person_ids.append(record.person_id)
            embedding_ids.append(embedding_id)

        if embeddings:
            self._gallery_matrix = np.stack(embeddings, axis=0)
        else:
            self._gallery_matrix = np.empty(
                (0, self._expected_dim), dtype=np.float32
            )

        self._gallery_person_ids = person_ids
        self._gallery_embedding_ids = embedding_ids
        self._gallery_loaded = True

        _logger.info(
            f"Gallery loaded: {len(embeddings)} embeddings, "
            f"{len(person_counts)} unique persons"
        )
        return len(embeddings)

    def refresh_gallery(self) -> int:
        """
        Reload the gallery from storage (cache invalidation).

        Returns:
            Number of embeddings loaded after refresh.
        """
        _logger.info("Refreshing recognition gallery from storage")
        return self.load_gallery()

    @timed(name="recognition.recognize")
    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        """
        Match a query embedding against the known gallery.

        Args:
            query_embedding: L2-normalized face embedding, shape (dim,).

        Returns:
            RecognitionResult with KNOWN/UNKNOWN status and candidates.

        Raises:
            TypeError: If the query is not a numpy array.
            ValueError: If the query has wrong dimension, contains NaN/Inf,
                or is a zero vector.
            RuntimeError: If the gallery has not been loaded.
        """
        if not self._gallery_loaded:
            raise RuntimeError(
                "Gallery not loaded. Call load_gallery() before recognize()."
            )

        # Validate query
        validate_embedding(
            query_embedding,
            self._expected_dim,
            label="query_embedding",
        )

        # Empty gallery → always UNKNOWN
        if self._gallery_matrix is None or self._gallery_matrix.shape[0] == 0:
            _logger.debug("Empty gallery — returning UNKNOWN")
            return RecognitionResult(
                person_id=None,
                status=RecognitionStatus.UNKNOWN,
                similarity=0.0,
                threshold=self._threshold,
            )

        # Vectorized cosine similarity against entire gallery
        query = query_embedding.astype(np.float32)
        scores = cosine_similarity_batch(query, self._gallery_matrix)

        # Build per-person best scores and select top-K
        candidates = self._build_candidates(scores)

        # Decision
        if candidates and candidates[0].similarity >= self._threshold:
            best = candidates[0]
            _logger.debug(
                f"KNOWN: person_id='{best.person_id}', "
                f"similarity={best.similarity:.4f}, "
                f"threshold={self._threshold}"
            )
            return RecognitionResult(
                person_id=best.person_id,
                status=RecognitionStatus.KNOWN,
                similarity=best.similarity,
                threshold=self._threshold,
                matched_embedding_id=best.embedding_id,
                candidates=candidates,
            )
        else:
            best_score = candidates[0].similarity if candidates else 0.0
            _logger.debug(
                f"UNKNOWN: best_similarity={best_score:.4f}, "
                f"threshold={self._threshold}"
            )
            return RecognitionResult(
                person_id=None,
                status=RecognitionStatus.UNKNOWN,
                similarity=best_score,
                threshold=self._threshold,
                candidates=candidates,
            )

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
        return [self.recognize(q) for q in query_embeddings]

    @property
    def gallery_size(self) -> int:
        """Return the number of embeddings in the current gallery."""
        if self._gallery_matrix is None:
            return 0
        return self._gallery_matrix.shape[0]

    @property
    def threshold(self) -> float:
        """Return the current similarity threshold."""
        return self._threshold

    # ── Internal helpers ──────────────────────────────────────────────

    def _build_candidates(
        self,
        scores: np.ndarray,
    ) -> List[MatchCandidate]:
        """
        Build top-K candidate list from raw similarity scores.

        Groups by person_id and selects the best-scoring embedding
        per person to handle multiple embeddings per identity.

        Args:
            scores: Raw similarity scores, shape (N,).

        Returns:
            List of MatchCandidate ordered by descending similarity,
            limited to top_k entries.
        """
        # Find best score per person_id
        best_per_person: Dict[str, Tuple[float, str]] = {}

        for idx in range(len(scores)):
            person_id = self._gallery_person_ids[idx]
            score = float(scores[idx])
            embedding_id = self._gallery_embedding_ids[idx]

            if person_id not in best_per_person:
                best_per_person[person_id] = (score, embedding_id)
            elif score > best_per_person[person_id][0]:
                best_per_person[person_id] = (score, embedding_id)

        # Sort by similarity descending
        sorted_candidates = sorted(
            best_per_person.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )

        # Build MatchCandidate list, limited to top_k
        candidates = [
            MatchCandidate(
                person_id=person_id,
                similarity=score,
                embedding_id=emb_id,
            )
            for person_id, (score, emb_id) in sorted_candidates[:self._top_k]
        ]

        return candidates

    def _validate_gallery_embedding(self, embedding: np.ndarray) -> None:
        """Validate a single gallery embedding during load."""
        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                f"Embedding must be numpy ndarray, got {type(embedding).__name__}"
            )

        if embedding.ndim != 1:
            raise ValueError(
                f"Embedding must be 1-dimensional, got shape {embedding.shape}"
            )

        if embedding.shape[0] != self._expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._expected_dim}, "
                f"got {embedding.shape[0]}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding contains NaN or Inf values")
