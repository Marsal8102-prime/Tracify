"""
Registration — Duplicate identity detection.

Before registering a new person, checks whether their face embedding
already matches an existing identity in the gallery.

Reuses the Phase 3 recognition engine for similarity comparison,
but applies a separate (typically stricter) threshold to avoid
false-positive duplicates.

Design rationale:
    The recognition threshold (default 0.6) is tuned for real-time
    identification where recall matters. The duplicate threshold
    (default 0.7) is intentionally higher because during registration
    it's better to ask "is this already registered?" than to silently
    create a duplicate identity. Both thresholds must be calibrated
    with real validation data.

Usage:
    from registration.duplicate import DuplicateChecker
    checker = DuplicateChecker(recognizer)
    result = checker.check(embedding, threshold=0.7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from recognition.base import BaseRecognizer
from utils.logger import get_logger

_logger = get_logger("tracify.registration.duplicate")


@dataclass
class DuplicateCheckResult:
    """
    Result of a duplicate identity check.

    Attributes:
        is_duplicate: True if an existing identity exceeds the threshold.
        matched_person_id: The existing person ID, if a duplicate.
        similarity: Best similarity score found.
    """
    is_duplicate: bool
    matched_person_id: Optional[str] = None
    similarity: float = 0.0


class DuplicateChecker:
    """
    Checks whether a face embedding already belongs to a registered person.

    Delegates to the Phase 3 recognition engine for similarity computation,
    but applies its own configurable threshold.
    """

    def __init__(self, recognizer: BaseRecognizer):
        """
        Args:
            recognizer: A loaded BaseRecognizer (gallery must be loaded).
        """
        self._recognizer = recognizer

    def check(
        self,
        embedding: np.ndarray,
        threshold: float,
    ) -> DuplicateCheckResult:
        """
        Check whether an embedding matches any existing identity.

        Args:
            embedding: L2-normalized face embedding to check.
            threshold: Cosine similarity threshold for duplicate detection.

        Returns:
            DuplicateCheckResult indicating whether a duplicate was found.
        """
        # Empty gallery → no possible duplicates
        if self._recognizer.gallery_size == 0:
            _logger.debug("Empty gallery — no duplicate possible")
            return DuplicateCheckResult(is_duplicate=False)

        result = self._recognizer.recognize(embedding)

        if result.similarity >= threshold and result.person_id is not None:
            _logger.info(
                f"Potential duplicate detected: "
                f"matched_person='{result.person_id}', "
                f"similarity={result.similarity:.4f}, "
                f"threshold={threshold}"
            )
            return DuplicateCheckResult(
                is_duplicate=True,
                matched_person_id=result.person_id,
                similarity=result.similarity,
            )

        _logger.debug(
            f"No duplicate: best_similarity={result.similarity:.4f}, "
            f"threshold={threshold}"
        )
        return DuplicateCheckResult(
            is_duplicate=False,
            similarity=result.similarity,
        )
