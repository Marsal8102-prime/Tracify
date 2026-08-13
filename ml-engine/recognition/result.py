"""
Recognition — Structured result types for face recognition.

Defines the output contract for all recognizer implementations:
  - RecognitionStatus enum: KNOWN / UNKNOWN
  - MatchCandidate: a single candidate match with score
  - RecognitionResult: the full recognition outcome

These types are backend-agnostic and used by all downstream consumers
(attendance, alerts, API responses).

Usage:
    from recognition.result import RecognitionResult, RecognitionStatus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class RecognitionStatus(Enum):
    """Explicit recognition outcome — never infer status from person_id."""

    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass
class MatchCandidate:
    """
    A single candidate match from the gallery.

    Attributes:
        person_id: The registered person's identifier.
        similarity: Cosine similarity score (−1.0 to 1.0 for normalized vectors).
        embedding_id: Identifier for the specific embedding that matched.
    """
    person_id: str
    similarity: float
    embedding_id: str


@dataclass
class RecognitionResult:
    """
    The structured output of a recognition query.

    Attributes:
        person_id: Matched person ID, or None if unknown.
        status: Explicit KNOWN or UNKNOWN status.
        similarity: Best similarity score (0.0 if no candidates).
        threshold: The threshold that was applied.
        matched_embedding_id: ID of the specific embedding that produced
            the best match, or None if unknown.
        candidates: Top-K candidate matches, ordered by descending similarity.
        timestamp: ISO-8601 timestamp of when the recognition was performed.
    """
    person_id: Optional[str]
    status: RecognitionStatus
    similarity: float
    threshold: float
    matched_embedding_id: Optional[str] = None
    candidates: List[MatchCandidate] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_known(self) -> bool:
        """Convenience check — True if status is KNOWN."""
        return self.status is RecognitionStatus.KNOWN

    @property
    def is_unknown(self) -> bool:
        """Convenience check — True if status is UNKNOWN."""
        return self.status is RecognitionStatus.UNKNOWN
