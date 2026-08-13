"""
Registration — Structured result types for face registration.

Defines the output contract for the registration service:
  - RegistrationStatus enum: SUCCESS / REJECTED / DUPLICATE / INVALID
  - SampleResult: per-sample outcome with acceptance/rejection reason
  - RegistrationResult: the full registration outcome

These types are used by all downstream consumers (API, CLI, etc.).

Usage:
    from registration.result import RegistrationResult, RegistrationStatus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class RegistrationStatus(Enum):
    """Explicit registration outcome."""

    SUCCESS = "success"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    INVALID = "invalid"


@dataclass
class SampleResult:
    """
    Per-sample outcome during registration.

    Attributes:
        accepted: Whether this sample passed quality checks.
        reason: Human-readable explanation (acceptance or rejection).
        sample_index: Zero-based index of this sample in the batch.
    """
    accepted: bool
    reason: str
    sample_index: int


@dataclass
class RegistrationResult:
    """
    The structured output of a registration attempt.

    Attributes:
        person_id: The person ID that was being registered.
        status: Explicit SUCCESS / REJECTED / DUPLICATE / INVALID.
        accepted_count: Number of samples that passed quality checks.
        rejected_count: Number of samples that failed quality checks.
        rejection_reasons: Aggregated list of unique rejection reasons.
        sample_results: Per-sample detailed results.
        duplicate_person_id: If status is DUPLICATE, the existing person ID.
        duplicate_similarity: Cosine similarity with the duplicate candidate.
        timestamp: ISO-8601 timestamp of the registration attempt.
    """
    person_id: str
    status: RegistrationStatus
    accepted_count: int = 0
    rejected_count: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    sample_results: List[SampleResult] = field(default_factory=list)
    duplicate_person_id: Optional[str] = None
    duplicate_similarity: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_success(self) -> bool:
        """Convenience check — True if registration succeeded."""
        return self.status is RegistrationStatus.SUCCESS

    @property
    def is_duplicate(self) -> bool:
        """Convenience check — True if a duplicate was detected."""
        return self.status is RegistrationStatus.DUPLICATE
