"""
Registration package — face registration and identity management (Phase 4).

Orchestrates the registration of new identities by processing multiple
face samples through the ML pipeline, validating quality, checking for
duplicates, and storing embeddings.

Usage:
    from registration import RegistrationService, RegistrationResult, RegistrationStatus
    from registration.models import PersonIdentity
"""

from registration.result import (
    RegistrationResult,
    RegistrationStatus,
    SampleResult,
)
from registration.models import PersonIdentity
from registration.quality import FaceQualityValidator
from registration.duplicate import DuplicateChecker, DuplicateCheckResult
from registration.service import RegistrationService

__all__ = [
    "RegistrationService",
    "RegistrationResult",
    "RegistrationStatus",
    "SampleResult",
    "PersonIdentity",
    "FaceQualityValidator",
    "DuplicateChecker",
    "DuplicateCheckResult",
]
