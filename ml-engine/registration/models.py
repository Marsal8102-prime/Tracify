"""
Registration — Person identity model.

Defines the PersonIdentity dataclass that represents a registered person's
metadata. This is intentionally separate from raw face embeddings — a
person has one identity but potentially many embeddings.

Designed for future database migration: all fields are serializable
and the model does not depend on file paths or storage internals.

Usage:
    from registration.models import PersonIdentity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class PersonIdentity:
    """
    A registered person's identity and metadata.

    Attributes:
        person_id: Unique stable identifier (e.g., employee ID, UUID).
                   Must not change after registration.
        display_name: Human-readable name for UI/logs.
        registered_at: ISO-8601 timestamp of initial registration.
        status: Lifecycle status — "active" or "inactive".
        embedding_count: Number of face embeddings stored for this person.
        model_version: The embedding model used during registration.
        metadata: Arbitrary key-value metadata (department, role, etc.).
    """
    person_id: str
    display_name: str
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "active"
    embedding_count: int = 0
    model_version: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
