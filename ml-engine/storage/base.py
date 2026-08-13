"""
Storage — Abstract base class for embedding storage.

Defines a clean, backend-agnostic interface for persisting and
retrieving face embeddings. The abstraction allows swapping the
storage backend (local files → PostgreSQL + pgvector → Pinecone)
without changing the recognition engine.

Usage:
    from storage.base import EmbeddingStore, EmbeddingRecord
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingRecord:
    """
    A stored face embedding with associated metadata.

    Attributes:
        person_id: Unique identifier for the person (e.g., employee ID, UUID).
        embedding: The L2-normalized face embedding vector.
        metadata: Additional information (model version, capture date, etc.).
        created_at: ISO-8601 timestamp of when the record was created.
    """
    person_id: str
    embedding: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EmbeddingStore(ABC):
    """
    Abstract base class for embedding storage backends.

    Implementations must provide CRUD operations for embedding records.
    """

    @abstractmethod
    def save(self, record: EmbeddingRecord) -> None:
        """
        Save or update an embedding record.

        If a record for the same person_id already exists, it should
        be overwritten.

        Args:
            record: The EmbeddingRecord to persist.

        Raises:
            ValueError: If the record contains invalid data.
            IOError: If the storage backend fails.
        """
        ...

    @abstractmethod
    def get(self, person_id: str) -> Optional[EmbeddingRecord]:
        """
        Retrieve an embedding record by person ID.

        Args:
            person_id: The unique person identifier.

        Returns:
            The EmbeddingRecord if found, or None.
        """
        ...

    @abstractmethod
    def delete(self, person_id: str) -> bool:
        """
        Delete an embedding record by person ID.

        Args:
            person_id: The unique person identifier.

        Returns:
            True if the record was deleted, False if it didn't exist.
        """
        ...

    @abstractmethod
    def list_ids(self) -> List[str]:
        """
        List all stored person IDs.

        Returns:
            List of person_id strings.
        """
        ...

    @abstractmethod
    def get_all(self) -> List[EmbeddingRecord]:
        """
        Retrieve all stored embedding records.

        Returns:
            List of all EmbeddingRecords. May be expensive for large stores.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored embeddings."""
        ...
