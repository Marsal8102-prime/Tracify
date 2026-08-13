"""
Storage — Local filesystem embedding storage backend.

Stores each embedding as a .npz file (numpy compressed archive)
containing the embedding vector and metadata. Files are named using
a sanitized version of the person_id.

File structure:
    storage/embeddings/
        <person_id>.npz
            - "embedding": np.ndarray shape (dim,)
            - "person_id": str
            - "metadata_keys": list of metadata key names
            - "metadata_values": list of metadata values
            - "created_at": str (ISO-8601)

This is the development/default backend. For production, swap to
a database-backed EmbeddingStore (e.g., PostgreSQL + pgvector).

Usage:
    from storage.local_store import LocalEmbeddingStore
    from config import load_settings

    settings = load_settings()
    store = LocalEmbeddingStore(
        storage_dir=settings.storage.embeddings_dir,
        expected_dimension=settings.embedding.dimension,
    )
    store.save(record)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from storage.base import EmbeddingRecord, EmbeddingStore
from utils.logger import get_logger

_logger = get_logger("tracify.storage.local")

# Safe filename pattern: allow alphanumeric, hyphens, underscores
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _sanitize_filename(person_id: str) -> str:
    """Convert person_id to a safe filename component."""
    sanitized = _SAFE_FILENAME_RE.sub("_", person_id)
    # Strip replacement underscores to detect inputs with no valid characters
    if not sanitized.strip("_"):
        raise ValueError(f"Person ID '{person_id}' produces an empty filename")
    return sanitized


class LocalEmbeddingStore(EmbeddingStore):
    """
    File-based embedding storage using NumPy .npz archives.

    Each person's embedding is stored as a separate .npz file in
    the configured storage directory.
    """

    def __init__(self, storage_dir: str, expected_dimension: int):
        """
        Args:
            storage_dir: Absolute path to the embeddings directory.
            expected_dimension: Expected embedding dimensionality for validation.
        """
        self._storage_dir = Path(storage_dir)
        self._expected_dim = expected_dimension

        # Ensure the directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(
            f"LocalEmbeddingStore initialized: dir={self._storage_dir}, "
            f"expected_dim={self._expected_dim}"
        )

    def save(self, record: EmbeddingRecord) -> None:
        """Save an embedding record as a .npz file."""
        # Validate
        self._validate_record(record)

        filepath = self._id_to_path(record.person_id)

        # Serialize metadata as parallel arrays (npz doesn't store dicts natively)
        meta_keys = list(record.metadata.keys()) if record.metadata else []
        meta_values = list(record.metadata.values()) if record.metadata else []

        try:
            np.savez_compressed(
                filepath,
                embedding=record.embedding.astype(np.float32),
                person_id=np.array(record.person_id),
                metadata_keys=np.array(meta_keys),
                metadata_values=np.array(meta_values),
                created_at=np.array(record.created_at),
            )
        except Exception as e:
            raise IOError(f"Failed to save embedding for '{record.person_id}': {e}") from e

        _logger.debug(f"Saved embedding: person_id='{record.person_id}', path={filepath}")

    def get(self, person_id: str) -> Optional[EmbeddingRecord]:
        """Load an embedding record from a .npz file."""
        filepath = self._id_to_path(person_id)

        if not filepath.exists():
            _logger.debug(f"Embedding not found: person_id='{person_id}'")
            return None

        try:
            return self._load_file(filepath)
        except Exception as e:
            _logger.error(
                f"Failed to load embedding for '{person_id}': {e}",
                exc_info=True,
            )
            return None

    def delete(self, person_id: str) -> bool:
        """Delete an embedding .npz file."""
        filepath = self._id_to_path(person_id)

        if not filepath.exists():
            _logger.debug(f"Cannot delete: person_id='{person_id}' not found")
            return False

        try:
            filepath.unlink()
            _logger.info(f"Deleted embedding: person_id='{person_id}'")
            return True
        except OSError as e:
            _logger.error(f"Failed to delete embedding for '{person_id}': {e}")
            return False

    def list_ids(self) -> List[str]:
        """List all person IDs with stored embeddings."""
        ids = []
        for filepath in self._storage_dir.glob("*.npz"):
            try:
                record = self._load_file(filepath)
                if record is not None:
                    ids.append(record.person_id)
            except Exception:
                _logger.warning(f"Skipping corrupted file: {filepath.name}")
        return sorted(ids)

    def get_all(self) -> List[EmbeddingRecord]:
        """Load all embedding records from storage."""
        records = []
        for filepath in self._storage_dir.glob("*.npz"):
            try:
                record = self._load_file(filepath)
                if record is not None:
                    records.append(record)
            except Exception:
                _logger.warning(f"Skipping corrupted file: {filepath.name}")
        return records

    def count(self) -> int:
        """Count the number of stored embeddings."""
        return len(list(self._storage_dir.glob("*.npz")))

    # ── Internal helpers ──

    def _id_to_path(self, person_id: str) -> Path:
        """Convert person_id to a .npz file path."""
        safe_name = _sanitize_filename(person_id)
        return self._storage_dir / f"{safe_name}.npz"

    def _validate_record(self, record: EmbeddingRecord) -> None:
        """Validate an embedding record before saving."""
        if not record.person_id or not record.person_id.strip():
            raise ValueError("person_id must be a non-empty string")

        if record.embedding is None:
            raise ValueError("embedding must not be None")

        if record.embedding.ndim != 1:
            raise ValueError(
                f"embedding must be 1-dimensional, got shape {record.embedding.shape}"
            )

        if record.embedding.shape[0] != self._expected_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self._expected_dim}, "
                f"got {record.embedding.shape[0]}"
            )

        if not np.all(np.isfinite(record.embedding)):
            raise ValueError("embedding contains NaN or Inf values")

    def _load_file(self, filepath: Path) -> Optional[EmbeddingRecord]:
        """Load and validate a single .npz file."""
        try:
            data = np.load(filepath, allow_pickle=False)
        except Exception as e:
            _logger.warning(f"Cannot read {filepath.name}: {e}")
            return None

        # Validate required fields
        required_keys = {"embedding", "person_id", "created_at"}
        if not required_keys.issubset(data.files):
            _logger.warning(
                f"Corrupted embedding file {filepath.name}: "
                f"missing keys {required_keys - set(data.files)}"
            )
            return None

        embedding = data["embedding"].astype(np.float32)

        # Validate dimension
        if embedding.ndim != 1 or embedding.shape[0] != self._expected_dim:
            _logger.warning(
                f"Dimension mismatch in {filepath.name}: "
                f"expected ({self._expected_dim},), got {embedding.shape}"
            )
            return None

        # Reconstruct metadata dict
        metadata: Dict[str, str] = {}
        if "metadata_keys" in data.files and "metadata_values" in data.files:
            keys = data["metadata_keys"].tolist()
            values = data["metadata_values"].tolist()
            if len(keys) == len(values):
                metadata = dict(zip(keys, values))

        return EmbeddingRecord(
            person_id=str(data["person_id"]),
            embedding=embedding,
            metadata=metadata,
            created_at=str(data["created_at"]),
        )
