"""
Storage package — embedding persistence.

Provides an abstract EmbeddingStore interface and a local filesystem
implementation for development.

Usage:
    from storage import LocalEmbeddingStore, EmbeddingStore, EmbeddingRecord
    from config import load_settings

    settings = load_settings()
    store = LocalEmbeddingStore(
        storage_dir=settings.storage.embeddings_dir,
        expected_dimension=settings.embedding.dimension,
    )
"""

from storage.base import EmbeddingStore, EmbeddingRecord
from storage.local_store import LocalEmbeddingStore

__all__ = [
    "EmbeddingStore",
    "EmbeddingRecord",
    "LocalEmbeddingStore",
]
