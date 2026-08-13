"""
Recognition package — face recognition and identity matching.

Matches query face embeddings against a gallery of known identities
using configurable cosine similarity thresholds.

Usage:
    from recognition import EmbeddingRecognizer, RecognitionResult, RecognitionStatus
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

from recognition.base import BaseRecognizer
from recognition.result import (
    MatchCandidate,
    RecognitionResult,
    RecognitionStatus,
)
from recognition.similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    validate_embedding,
)
from recognition.embedding_recognizer import EmbeddingRecognizer

__all__ = [
    "BaseRecognizer",
    "EmbeddingRecognizer",
    "MatchCandidate",
    "RecognitionResult",
    "RecognitionStatus",
    "cosine_similarity",
    "cosine_similarity_batch",
    "validate_embedding",
]
