"""
Embedding package — face embedding extraction.

Converts aligned face crops into normalized embedding vectors.

Usage:
    from embedding import ArcFaceEmbedder, BaseEmbedder
    from config import load_settings

    settings = load_settings()
    embedder = ArcFaceEmbedder(settings.embedding)
    embedder.load_model()
    vector = embedder.generate(aligned_face)
"""

from embedding.base import BaseEmbedder
from embedding.arcface_embedder import ArcFaceEmbedder

__all__ = [
    "BaseEmbedder",
    "ArcFaceEmbedder",
]
