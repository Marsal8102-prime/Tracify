"""
Recognition — Similarity calculation for normalized face embeddings.

Provides cosine similarity computation and embedding validation
utilities used by all recognizer implementations.

For L2-normalized embeddings (as produced by Phase 2 ArcFace):
    cosine_similarity(a, b) = dot(a, b)

This module keeps similarity logic isolated from the matching engine
so it can be reused, tested, and potentially swapped independently.

Usage:
    from recognition.similarity import cosine_similarity, cosine_similarity_batch

    score = cosine_similarity(query, stored)
    scores = cosine_similarity_batch(query, gallery_matrix)
"""

from __future__ import annotations

import numpy as np

from utils.logger import get_logger

_logger = get_logger("tracify.recognition.similarity")


def validate_embedding(
    embedding: np.ndarray,
    expected_dim: int,
    *,
    label: str = "embedding",
) -> None:
    """
    Validate that an embedding is well-formed for similarity computation.

    Args:
        embedding: The embedding vector to validate.
        expected_dim: Expected dimensionality (e.g. 512).
        label: Human-readable label for error messages.

    Raises:
        TypeError: If embedding is not a numpy array.
        ValueError: If the embedding has wrong shape, dimension,
            contains NaN/Inf, or is a zero vector.
    """
    if not isinstance(embedding, np.ndarray):
        raise TypeError(
            f"{label} must be a numpy ndarray, got {type(embedding).__name__}"
        )

    if embedding.ndim != 1:
        raise ValueError(
            f"{label} must be 1-dimensional, got shape {embedding.shape}"
        )

    if embedding.shape[0] != expected_dim:
        raise ValueError(
            f"{label} dimension mismatch: expected {expected_dim}, "
            f"got {embedding.shape[0]}"
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"{label} contains NaN or Inf values")

    if np.linalg.norm(embedding) < 1e-10:
        raise ValueError(f"{label} is a zero vector")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized embedding vectors.

    For normalized vectors: cosine_similarity = dot(a, b).
    Both inputs are validated before computation.

    Args:
        a: First embedding vector, shape (dim,).
        b: Second embedding vector, shape (dim,).

    Returns:
        Similarity score as float. For normalized vectors, range is [-1.0, 1.0].

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If inputs have mismatched dimensions, wrong shape,
            contain NaN/Inf, or are zero vectors.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Both inputs must be numpy ndarrays")

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"Both inputs must be 1-dimensional, got shapes {a.shape} and {b.shape}"
        )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Dimension mismatch: {a.shape[0]} vs {b.shape[0]}"
        )

    if not np.all(np.isfinite(a)):
        raise ValueError("First embedding contains NaN or Inf values")

    if not np.all(np.isfinite(b)):
        raise ValueError("Second embedding contains NaN or Inf values")

    if np.linalg.norm(a) < 1e-10:
        raise ValueError("First embedding is a zero vector")

    if np.linalg.norm(b) < 1e-10:
        raise ValueError("Second embedding is a zero vector")

    # For L2-normalized vectors, cosine similarity = dot product
    return float(np.dot(a, b))


def cosine_similarity_batch(
    query: np.ndarray,
    gallery: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a query and a gallery of embeddings.

    Uses vectorized NumPy operations for efficient batch comparison.
    Assumes all vectors are L2-normalized.

    Args:
        query: Query embedding, shape (dim,).
        gallery: Gallery embeddings, shape (N, dim).

    Returns:
        Similarity scores as np.ndarray of shape (N,).

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If shapes are incompatible, query contains NaN/Inf,
            or query is a zero vector.
    """
    if not isinstance(query, np.ndarray) or not isinstance(gallery, np.ndarray):
        raise TypeError("Both inputs must be numpy ndarrays")

    if query.ndim != 1:
        raise ValueError(
            f"Query must be 1-dimensional, got shape {query.shape}"
        )

    if gallery.ndim != 2:
        raise ValueError(
            f"Gallery must be 2-dimensional (N, dim), got shape {gallery.shape}"
        )

    if query.shape[0] != gallery.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {query.shape[0]}, "
            f"gallery has {gallery.shape[1]}"
        )

    if not np.all(np.isfinite(query)):
        raise ValueError("Query embedding contains NaN or Inf values")

    if np.linalg.norm(query) < 1e-10:
        raise ValueError("Query embedding is a zero vector")

    # Vectorized dot product: (N, dim) @ (dim,) → (N,)
    return gallery @ query
