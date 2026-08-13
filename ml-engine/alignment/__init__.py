"""
Alignment package — face alignment for embedding extraction.

Produces normalized, pose-corrected face crops from detector output.

Usage:
    from alignment import FaceAligner, BaseAligner
    from config import load_settings

    settings = load_settings()
    aligner = FaceAligner(settings.alignment)
    aligned = aligner.align(frame, detection)
"""

from alignment.base import BaseAligner
from alignment.face_aligner import FaceAligner

__all__ = [
    "BaseAligner",
    "FaceAligner",
]
