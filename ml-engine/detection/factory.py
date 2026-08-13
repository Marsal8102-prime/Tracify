"""
Detection — Factory function for creating detector instances from config.

Usage:
    from detection.factory import create_detector
    from config import load_settings

    settings = load_settings()
    detector = create_detector(settings.detection)
    detector.load_model()
"""

from __future__ import annotations

from config.settings import DetectionSettings
from detection.base import BaseDetector
from utils.logger import get_logger

_logger = get_logger("tracify.detection.factory")

# Registry of supported detector backends
_DETECTOR_REGISTRY = {
    "scrfd": "detection.scrfd_detector.SCRFDDetector",
}


def create_detector(config: DetectionSettings) -> BaseDetector:
    """
    Factory: create the right detector based on config.

    Args:
        config: DetectionSettings from the loaded configuration.

    Returns:
        A BaseDetector instance (not yet loaded — call .load_model()).

    Raises:
        ValueError: If the detector backend is not supported.
    """
    backend = config.backend.lower()

    if backend not in _DETECTOR_REGISTRY:
        supported = ", ".join(_DETECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown detector backend: '{backend}'. "
            f"Supported backends: {supported}"
        )

    _logger.info(f"Creating detector: backend={backend}")

    # Lazy import to avoid loading heavy dependencies until needed
    if backend == "scrfd":
        from detection.scrfd_detector import SCRFDDetector
        return SCRFDDetector(config)

    # Extensibility: add new backends here
    raise ValueError(f"Backend '{backend}' is registered but not implemented.")
