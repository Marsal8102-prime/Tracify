"""
Detection package — face detection with bounding boxes and landmarks.

Usage:
    from detection import create_detector, BaseDetector, DetectionResult
    from config import load_settings

    settings = load_settings()
    detector = create_detector(settings.detection)
    detector.load_model()
    results = detector.detect(frame)
"""

from detection.base import BaseDetector, DetectionResult
from detection.scrfd_detector import SCRFDDetector
from detection.factory import create_detector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "SCRFDDetector",
    "create_detector",
]
