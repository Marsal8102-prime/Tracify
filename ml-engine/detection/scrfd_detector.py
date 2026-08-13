"""
Detection — SCRFD face detector implementation (via InsightFace).

SCRFD (Sample and Computation Redistribution for Face Detection) is a
state-of-the-art lightweight face detector. We use it through the
InsightFace package which provides pre-trained ONNX models.

Model details:
  - buffalo_l pack includes SCRFD-10GF (det_10g.onnx)
  - Returns bounding boxes + 5-point landmarks per face
  - Runs on ONNX Runtime (CPU or GPU)

Usage:
    from detection.scrfd_detector import SCRFDDetector
    from config import load_settings

    settings = load_settings()
    detector = SCRFDDetector(settings.detection)
    detector.load_model()
    results = detector.detect(frame)
"""

from __future__ import annotations

from typing import List

import numpy as np

from config.settings import DetectionSettings
from detection.base import BaseDetector, DetectionResult
from utils.logger import get_logger
from utils.timing import timed

_logger = get_logger("tracify.detection.scrfd")


class SCRFDDetector(BaseDetector):
    """
    Face detector using InsightFace's SCRFD model.

    Wraps the InsightFace FaceAnalysis app to perform detection and
    return standardized DetectionResult objects.
    """

    def __init__(self, config: DetectionSettings):
        """
        Args:
            config: DetectionSettings from the loaded configuration.
        """
        self._model_name = config.model_name
        self._confidence = config.confidence_threshold
        self._nms_threshold = config.nms_threshold
        self._max_faces = config.max_faces
        self._input_size = tuple(config.input_size)
        self._app = None  # InsightFace FaceAnalysis instance

        _logger.info(
            f"SCRFDDetector configured: model={self._model_name}, "
            f"confidence={self._confidence}, input_size={self._input_size}"
        )

    def load_model(self) -> None:
        """
        Load the InsightFace model pack.

        InsightFace downloads the model pack (~600MB for buffalo_l) on first
        run to ~/.insightface/models/. Subsequent loads are from cache.
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace is not installed. Run: pip install insightface onnxruntime"
            )

        _logger.info(f"Loading InsightFace model pack: {self._model_name}")

        self._app = FaceAnalysis(
            name=self._model_name,
            # Only load the detection model, skip recognition for now
            # (recognition model will be loaded by the embedding module)
            allowed_modules=["detection"],
        )
        self._app.prepare(
            ctx_id=0,  # 0 = GPU if available, -1 = CPU only
            det_size=self._input_size,
            det_thresh=self._confidence,
        )

        _logger.info(
            f"SCRFD model loaded successfully "
            f"(det_size={self._input_size}, providers={self._get_providers()})"
        )

    @timed(name="detection.detect")
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of DetectionResult sorted by confidence (highest first).
        """
        if self._app is None:
            raise RuntimeError(
                "Detector model not loaded. Call load_model() first."
            )

        # InsightFace expects BGR input (same as OpenCV default)
        faces = self._app.get(frame)

        # Convert InsightFace Face objects to our DetectionResult format
        results: List[DetectionResult] = []
        for face in faces:
            det = DetectionResult(
                bbox=np.array(face.bbox, dtype=np.float32),
                confidence=float(face.det_score),
                landmarks=(
                    np.array(face.kps, dtype=np.float32)
                    if face.kps is not None
                    else None
                ),
            )
            results.append(det)

        # Sort by confidence (highest first)
        results.sort(key=lambda d: d.confidence, reverse=True)

        # Apply max_faces limit
        if self._max_faces > 0:
            results = results[: self._max_faces]

        _logger.debug(
            f"Detected {len(results)} face(s) in frame "
            f"({frame.shape[1]}x{frame.shape[0]})"
        )

        return results

    @property
    def is_loaded(self) -> bool:
        return self._app is not None

    def _get_providers(self) -> list[str]:
        """Get the ONNX Runtime execution providers being used."""
        try:
            import onnxruntime
            return onnxruntime.get_available_providers()
        except Exception:
            return ["unknown"]
