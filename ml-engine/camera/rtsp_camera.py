"""
Camera — RTSP/CCTV stream frame source with auto-reconnect.

Handles unreliable network streams gracefully with exponential
backoff reconnection logic. This is critical for production CCTV
deployments where streams can drop due to network hiccups.

Usage:
    from camera.rtsp_camera import RTSPCamera
    cam = RTSPCamera(url="rtsp://192.168.1.100:554/stream1")
    with cam:
        frame = cam.read_frame()
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from camera.base import BaseCamera
from utils.logger import get_logger

_logger = get_logger("tracify.camera.rtsp")


class RTSPCamera(BaseCamera):
    """Frame source for RTSP / network camera streams with auto-reconnect."""

    def __init__(
        self,
        url: str,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 10,
        width: int = 1280,
        height: int = 720,
    ):
        """
        Args:
            url: RTSP stream URL (e.g., rtsp://user:pass@ip:port/path).
            reconnect_delay: Base delay in seconds between reconnection attempts.
            max_reconnect_attempts: Max attempts before giving up (0 = unlimited).
            width: Requested capture width.
            height: Requested capture height.
        """
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._max_attempts = max_reconnect_attempts
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0

    def open(self) -> None:
        # Sanitize URL for logging (hide credentials)
        safe_url = self._sanitize_url(self._url)
        _logger.info(f"Connecting to RTSP stream: {safe_url}")

        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot connect to RTSP stream: {safe_url}. "
                "Check URL, credentials, and network connectivity."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Reduce RTSP buffer to minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._consecutive_failures = 0
        _logger.info(f"RTSP stream connected: {safe_url}")

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return self._try_reconnect()

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._consecutive_failures += 1
            _logger.warning(
                f"RTSP frame read failed (attempt {self._consecutive_failures})"
            )
            return self._try_reconnect()

        self._consecutive_failures = 0
        return frame

    def _try_reconnect(self) -> Optional[np.ndarray]:
        """Attempt to reconnect with exponential backoff."""
        if self._max_attempts > 0 and self._consecutive_failures >= self._max_attempts:
            _logger.error(
                f"RTSP reconnect failed after {self._max_attempts} attempts. Giving up."
            )
            return None

        # Exponential backoff: 2s, 4s, 8s, ... capped at 60s
        delay = min(
            self._reconnect_delay * (2 ** min(self._consecutive_failures, 5)),
            60.0,
        )
        safe_url = self._sanitize_url(self._url)
        _logger.info(f"RTSP reconnecting in {delay:.1f}s: {safe_url}")
        time.sleep(delay)

        # Release old capture and try again
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _logger.info("RTSP reconnected successfully")
            self._consecutive_failures = 0
            ret, frame = self._cap.read()
            return frame if ret else None

        self._consecutive_failures += 1
        return None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            _logger.info("RTSP stream released")

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        if self._cap is None or not self._cap.isOpened():
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove credentials from URL for safe logging."""
        # rtsp://user:pass@host:port/path → rtsp://***@host:port/path
        if "@" in url:
            protocol_end = url.find("://") + 3
            at_pos = url.find("@")
            return url[:protocol_end] + "***" + url[at_pos:]
        return url
