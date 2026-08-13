"""
Camera — USB webcam frame source.

Wraps OpenCV VideoCapture for local USB/built-in cameras.

Usage:
    from camera.usb_camera import USBCamera
    cam = USBCamera(device_id=0, width=1280, height=720)
    with cam:
        frame = cam.read_frame()
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from camera.base import BaseCamera
from utils.logger import get_logger

_logger = get_logger("tracify.camera.usb")


class USBCamera(BaseCamera):
    """Frame source for USB / built-in webcams."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Args:
            device_id: USB camera device index (0 = default camera).
            width: Requested capture width.
            height: Requested capture height.
            fps: Requested capture FPS.
        """
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        _logger.info(f"Opening USB camera (device={self._device_id})")
        self._cap = cv2.VideoCapture(self._device_id)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open USB camera with device_id={self._device_id}. "
                "Check that the camera is connected and not in use."
            )

        # Request desired resolution and FPS (camera may not honor these)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _logger.info(
            f"USB camera opened: requested=({self._width}x{self._height}), "
            f"actual=({actual_w}x{actual_h})"
        )

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            _logger.warning("USB camera: failed to read frame")
            return None

        return frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            _logger.info("USB camera released")

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        if self._cap is None or not self._cap.isOpened():
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
