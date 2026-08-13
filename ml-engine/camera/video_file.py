"""
Camera — Video file playback source.

Plays back a video file as if it were a live camera feed.
Essential for development, testing, and demo scenarios where
a real camera isn't available.

Usage:
    from camera.video_file import VideoFileSource
    cam = VideoFileSource(path="data/sample/test_video.mp4", loop=True)
    with cam:
        frame = cam.read_frame()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from camera.base import BaseCamera
from utils.logger import get_logger

_logger = get_logger("tracify.camera.video_file")


class VideoFileSource(BaseCamera):
    """Frame source that reads from a video file on disk."""

    def __init__(self, path: str, loop: bool = False):
        """
        Args:
            path: Path to the video file.
            loop: If True, restart from the beginning when the video ends.
        """
        self._path = Path(path)
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0
        self._total_frames: int = 0

    def open(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        _logger.info(f"Opening video file: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        _logger.info(
            f"Video opened: {w}x{h}, {fps:.1f} FPS, "
            f"{self._total_frames} frames, loop={self._loop}"
        )
        self._frame_count = 0

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            if self._loop:
                _logger.debug("Video ended, looping back to start")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_count = 0
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                _logger.info(
                    f"Video playback complete ({self._frame_count} frames read)"
                )
                return None

        self._frame_count += 1
        return frame

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            _logger.info(
                f"Video file released ({self._frame_count} frames read)"
            )

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        if self._cap is None or not self._cap.isOpened():
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def progress(self) -> float:
        """Return playback progress as a percentage (0.0 - 100.0)."""
        if self._total_frames == 0:
            return 0.0
        return (self._frame_count / self._total_frames) * 100.0
