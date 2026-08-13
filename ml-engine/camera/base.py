"""
Camera — Abstract base class for all frame sources.

Defines the contract that every camera implementation must follow.
This allows the rest of the pipeline to work with any source
(USB, RTSP, video file) without knowing the specifics.

Usage:
    camera = create_camera(settings.camera)  # via factory
    with camera:
        frame = camera.read_frame()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseCamera(ABC):
    """
    Abstract base class for frame capture sources.

    All camera implementations must support:
      - Opening/closing the source
      - Reading individual frames
      - Context manager protocol (with statement)
    """

    @abstractmethod
    def open(self) -> None:
        """Open the camera/stream and prepare for frame capture."""
        ...

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the source.

        Returns:
            BGR numpy array (H, W, 3) on success, or None if no frame
            is available (e.g., stream dropped, end of file).
        """
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """Return True if the source is currently open and readable."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release the source and free resources."""
        ...

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        """Return (width, height) of frames, or None if not opened."""
        return None

    # ── Context manager support ──

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # Don't suppress exceptions
