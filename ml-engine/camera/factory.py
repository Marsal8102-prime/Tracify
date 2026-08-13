"""
Camera — Factory function for creating camera instances from config.

Reads the camera section of Settings and returns the appropriate
BaseCamera implementation. This keeps the rest of the pipeline
decoupled from specific camera types.

Usage:
    from camera.factory import create_camera
    from config import load_settings

    settings = load_settings()
    camera = create_camera(settings.camera)
"""

from __future__ import annotations

from config.settings import CameraSettings
from camera.base import BaseCamera
from camera.usb_camera import USBCamera
from camera.rtsp_camera import RTSPCamera
from camera.video_file import VideoFileSource
from utils.logger import get_logger

_logger = get_logger("tracify.camera.factory")

# Registry of supported camera types
_CAMERA_REGISTRY = {
    "usb": "_create_usb",
    "rtsp": "_create_rtsp",
    "file": "_create_file",
}


def create_camera(config: CameraSettings) -> BaseCamera:
    """
    Factory: create the right camera instance based on config.

    Args:
        config: CameraSettings from the loaded configuration.

    Returns:
        A BaseCamera instance (not yet opened — call .open() or use `with`).

    Raises:
        ValueError: If the camera type is not supported.
    """
    camera_type = config.type.lower()

    if camera_type not in _CAMERA_REGISTRY:
        supported = ", ".join(_CAMERA_REGISTRY.keys())
        raise ValueError(
            f"Unknown camera type: '{camera_type}'. "
            f"Supported types: {supported}"
        )

    _logger.info(f"Creating camera: type={camera_type}, source={config.source}")
    factory_method = globals()[_CAMERA_REGISTRY[camera_type]]
    return factory_method(config)


def _create_usb(config: CameraSettings) -> USBCamera:
    device_id = int(config.source) if isinstance(config.source, str) else config.source
    return USBCamera(
        device_id=device_id,
        width=config.width,
        height=config.height,
        fps=config.fps,
    )


def _create_rtsp(config: CameraSettings) -> RTSPCamera:
    return RTSPCamera(
        url=str(config.source),
        reconnect_delay=config.reconnect_delay,
        max_reconnect_attempts=config.max_reconnect_attempts,
        width=config.width,
        height=config.height,
    )


def _create_file(config: CameraSettings) -> VideoFileSource:
    return VideoFileSource(
        path=str(config.source),
        loop=False,  # Default to non-looping; can be extended via config
    )
