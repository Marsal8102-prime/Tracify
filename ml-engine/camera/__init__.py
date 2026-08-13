"""
Camera package — frame capture from USB, RTSP, and video file sources.

Usage:
    from camera import create_camera, BaseCamera
    from config import load_settings

    settings = load_settings()
    camera = create_camera(settings.camera)
    with camera:
        frame = camera.read_frame()
"""

from camera.base import BaseCamera
from camera.usb_camera import USBCamera
from camera.rtsp_camera import RTSPCamera
from camera.video_file import VideoFileSource
from camera.factory import create_camera

__all__ = [
    "BaseCamera",
    "USBCamera",
    "RTSPCamera",
    "VideoFileSource",
    "create_camera",
]
