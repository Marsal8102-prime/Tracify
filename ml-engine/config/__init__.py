"""
Config package — centralized configuration for Tracify ML Engine.

Usage:
    from config import load_settings, Settings
    settings = load_settings()
"""

from config.settings import (
    Settings,
    load_settings,
    ML_ENGINE_ROOT,
    PROJECT_ROOT,
    CameraSettings,
    PreprocessingSettings,
    DetectionSettings,
    AlignmentSettings,
    EmbeddingSettings,
    RecognitionSettings,
    RegistrationSettings,
    AttendanceSettings,
    AlertSettings,
    StorageSettings,
    LoggingSettings,
)

__all__ = [
    "Settings",
    "load_settings",
    "ML_ENGINE_ROOT",
    "PROJECT_ROOT",
    "CameraSettings",
    "PreprocessingSettings",
    "DetectionSettings",
    "AlignmentSettings",
    "EmbeddingSettings",
    "RecognitionSettings",
    "RegistrationSettings",
    "AttendanceSettings",
    "AlertSettings",
    "StorageSettings",
    "LoggingSettings",
]
