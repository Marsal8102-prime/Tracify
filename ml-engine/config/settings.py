"""
Config — Settings loader for Tracify ML Engine.

Reads settings.yaml into structured Python dataclasses.
Supports environment variable overrides via .env file.

Usage:
    from config import load_settings
    settings = load_settings()
    print(settings.detection.confidence_threshold)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv


# ── Resolve paths relative to ml-engine/ root ──────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # config/
ML_ENGINE_ROOT = _THIS_DIR.parent                    # ml-engine/
PROJECT_ROOT = ML_ENGINE_ROOT.parent                 # Tracify/
DEFAULT_CONFIG_PATH = _THIS_DIR / "settings.yaml"


# ── Nested dataclasses matching settings.yaml structure ─────────────────

@dataclass
class CameraSettings:
    type: str = "usb"
    source: str | int = 0
    width: int = 1280
    height: int = 720
    fps: int = 15
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 10


@dataclass
class PreprocessingSettings:
    max_dimension: int = 640
    equalize_histogram: bool = False


@dataclass
class DetectionSettings:
    backend: str = "scrfd"
    model_name: str = "buffalo_l"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_faces: int = 0
    input_size: List[int] = field(default_factory=lambda: [640, 640])


@dataclass
class AlignmentSettings:
    output_size: List[int] = field(default_factory=lambda: [112, 112])
    landmark_type: str = "2d"


@dataclass
class EmbeddingSettings:
    backend: str = "arcface"
    model_name: str = "buffalo_l"
    dimension: int = 512
    # ONNX Runtime provider: "cpu" or "gpu"
    provider: str = "cpu"


@dataclass
class RecognitionSettings:
    strategy: str = "cosine"
    similarity_threshold: float = 0.6
    top_k: int = 1


@dataclass
class RegistrationSettings:
    """Configuration for Phase 4 face registration.

    Note:
        Thresholds (duplicate_threshold, minimum_face_size) must be
        calibrated with real validation data before production use.
    """
    minimum_samples: int = 3
    maximum_samples: int = 10
    minimum_face_size: int = 80
    duplicate_threshold: float = 0.7
    quality_checks_enabled: bool = True


@dataclass
class AttendanceSettings:
    cooldown_seconds: int = 300
    timezone: str = "Asia/Kolkata"


@dataclass
class AlertSettings:
    enabled: bool = True
    min_frames: int = 5
    cooldown_seconds: int = 60


@dataclass
class StorageSettings:
    embeddings_dir: str = "storage/embeddings"
    known_faces_dir: str = "storage/known_faces"
    unknown_faces_dir: str = "storage/unknown_faces"
    models_dir: str = "models"

    def resolve(self, root: Path) -> None:
        """Convert relative paths to absolute paths based on ml-engine root."""
        self.embeddings_dir = str(root / self.embeddings_dir)
        self.known_faces_dir = str(root / self.known_faces_dir)
        self.unknown_faces_dir = str(root / self.unknown_faces_dir)
        self.models_dir = str(root / self.models_dir)


@dataclass
class LoggingSettings:
    level: str = "INFO"
    format: str = "text"
    log_dir: str = "logs"
    max_file_size_mb: int = 50
    backup_count: int = 5

    def resolve(self, root: Path) -> None:
        """Convert relative log_dir to absolute path."""
        self.log_dir = str(root / self.log_dir)


# ── Top-level Settings container ────────────────────────────────────────

@dataclass
class Settings:
    """Top-level configuration container for the entire ML engine."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    registration: RegistrationSettings = field(default_factory=RegistrationSettings)
    attendance: AttendanceSettings = field(default_factory=AttendanceSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


# ── Loader ──────────────────────────────────────────────────────────────

def _apply_env_overrides(settings: Settings) -> None:
    """Override settings with TRACIFY_* environment variables if set."""

    env_map = {
        "TRACIFY_CAMERA_TYPE":            lambda v: setattr(settings.camera, "type", v),
        "TRACIFY_CAMERA_SOURCE":          lambda v: setattr(settings.camera, "source", int(v) if v.isdigit() else v),
        "TRACIFY_DETECTION_CONFIDENCE":   lambda v: setattr(settings.detection, "confidence_threshold", float(v)),
        "TRACIFY_DETECTION_MAX_FACES":    lambda v: setattr(settings.detection, "max_faces", int(v)),
        "TRACIFY_EMBEDDING_PROVIDER":     lambda v: setattr(settings.embedding, "provider", v.lower()),
        "TRACIFY_RECOGNITION_THRESHOLD":  lambda v: setattr(settings.recognition, "similarity_threshold", float(v)),
        "TRACIFY_LOG_LEVEL":              lambda v: setattr(settings.logging, "level", v.upper()),
        "TRACIFY_LOG_FORMAT":             lambda v: setattr(settings.logging, "format", v.lower()),
        "TRACIFY_MODELS_DIR":             lambda v: setattr(settings.storage, "models_dir", v),
        "TRACIFY_STORAGE_DIR":            lambda v: _set_storage_root(settings.storage, v),
        "TRACIFY_LOGS_DIR":               lambda v: setattr(settings.logging, "log_dir", v),
    }

    for env_key, setter in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            setter(value)


def _set_storage_root(storage: StorageSettings, root: str) -> None:
    """Override all storage paths with a new root."""
    storage.embeddings_dir = f"{root}/embeddings"
    storage.known_faces_dir = f"{root}/known_faces"
    storage.unknown_faces_dir = f"{root}/unknown_faces"


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_settings(
    config_path: Optional[str | Path] = None,
    load_env: bool = True,
) -> Settings:
    """
    Load settings from YAML config file with optional .env overrides.

    Args:
        config_path: Path to settings.yaml. Defaults to config/settings.yaml.
        load_env: Whether to load .env file and apply TRACIFY_* overrides.

    Returns:
        Fully populated Settings dataclass.
    """
    # Load .env file from project root if it exists
    if load_env:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    # Read YAML
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Build Settings from YAML
    settings = Settings(
        camera=_dict_to_dataclass(CameraSettings, raw.get("camera")),
        preprocessing=_dict_to_dataclass(PreprocessingSettings, raw.get("preprocessing")),
        detection=_dict_to_dataclass(DetectionSettings, raw.get("detection")),
        alignment=_dict_to_dataclass(AlignmentSettings, raw.get("alignment")),
        embedding=_dict_to_dataclass(EmbeddingSettings, raw.get("embedding")),
        recognition=_dict_to_dataclass(RecognitionSettings, raw.get("recognition")),
        registration=_dict_to_dataclass(RegistrationSettings, raw.get("registration")),
        attendance=_dict_to_dataclass(AttendanceSettings, raw.get("attendance")),
        alerts=_dict_to_dataclass(AlertSettings, raw.get("alerts")),
        storage=_dict_to_dataclass(StorageSettings, raw.get("storage")),
        logging=_dict_to_dataclass(LoggingSettings, raw.get("logging")),
    )

    # Apply environment variable overrides
    if load_env:
        _apply_env_overrides(settings)

    # Resolve relative paths to absolute
    settings.storage.resolve(ML_ENGINE_ROOT)
    settings.logging.resolve(ML_ENGINE_ROOT)

    return settings
