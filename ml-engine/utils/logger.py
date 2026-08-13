"""
Utils — Structured logging for Tracify ML Engine.

Provides a unified logger with two modes:
  - "text": human-readable format for development
  - "json": structured JSON lines for production log aggregation

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Detection complete", extra={"faces": 3, "fps": 24.5})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields passed via `extra={...}`
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__ and key not in ("message", "msg"):
                try:
                    json.dumps(value)  # Only include JSON-serializable values
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Human-readable colored format for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


# ── Singleton registry to prevent duplicate handlers ────────────────────
_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_file_size_mb: int = 50,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Args:
        name: Logger name (typically __name__).
        level: Override log level (reads TRACIFY_LOG_LEVEL env var if None).
        log_format: "text" or "json" (reads TRACIFY_LOG_FORMAT env var if None).
        log_dir: Directory for log files. None = console only.
        max_file_size_mb: Max size per log file before rotation.
        backup_count: Number of rotated files to keep.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _configured_loggers:
        return logging.getLogger(name)

    # Resolve settings from env or defaults
    level = (level or os.environ.get("TRACIFY_LOG_LEVEL", "INFO")).upper()
    log_format = (log_format or os.environ.get("TRACIFY_LOG_FORMAT", "text")).lower()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False  # Prevent duplicate logs from root logger

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level, logging.INFO))

    if log_format == "json":
        console_handler.setFormatter(_JSONFormatter())
    else:
        console_handler.setFormatter(
            _TextFormatter(
                fmt="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger.addHandler(console_handler)

    # ── File handler (if log_dir provided) ──
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_path / "tracify.log",
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, level, logging.INFO))
        file_handler.setFormatter(_JSONFormatter())  # Always JSON for files
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger
