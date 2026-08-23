"""Recognition event ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, JSON, String, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, utc_now


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    person_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey("persons.person_id", ondelete="SET NULL"),
        nullable=True,
    )
    recognition_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    similarity: Mapped[float] = mapped_column(
        nullable=False,
    )
    threshold: Mapped[float] = mapped_column(
        nullable=False,
    )
    detection_confidence: Mapped[float] = mapped_column(
        nullable=False,
    )
    bbox: Mapped[list[int]] = mapped_column(
        JSON,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", uuid.uuid4())
        kwargs.setdefault("created_at", utc_now())
        super().__init__(**kwargs)

    __table_args__ = (
        CheckConstraint(
            "recognition_status IN ('known', 'unknown')",
            name="recognition_status_allowed",
        ),
        CheckConstraint(
            "similarity >= -1.0 AND similarity <= 1.0",
            name="similarity_range",
        ),
        CheckConstraint(
            "threshold >= -1.0 AND threshold <= 1.0",
            name="threshold_range",
        ),
        CheckConstraint(
            "detection_confidence >= 0.0 AND detection_confidence <= 1.0",
            name="detection_confidence_range",
        ),
        Index("ix_recognition_events_person_id", "person_id"),
    )
