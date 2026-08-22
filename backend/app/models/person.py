"""Person ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, utc_now


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    person_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
    )
    display_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    department: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
        default=None,
    )
    designation: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
        default=None,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
    )
    profile_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", uuid.uuid4())
        kwargs.setdefault("status", "active")
        kwargs.setdefault("profile_metadata", {})
        kwargs.setdefault("created_at", utc_now())
        kwargs.setdefault("updated_at", utc_now())
        super().__init__(**kwargs)

    __table_args__ = (
        CheckConstraint(
            "person_id <> ''",
            name="person_id_not_blank",
        ),
        CheckConstraint(
            "display_name <> ''",
            name="display_name_not_blank",
        ),
        CheckConstraint(
            "department IS NULL OR department <> ''",
            name="department_not_blank",
        ),
        CheckConstraint(
            "designation IS NULL OR designation <> ''",
            name="designation_not_blank",
        ),
        CheckConstraint(
            "status IN ('active', 'inactive', 'archived')",
            name="status_allowed",
        ),
        Index("ix_persons_status", "status"),
    )
