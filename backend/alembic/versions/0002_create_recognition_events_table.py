"""create recognition_events table

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-23 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "recognition_events",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("person_id", sa.String(128), nullable=True),
        sa.Column("recognition_status", sa.String(20), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("detection_confidence", sa.Float(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_recognition_events"),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["persons.person_id"],
            name="fk_recognition_events_person_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "recognition_status IN ('known', 'unknown')",
            name="ck_recognition_events_status_allowed",
        ),
        sa.CheckConstraint(
            "similarity >= -1.0 AND similarity <= 1.0",
            name="ck_recognition_events_similarity_range",
        ),
        sa.CheckConstraint(
            "threshold >= -1.0 AND threshold <= 1.0",
            name="ck_recognition_events_threshold_range",
        ),
        sa.CheckConstraint(
            "detection_confidence >= 0.0 AND detection_confidence <= 1.0",
            name="ck_recognition_events_detection_confidence_range",
        ),
    )
    op.create_index("ix_recognition_events_request_id", "recognition_events", ["request_id"])
    op.create_index("ix_recognition_events_person_id", "recognition_events", ["person_id"])
    op.create_index("ix_recognition_events_created_at", "recognition_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_recognition_events_created_at", table_name="recognition_events")
    op.drop_index("ix_recognition_events_person_id", table_name="recognition_events")
    op.drop_index("ix_recognition_events_request_id", table_name="recognition_events")
    op.drop_table("recognition_events")
