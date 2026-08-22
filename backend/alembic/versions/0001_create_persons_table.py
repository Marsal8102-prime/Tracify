"""create persons table

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "persons",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("person_id", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column("department", sa.String(120), nullable=True),
        sa.Column("designation", sa.String(120), nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "metadata",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_persons"),
        sa.UniqueConstraint("person_id", name="uq_persons_person_id"),
        sa.CheckConstraint(
            "person_id <> ''",
            name="ck_persons_person_id_not_blank",
        ),
        sa.CheckConstraint(
            "display_name <> ''",
            name="ck_persons_display_name_not_blank",
        ),
        sa.CheckConstraint(
            "department IS NULL OR department <> ''",
            name="ck_persons_department_not_blank",
        ),
        sa.CheckConstraint(
            "designation IS NULL OR designation <> ''",
            name="ck_persons_designation_not_blank",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'inactive', 'archived')",
            name="ck_persons_status_allowed",
        ),
    )
    op.create_index("ix_persons_status", "persons", ["status"])


def downgrade() -> None:
    op.drop_index("ix_persons_status", table_name="persons")
    op.drop_table("persons")
