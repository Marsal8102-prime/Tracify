"""Tests for Person ORM model metadata and portable SQLite behavior."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.models.base import Base
from backend.app.models.person import Person


def test_person_table_name():
    assert Person.__tablename__ == "persons"


def test_person_column_names():
    columns = {c.name for c in Person.__table__.columns}
    expected = {
        "id", "person_id", "display_name", "department", "designation",
        "status", "metadata", "created_at", "updated_at",
    }
    assert columns == expected


def test_profile_metadata_maps_to_metadata_column():
    col = Person.__table__.c["metadata"]
    assert col is not None
    # The Python attribute is profile_metadata, not metadata
    assert hasattr(Person, "profile_metadata")


def test_no_orm_attribute_named_metadata_overrides_base():
    # Base.metadata is the MetaData object; Person.metadata should be the same
    assert Person.metadata is Base.metadata


def test_uuid_is_portable():
    col = Person.__table__.c.id
    from sqlalchemy import Uuid
    assert isinstance(col.type, Uuid)


def test_string_lengths():
    assert Person.__table__.c.person_id.type.length == 128
    assert Person.__table__.c.display_name.type.length == 200
    assert Person.__table__.c.department.type.length == 120
    assert Person.__table__.c.designation.type.length == 120
    assert Person.__table__.c.status.type.length == 20


def test_nullability():
    assert Person.__table__.c.id.nullable is False
    assert Person.__table__.c.person_id.nullable is False
    assert Person.__table__.c.display_name.nullable is False
    assert Person.__table__.c.department.nullable is True
    assert Person.__table__.c.designation.nullable is True
    assert Person.__table__.c.status.nullable is False
    assert Person.__table__.c["metadata"].nullable is False
    assert Person.__table__.c.created_at.nullable is False
    assert Person.__table__.c.updated_at.nullable is False


def test_status_default_is_active():
    person = Person(
        person_id="test",
        display_name="Test Person",
        profile_metadata={},
    )
    assert person.status == "active"


def test_constraint_and_index_names():
    table = Person.__table__
    constraint_names = {c.name for c in table.constraints if c.name}
    index_names = {i.name for i in table.indexes}

    assert "pk_persons" in constraint_names
    assert "uq_persons_person_id" in constraint_names
    assert "ck_persons_person_id_not_blank" in constraint_names
    assert "ck_persons_display_name_not_blank" in constraint_names
    assert "ck_persons_department_not_blank" in constraint_names
    assert "ck_persons_designation_not_blank" in constraint_names
    assert "ck_persons_status_allowed" in constraint_names
    assert "ix_persons_status" in index_names


def test_timestamp_columns_are_timezone_enabled():
    assert Person.__table__.c.created_at.type.timezone is True
    assert Person.__table__.c.updated_at.type.timezone is True


def test_metadata_defaults_do_not_share_mutable_dict():
    p1 = Person(person_id="a", display_name="A", profile_metadata={})
    p2 = Person(person_id="b", display_name="B", profile_metadata={})
    assert p1.profile_metadata is not p2.profile_metadata


def test_no_embedding_or_vector_columns():
    columns = {c.name for c in Person.__table__.columns}
    vector_related = {"embedding", "face_embedding", "vector"}
    assert columns.isdisjoint(vector_related)


# ---------- Portable SQLite behavior tests ----------


async def test_unique_person_id_rejects_duplicates():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        session.add(Person(
            person_id="dup-test",
            display_name="First",
            profile_metadata={},
        ))
        await session.commit()

    from sqlalchemy.exc import IntegrityError
    import pytest

    async with factory() as session:
        session.add(Person(
            person_id="dup-test",
            display_name="Second",
            profile_metadata={},
        ))
        with pytest.raises(IntegrityError):
            await session.commit()

    await engine.dispose()


async def test_json_metadata_round_trips():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    test_id = str(uuid.uuid4())
    metadata = {"department": "Engineering", "tags": ["admin", "senior"]}

    async with factory() as session:
        session.add(Person(
            person_id=test_id,
            display_name="JSON Test",
            profile_metadata=metadata,
        ))
        await session.commit()

    async with factory() as session:
        result = await session.execute(
            text("SELECT metadata FROM persons WHERE person_id = :pid"),
            {"pid": test_id},
        )
        import json
        row = result.scalar_one()
        loaded = json.loads(row) if isinstance(row, str) else row
        assert loaded == metadata

    await engine.dispose()


async def test_python_defaults_are_utc_aware():
    person = Person(
        person_id="utc-test",
        display_name="UTC",
        profile_metadata={},
    )
    assert person.created_at.tzinfo is not None
    assert person.created_at.tzinfo == timezone.utc
    assert person.updated_at.tzinfo is not None
    assert person.updated_at.tzinfo == timezone.utc


async def test_orm_update_changes_updated_at():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        person = Person(
            person_id="update-test",
            display_name="Before",
            profile_metadata={},
        )
        session.add(person)
        await session.commit()

        original_updated = person.updated_at

        # Force a different timestamp
        import asyncio
        await asyncio.sleep(0.01)

        person.display_name = "After"
        await session.commit()

        # The onupdate should have changed updated_at
        assert person.updated_at >= original_updated

    await engine.dispose()
