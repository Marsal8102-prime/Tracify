"""Tests for database session dependency behavior."""

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.models.base import Base
from backend.app.models.person import Person


async def test_session_opens_and_closes():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    await engine.dispose()


async def test_session_rolls_back_on_exception():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with factory() as session:
            person = Person(
                person_id="rollback-test",
                display_name="Rollback",
                status="active",
                profile_metadata={},
            )
            session.add(person)
            await session.flush()
            raise ValueError("simulated failure")
    except ValueError:
        pass

    # Verify the record was NOT persisted
    async with factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM persons WHERE person_id = 'rollback-test'")
        )
        assert result.scalar() == 0

    await engine.dispose()


async def test_session_does_not_auto_commit():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Add without committing
    async with factory() as session:
        person = Person(
            person_id="no-autocommit",
            display_name="NoCommit",
            status="active",
            profile_metadata={},
        )
        session.add(person)
        await session.flush()
        # Session closes without commit

    # Verify the record was NOT persisted
    async with factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM persons WHERE person_id = 'no-autocommit'")
        )
        assert result.scalar() == 0

    await engine.dispose()


async def test_explicit_commit_persists():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        person = Person(
            person_id="committed-test",
            display_name="Committed",
            status="active",
            profile_metadata={},
        )
        session.add(person)
        await session.commit()

    # Verify persisted
    async with factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM persons WHERE person_id = 'committed-test'")
        )
        assert result.scalar() == 1

    await engine.dispose()
