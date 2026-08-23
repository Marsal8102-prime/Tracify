"""Tests for RecognitionEvent ORM model metadata and portable SQLite behavior."""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.models.base import Base
from backend.app.models.person import Person
from backend.app.models.recognition_event import RecognitionEvent


@pytest.mark.asyncio
async def test_create_recognition_event_known():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        p = Person(person_id="emp_001", display_name="Test")
        session.add(p)
        await session.commit()

        event = RecognitionEvent(
            request_id="req-123",
            person_id=p.person_id,
            recognition_status="known",
            similarity=0.95,
            threshold=0.6,
            detection_confidence=0.99,
            bbox=[100, 150, 300, 400],
        )
        session.add(event)
        await session.commit()

        assert event.id is not None
        assert event.request_id == "req-123"
        assert event.person_id == "emp_001"
        assert event.recognition_status == "known"
        assert event.similarity == 0.95
        assert event.bbox == [100, 150, 300, 400]
        assert event.created_at is not None

    await engine.dispose()


@pytest.mark.asyncio
async def test_create_recognition_event_unknown():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        event = RecognitionEvent(
            request_id="req-456",
            person_id=None,
            recognition_status="unknown",
            similarity=0.3,
            threshold=0.6,
            detection_confidence=0.88,
            bbox=[50, 50, 200, 200],
        )
        session.add(event)
        await session.commit()

        assert event.person_id is None
        assert event.recognition_status == "unknown"

    await engine.dispose()


@pytest.mark.asyncio
async def test_invalid_status_constraint():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        event = RecognitionEvent(
            request_id="req-789",
            person_id=None,
            recognition_status="invalid_status",
            similarity=0.5,
            threshold=0.6,
            detection_confidence=0.9,
            bbox=[0, 0, 10, 10],
        )
        session.add(event)
        with pytest.raises(IntegrityError):
            await session.commit()

    await engine.dispose()


@pytest.mark.asyncio
async def test_invalid_similarity_constraint():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        event = RecognitionEvent(
            request_id="req-111",
            person_id=None,
            recognition_status="unknown",
            similarity=1.5,  # Out of range [-1.0, 1.0]
            threshold=0.6,
            detection_confidence=0.9,
            bbox=[0, 0, 10, 10],
        )
        session.add(event)
        with pytest.raises(IntegrityError):
            await session.commit()

    await engine.dispose()
