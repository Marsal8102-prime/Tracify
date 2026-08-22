"""Tests for database health check behavior."""

import asyncio

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.config import Settings
from backend.app.database import _DatabaseHealthChecker, create_database_health_checker
from backend.app.errors import DatabaseUnavailableError
from backend.app.models.base import Base


async def test_health_check_executes_select_1():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    checker = _DatabaseHealthChecker(engine, timeout_seconds=5.0)
    # Should not raise
    await checker.check()
    await engine.dispose()


async def test_health_check_uses_configured_timeout():
    settings = Settings(_env_file=None, database_health_timeout_seconds=2.5)
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    checker = create_database_health_checker(engine, settings)
    assert checker._timeout_seconds == 2.5
    await engine.dispose()


async def test_health_check_timeout_maps_to_unavailable():
    from unittest.mock import MagicMock, AsyncMock
    from contextlib import asynccontextmanager

    conn = AsyncMock()
    @asynccontextmanager
    async def slow_connect(*args, **kwargs):
        await asyncio.sleep(1.0)
        yield conn

    engine = MagicMock()
    engine.connect = slow_connect

    checker = _DatabaseHealthChecker(engine, timeout_seconds=0.0)

    with pytest.raises(DatabaseUnavailableError):
        await checker.check()


async def test_health_check_sqlalchemy_error_maps_to_unavailable():
    from unittest.mock import MagicMock
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def failing_connect(*args, **kwargs):
        raise SQLAlchemyError("connection failed")
        yield None

    engine = MagicMock()
    engine.connect = failing_connect

    checker = _DatabaseHealthChecker(engine, timeout_seconds=5.0)

    with pytest.raises(DatabaseUnavailableError):
        await checker.check()


async def test_health_check_programming_error_propagates():
    from unittest.mock import MagicMock
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def bug_connect(*args, **kwargs):
        raise RuntimeError("unexpected programming error")
        yield None

    engine = MagicMock()
    engine.connect = bug_connect

    checker = _DatabaseHealthChecker(engine, timeout_seconds=5.0)

    with pytest.raises(RuntimeError, match="unexpected programming error"):
        await checker.check()


async def test_database_unavailable_error_has_no_raw_details():
    try:
        raise DatabaseUnavailableError()
    except DatabaseUnavailableError as exc:
        assert str(exc) == ""
        assert not hasattr(exc, "sql")
        assert not hasattr(exc, "url")
