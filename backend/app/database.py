"""Database engine, session factory, and health-check infrastructure."""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.config import Settings
from backend.app.errors import DatabaseUnavailableError


# ---------------------------------------------------------------------------
# Protocols / type aliases
# ---------------------------------------------------------------------------


@runtime_checkable
class DatabaseHealthCheck(Protocol):
    """Narrow abstraction for database readiness probes."""

    async def check(self) -> None: ...


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Build an async engine from *settings*.

    The engine is lazy — no connection is made until a session is used.
    """
    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout_seconds,
        pool_pre_ping=True,
        echo=False,
    )


# ---------------------------------------------------------------------------
# Session factory builder
# ---------------------------------------------------------------------------


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a sessionmaker bound to *engine*."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
    )


# ---------------------------------------------------------------------------
# Health checker
# ---------------------------------------------------------------------------


class _DatabaseHealthChecker:
    """Executes ``SELECT 1`` against the engine under a timeout."""

    def __init__(self, engine: AsyncEngine, timeout_seconds: float) -> None:
        self._engine = engine
        self._timeout_seconds = timeout_seconds

    async def check(self) -> None:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
        except TimeoutError as exc:
            raise DatabaseUnavailableError() from exc
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError() from exc


def create_database_health_checker(
    engine: AsyncEngine,
    settings: Settings,
) -> DatabaseHealthCheck:
    """Return a production health checker for *engine*."""
    return _DatabaseHealthChecker(engine, settings.database_health_timeout_seconds)
