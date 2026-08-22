"""Alembic environment for async migrations.

Supports:
- Online async migrations with NullPool.
- Offline SQL generation with literal_binds and PostgreSQL dialect.
- Optional test URL injection via ``config.attributes["database_url"]``.
- Secure percent-escaping for ConfigParser safety.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models so Base.metadata is complete.
import backend.app.models  # noqa: F401
from backend.app.models.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    """Resolve the database URL for this Alembic invocation.

    Priority:
    1. ``config.attributes["database_url"]`` — test injection.
    2. ``Settings().database_url`` — production typed settings.
    """
    injected = config.attributes.get("database_url", None)
    if injected is not None:
        return injected

    from backend.app.config import Settings

    settings = Settings()
    return settings.database_url.get_secret_value()


def _set_url() -> None:
    """Escape and publish the URL so Alembic/ConfigParser can consume it."""
    raw = _get_url()
    escaped = raw.replace("%", "%%")
    config.set_main_option("sqlalchemy.url", escaped)


def run_migrations_offline() -> None:
    """Generate SQL without a live database connection."""
    _set_url()
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        dialect_name="postgresql",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations against a live async engine."""
    _set_url()
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — bridges async."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
