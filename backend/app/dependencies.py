from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.database import DatabaseHealthCheck
from backend.app.errors import ServiceUnavailableError


def get_ml_engine_client(request: Request) -> MLEngineClient:
    client = getattr(request.app.state, "ml_engine_client", None)
    if client is None:
        raise ServiceUnavailableError()
    return client


def get_database_session_factory(
    request: Request,
) -> async_sessionmaker[AsyncSession]:
    factory = getattr(request.app.state, "database_session_factory", None)
    if factory is None:
        raise ServiceUnavailableError()
    return factory


async def get_database_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = get_database_session_factory(request)
    async with factory() as session:
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise


def get_database_health_checker(request: Request) -> DatabaseHealthCheck:
    checker = getattr(request.app.state, "database_health_checker", None)
    if checker is None:
        raise ServiceUnavailableError()
    return checker
