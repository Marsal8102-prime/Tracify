import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.config import Settings
from backend.app.database import DatabaseHealthCheck
from backend.app.main import create_app


ML_ENGINE_ENV_VARS = (
    "ML_ENGINE_URL",
    "ML_ENGINE_CONNECT_TIMEOUT_SECONDS",
    "ML_ENGINE_READ_TIMEOUT_SECONDS",
    "ML_ENGINE_WRITE_TIMEOUT_SECONDS",
    "ML_ENGINE_POOL_TIMEOUT_SECONDS",
    "ML_ENGINE_HEALTH_TIMEOUT_SECONDS",
)

DATABASE_ENV_VARS = (
    "DATABASE_URL",
    "DATABASE_POOL_SIZE",
    "DATABASE_MAX_OVERFLOW",
    "DATABASE_POOL_TIMEOUT_SECONDS",
    "DATABASE_HEALTH_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in ML_ENGINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in DATABASE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def healthy_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "version": "0.1.0",
        "models_loaded": True,
        "gallery_loaded": True,
        "gallery_size": 0,
    }


class PassingHealthCheck:
    """A database health checker that always passes."""

    async def check(self) -> None:
        pass


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def healthy_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=healthy_payload()))


def _noop_engine_factory(settings):
    """Return a lightweight SQLite engine (never contacts PostgreSQL)."""
    return create_async_engine("sqlite+aiosqlite://", echo=False)


def _noop_session_factory(engine):
    """Return a session factory bound to the given engine."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def _noop_health_factory(engine, settings):
    """Return a health checker that always passes."""
    return PassingHealthCheck()


@pytest.fixture
def app(settings: Settings, healthy_transport: httpx.MockTransport):
    return create_app(
        settings=settings,
        http_client_factory=lambda config: httpx.AsyncClient(
            base_url=config.ml_engine_base_url,
            transport=healthy_transport,
        ),
        database_engine_factory=_noop_engine_factory,
        database_session_factory=_noop_session_factory,
        database_health_factory=_noop_health_factory,
    )


async def request_app(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)
