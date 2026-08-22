from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.config import Settings
from backend.app.database import (
    DatabaseHealthCheck,
    create_database_engine,
    create_database_health_checker,
    create_session_factory,
)
from backend.app.errors import install_exception_handlers
from backend.app.middleware import RequestIDMiddleware
from backend.app.routes.health import router as health_router
from backend.app.version import VERSION


SettingsFactory = Callable[[], Settings]
HTTPClientFactory = Callable[[Settings], httpx.AsyncClient]
DatabaseEngineFactory = Callable[[Settings], AsyncEngine]
SessionFactoryBuilder = Callable[[AsyncEngine], async_sessionmaker[AsyncSession]]
DatabaseHealthFactory = Callable[[AsyncEngine, Settings], DatabaseHealthCheck]


def _create_http_client(settings: Settings) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        connect=settings.ml_engine_connect_timeout_seconds,
        read=settings.ml_engine_read_timeout_seconds,
        write=settings.ml_engine_write_timeout_seconds,
        pool=settings.ml_engine_pool_timeout_seconds,
    )
    return httpx.AsyncClient(
        base_url=settings.ml_engine_base_url,
        timeout=timeout,
        trust_env=False,
    )


def create_app(
    *,
    settings: Settings | None = None,
    settings_factory: SettingsFactory = Settings,
    http_client_factory: HTTPClientFactory = _create_http_client,
    database_engine_factory: DatabaseEngineFactory = create_database_engine,
    database_session_factory: SessionFactoryBuilder = create_session_factory,
    database_health_factory: DatabaseHealthFactory = create_database_health_checker,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded_settings = settings if settings is not None else settings_factory()

        app.state.ml_engine_client = None
        app.state.database_engine = None
        app.state.database_session_factory = None
        app.state.database_health_checker = None

        async with AsyncExitStack() as stack:
            try:
                # HTTP client
                http_client = http_client_factory(loaded_settings)
                stack.push_async_callback(http_client.aclose)

                # Database engine
                engine = database_engine_factory(loaded_settings)
                stack.push_async_callback(engine.dispose)

                # Session factory and health checker
                session_factory = database_session_factory(engine)
                health_checker = database_health_factory(engine, loaded_settings)

                # Publish fully initialized resources
                app.state.ml_engine_client = MLEngineClient(
                    http_client,
                    loaded_settings.ml_engine_health_timeout_seconds,
                )
                app.state.database_engine = engine
                app.state.database_session_factory = session_factory
                app.state.database_health_checker = health_checker

                yield
            finally:
                app.state.ml_engine_client = None
                app.state.database_engine = None
                app.state.database_session_factory = None
                app.state.database_health_checker = None

    application = FastAPI(
        title="Tracify Backend API",
        version=VERSION,
        lifespan=lifespan,
    )
    application.add_middleware(RequestIDMiddleware)
    install_exception_handlers(application)
    application.include_router(health_router)
    return application


app = create_app()
