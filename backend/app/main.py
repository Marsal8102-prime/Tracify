from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.config import Settings
from backend.app.errors import install_exception_handlers
from backend.app.middleware import RequestIDMiddleware
from backend.app.routes.health import router as health_router
from backend.app.version import VERSION


SettingsFactory = Callable[[], Settings]
HTTPClientFactory = Callable[[Settings], httpx.AsyncClient]


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
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded_settings = settings if settings is not None else settings_factory()
        http_client: httpx.AsyncClient | None = None
        app.state.ml_engine_client = None
        try:
            http_client = http_client_factory(loaded_settings)
            app.state.ml_engine_client = MLEngineClient(
                http_client,
                loaded_settings.ml_engine_health_timeout_seconds,
            )
            yield
        finally:
            app.state.ml_engine_client = None
            if http_client is not None:
                await http_client.aclose()

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
