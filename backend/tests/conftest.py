import httpx
import pytest

from backend.app.config import Settings
from backend.app.main import create_app


ML_ENGINE_ENV_VARS = (
    "ML_ENGINE_URL",
    "ML_ENGINE_CONNECT_TIMEOUT_SECONDS",
    "ML_ENGINE_READ_TIMEOUT_SECONDS",
    "ML_ENGINE_WRITE_TIMEOUT_SECONDS",
    "ML_ENGINE_POOL_TIMEOUT_SECONDS",
    "ML_ENGINE_HEALTH_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def isolate_ml_engine_environment(monkeypatch):
    for name in ML_ENGINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def healthy_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "version": "0.1.0",
        "models_loaded": True,
        "gallery_loaded": True,
        "gallery_size": 0,
    }


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def healthy_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=healthy_payload()))


@pytest.fixture
def app(settings: Settings, healthy_transport: httpx.MockTransport):
    return create_app(
        settings=settings,
        http_client_factory=lambda config: httpx.AsyncClient(
            base_url=config.ml_engine_base_url,
            transport=healthy_transport,
        ),
    )


async def request_app(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)
