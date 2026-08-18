from unittest.mock import AsyncMock

import httpx
import pytest

from backend.app.main import _create_http_client, create_app


async def test_lifespan_constructs_one_client_and_closes_once(settings):
    http_client = AsyncMock(spec=httpx.AsyncClient)
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return http_client

    app = create_app(settings=settings, http_client_factory=factory)
    assert not hasattr(app.state, "ml_engine_client")

    async with app.router.lifespan_context(app):
        assert app.state.ml_engine_client is not None
        assert app.state.ml_engine_client._client is http_client
        assert factory_calls == [settings]
        http_client.aclose.assert_not_awaited()

    assert app.state.ml_engine_client is None
    http_client.aclose.assert_awaited_once_with()


async def test_lifespan_cleans_up_after_exception(settings):
    http_client = AsyncMock(spec=httpx.AsyncClient)
    app = create_app(settings=settings, http_client_factory=lambda _settings: http_client)

    with pytest.raises(RuntimeError, match="test failure"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("test failure")

    assert app.state.ml_engine_client is None
    http_client.aclose.assert_awaited_once_with()


async def test_production_http_client_factory_wiring():
    from backend.app.config import Settings

    settings = Settings(
        _env_file=None,
        ml_engine_url="https://ml.example.test:8443/",
        ml_engine_connect_timeout_seconds=1.0,
        ml_engine_read_timeout_seconds=2.0,
        ml_engine_write_timeout_seconds=3.0,
        ml_engine_pool_timeout_seconds=4.0,
    )
    client = _create_http_client(settings)
    try:
        assert str(client.base_url) == "https://ml.example.test:8443"
        assert client.timeout.connect == 1.0
        assert client.timeout.read == 2.0
        assert client.timeout.write == 3.0
        assert client.timeout.pool == 4.0
        assert client._trust_env is False
    finally:
        await client.aclose()


def test_module_import_does_not_create_settings_or_http_client(monkeypatch):
    import importlib
    import backend.app.config as config_module
    import backend.app.main as main_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("import created a resource")

    original_settings = config_module.Settings
    original_async_client = httpx.AsyncClient
    try:
        monkeypatch.setattr(config_module, "Settings", forbidden)
        monkeypatch.setattr(httpx, "AsyncClient", forbidden)
        importlib.reload(main_module)
    finally:
        monkeypatch.setattr(config_module, "Settings", original_settings)
        monkeypatch.setattr(httpx, "AsyncClient", original_async_client)
        importlib.reload(main_module)
