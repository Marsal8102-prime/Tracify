"""Tests for database lifespan resource management."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.app.config import Settings
from backend.app.main import create_app


def _make_app(
    settings,
    *,
    http_client=None,
    engine=None,
    session_factory_result=None,
    health_factory_result=None,
    http_client_factory=None,
    engine_factory=None,
    session_factory_builder=None,
    health_factory_builder=None,
):
    """Build an app with injectable fakes for lifecycle tests."""
    _http_client = http_client or AsyncMock(spec=httpx.AsyncClient)
    _engine = engine or MagicMock()
    _engine.dispose = AsyncMock()
    _session_factory = session_factory_result or MagicMock()
    _health_checker = health_factory_result or MagicMock()

    _http_factory = http_client_factory or (lambda _s: _http_client)
    _eng_factory = engine_factory or (lambda _s: _engine)
    _ses_factory = session_factory_builder or (lambda _e: _session_factory)
    _hlt_factory = health_factory_builder or (lambda _e, _s: _health_checker)

    app = create_app(
        settings=settings,
        http_client_factory=_http_factory,
        database_engine_factory=_eng_factory,
        database_session_factory=_ses_factory,
        database_health_factory=_hlt_factory,
    )
    return app, _http_client, _engine, _session_factory, _health_checker


async def test_lifespan_creates_and_disposes_all_resources(settings):
    app, http_client, engine, session_factory, health_checker = _make_app(settings)

    async with app.router.lifespan_context(app):
        assert app.state.ml_engine_client is not None
        assert app.state.database_engine is engine
        assert app.state.database_session_factory is session_factory
        assert app.state.database_health_checker is health_checker
        http_client.aclose.assert_not_awaited()
        engine.dispose.assert_not_awaited()

    assert app.state.ml_engine_client is None
    assert app.state.database_engine is None
    assert app.state.database_session_factory is None
    assert app.state.database_health_checker is None
    http_client.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_lifespan_cleans_up_after_exception(settings):
    app, http_client, engine, _, _ = _make_app(settings)

    with pytest.raises(RuntimeError, match="boom"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("boom")

    assert app.state.ml_engine_client is None
    assert app.state.database_engine is None
    assert app.state.database_session_factory is None
    assert app.state.database_health_checker is None
    http_client.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_database_factory_failure_closes_http_client(settings):
    http_client = AsyncMock(spec=httpx.AsyncClient)

    def failing_engine_factory(_settings):
        raise RuntimeError("engine creation failed")

    app = create_app(
        settings=settings,
        http_client_factory=lambda _s: http_client,
        database_engine_factory=failing_engine_factory,
    )

    with pytest.raises(RuntimeError, match="engine creation failed"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover

    http_client.aclose.assert_awaited_once()
    assert app.state.ml_engine_client is None
    assert app.state.database_engine is None


async def test_session_factory_failure_disposes_engine_and_closes_http(settings):
    http_client = AsyncMock(spec=httpx.AsyncClient)
    engine = MagicMock()
    engine.dispose = AsyncMock()

    def failing_session_factory(_engine):
        raise RuntimeError("session factory failed")

    app = create_app(
        settings=settings,
        http_client_factory=lambda _s: http_client,
        database_engine_factory=lambda _s: engine,
        database_session_factory=failing_session_factory,
    )

    with pytest.raises(RuntimeError, match="session factory failed"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover

    http_client.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_health_factory_failure_disposes_engine_and_closes_http(settings):
    http_client = AsyncMock(spec=httpx.AsyncClient)
    engine = MagicMock()
    engine.dispose = AsyncMock()

    def failing_health_factory(_engine, _settings):
        raise RuntimeError("health factory failed")

    app = create_app(
        settings=settings,
        http_client_factory=lambda _s: http_client,
        database_engine_factory=lambda _s: engine,
        database_session_factory=lambda _e: MagicMock(),
        database_health_factory=failing_health_factory,
    )

    with pytest.raises(RuntimeError, match="health factory failed"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover

    http_client.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_http_factory_failure_creates_no_database_resource(settings):
    engine_calls = []

    def failing_http_factory(_settings):
        raise RuntimeError("http factory failed")

    def tracking_engine_factory(_settings):
        engine_calls.append(1)
        return MagicMock()  # pragma: no cover

    app = create_app(
        settings=settings,
        http_client_factory=failing_http_factory,
        database_engine_factory=tracking_engine_factory,
    )

    with pytest.raises(RuntimeError, match="http factory failed"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover

    assert engine_calls == []
    assert app.state.database_engine is None


async def test_state_absent_before_lifespan(settings):
    app, _, _, _, _ = _make_app(settings)
    # Before lifespan, state attributes shouldn't be set to resources
    assert not hasattr(app.state, "ml_engine_client") or app.state.ml_engine_client is None
