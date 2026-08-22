import httpx
import pytest

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.config import Settings
from backend.app.dependencies import get_ml_engine_client
from backend.app.errors import (
    MLEngineConnectionError,
    MLEngineProtocolError,
    MLEngineTimeoutError,
    MLEngineUnavailableError,
)
from backend.app.main import create_app
from backend.tests.conftest import healthy_payload, request_app


class StubMLClient:
    def __init__(self, failure=None):
        self.calls = 0
        self.request_ids = []
        self.failure = failure

    async def health(self, *, request_id: str):
        self.calls += 1
        self.request_ids.append(request_id)
        if self.failure is not None:
            raise self.failure
        return healthy_payload()


class StubDatabaseHealth:
    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure

    async def check(self):
        self.calls += 1
        if self.failure is not None:
            raise self.failure


def app_with_stubs(settings: Settings, ml_stub: StubMLClient = None, db_stub: StubDatabaseHealth = None):
    app = create_app(settings=settings)
    if ml_stub:
        app.dependency_overrides[get_ml_engine_client] = lambda: ml_stub
    if db_stub:
        from backend.app.dependencies import get_database_health_checker
        app.dependency_overrides[get_database_health_checker] = lambda: db_stub
    return app


async def request_without_lifespan(app, path: str) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        return await client.get(path, headers={"X-Request-ID": "state-test"})


def resolve_schema(document, item):
    while "$ref" in item:
        resolved = document
        for part in item["$ref"].lstrip("#/").split("/"):
            resolved = resolved[part]
        item = resolved
    return item


async def test_liveness_contract_and_no_ml_resolution(settings):
    app = create_app(settings=settings)

    def forbidden_dependency():
        raise AssertionError("Liveness resolved a dependency")

    app.dependency_overrides[get_ml_engine_client] = forbidden_dependency
    from backend.app.dependencies import get_database_health_checker
    app.dependency_overrides[get_database_health_checker] = forbidden_dependency
    response = await request_app(app, "GET", "/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


async def test_liveness_is_200_without_lifespan_or_ml_client_state(settings):
    app = create_app(settings=settings)
    assert not hasattr(app.state, "ml_engine_client")
    assert not hasattr(app.state, "database_health_checker")
    response = await request_without_lifespan(app, "/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


async def test_readiness_without_state_is_sanitized_503(settings):
    app = create_app(settings=settings)
    assert not hasattr(app.state, "ml_engine_client")
    response = await request_without_lifespan(app, "/api/v1/health/ready")
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "SERVICE_UNAVAILABLE",
            "message": "A required service is unavailable.",
            "request_id": "state-test",
        }
    }


async def test_readiness_success_and_request_id_propagation(settings):
    ml_stub = StubMLClient()
    db_stub = StubDatabaseHealth()
    response = await request_app(
        app_with_stubs(settings, ml_stub, db_stub),
        "GET",
        "/api/v1/health/ready",
        headers={"X-Request-ID": "ready-123"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "version": "0.1.0"}
    assert ml_stub.calls == 1
    assert ml_stub.request_ids == ["ready-123"]
    assert db_stub.calls == 1


@pytest.mark.parametrize(
    "failure",
    [
        MLEngineConnectionError("hidden connection detail"),
        MLEngineTimeoutError("hidden timeout detail"),
        MLEngineProtocolError("hidden response detail"),
        MLEngineUnavailableError("hidden health detail"),
    ],
)
async def test_readiness_ml_known_failure_is_sanitized(settings, failure):
    ml_stub = StubMLClient(failure)
    db_stub = StubDatabaseHealth()
    response = await request_app(
        app_with_stubs(settings, ml_stub, db_stub),
        "GET",
        "/api/v1/health/ready",
        headers={"X-Request-ID": "backend-id"},
    )
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "SERVICE_UNAVAILABLE",
            "message": "A required service is unavailable.",
            "request_id": "backend-id",
        }
    }
    assert "hidden" not in response.text
    assert response.headers["X-Request-ID"] == "backend-id"
    # Sequential: if ML fails, DB is not checked
    assert db_stub.calls == 0


async def test_readiness_database_timeout_is_sanitized_503(settings):
    from backend.app.errors import DatabaseUnavailableError
    ml_stub = StubMLClient()
    db_stub = StubDatabaseHealth(DatabaseUnavailableError())
    response = await request_app(
        app_with_stubs(settings, ml_stub, db_stub),
        "GET",
        "/api/v1/health/ready",
        headers={"X-Request-ID": "backend-id"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert ml_stub.calls == 1
    assert db_stub.calls == 1


async def test_readiness_unexpected_programming_error_is_500(settings):
    ml_stub = StubMLClient(RuntimeError("secret programming detail"))
    db_stub = StubDatabaseHealth()
    response = await request_app(
        app_with_stubs(settings, ml_stub, db_stub),
        "GET",
        "/api/v1/health/ready",
        headers={"X-Request-ID": "backend-500"},
    )
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "INTERNAL_ERROR",
        "message": "An internal server error occurred.",
        "request_id": "backend-500",
    }
    assert "secret" not in response.text


async def test_readiness_uses_real_client_contract(settings):
    from backend.tests.conftest import PassingHealthCheck
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=healthy_payload()))
    app = create_app(
        settings=settings,
        http_client_factory=lambda config: httpx.AsyncClient(
            base_url=config.ml_engine_base_url,
            transport=transport,
        ),
        database_health_factory=lambda _e, _s: PassingHealthCheck(),
        database_engine_factory=lambda _s: __import__("unittest.mock").mock.AsyncMock(dispose=__import__("unittest.mock").mock.AsyncMock()),
        database_session_factory=lambda _e: None,  # no-op
    )
    response = await request_app(app, "GET", "/api/v1/health/ready")
    assert response.status_code == 200


async def test_health_openapi_documents_success_and_error_schemas(app):
    response = await request_app(app, "GET", "/openapi.json")
    schema = response.json()
    live = schema["paths"]["/api/v1/health/live"]["get"]["responses"]
    ready = schema["paths"]["/api/v1/health/ready"]["get"]["responses"]
    assert "200" in live
    assert {"200", "500", "503"}.issubset(ready)
    error_schema = resolve_schema(
        schema,
        ready["503"]["content"]["application/json"]["schema"],
    )
    assert set(error_schema["properties"]) == {"error"}
    detail_schema = resolve_schema(schema, error_schema["properties"]["error"])
    assert set(detail_schema["properties"]) == {"code", "message", "request_id"}
