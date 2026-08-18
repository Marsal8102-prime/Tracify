import re

import pytest

from backend.app.dependencies import get_ml_engine_client
from backend.app.errors import MLEngineUnavailableError
from backend.app.middleware import REQUEST_ID_REGEX, sanitize_request_id
from backend.tests.conftest import request_app


UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("abc-DEF_123.test", "abc-DEF_123.test"),
        ("  trimmed-id  ", "trimmed-id"),
        ("a" * 128, "a" * 128),
    ],
)
def test_sanitize_request_id_accepts_phase2_grammar(supplied, expected):
    assert sanitize_request_id(supplied) == expected


@pytest.mark.parametrize("supplied", [None, "", "   ", "bad id", "bad!", "é", "a" * 129])
def test_sanitize_request_id_replaces_invalid_values(supplied):
    value = sanitize_request_id(supplied)
    assert UUID_RE.fullmatch(value)
    assert REQUEST_ID_REGEX.fullmatch(value)


async def test_request_id_is_returned_by_middleware(app):
    response = await request_app(
        app,
        "GET",
        "/api/v1/health/live",
        headers={"X-Request-ID": " request.123 "},
    )
    assert response.headers["X-Request-ID"] == "request.123"


async def test_invalid_request_id_is_replaced_in_error_header_and_body(app):
    class UnavailableClient:
        async def health(self, *, request_id: str):
            raise MLEngineUnavailableError("hidden downstream detail")

    app.dependency_overrides[get_ml_engine_client] = UnavailableClient
    response = await request_app(
        app,
        "GET",
        "/api/v1/health/ready",
        headers={"X-Request-ID": "bad id!"},
    )
    assert response.status_code == 503
    returned_id = response.headers["X-Request-ID"]
    assert UUID_RE.fullmatch(returned_id)
    assert response.json()["error"]["request_id"] == returned_id
    assert "hidden" not in response.text
