"""Request-ID middleware tests — 5 cases covering propagation, replacement,
generation, and presence in both headers and error bodies."""
import re

import pytest
from httpx import AsyncClient

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


@pytest.mark.asyncio
async def test_valid_request_id_propagated(async_client: AsyncClient):
    """A valid X-Request-ID is echoed back unchanged."""
    response = await async_client.get(
        "/internal/v1/health",
        headers={"X-Request-ID": "my-valid-id.123"},
    )
    assert response.headers["X-Request-ID"] == "my-valid-id.123"


@pytest.mark.asyncio
async def test_invalid_request_id_replaced(async_client: AsyncClient):
    """An untrusted ID (special chars, spaces) is replaced with a UUID."""
    response = await async_client.get(
        "/internal/v1/health",
        headers={"X-Request-ID": "bad id with spaces!@#"},
    )
    returned_id = response.headers["X-Request-ID"]
    assert returned_id != "bad id with spaces!@#"
    assert UUID_RE.match(returned_id), f"Expected UUID, got {returned_id!r}"


@pytest.mark.asyncio
async def test_missing_request_id_generates_uuid(async_client: AsyncClient):
    """No X-Request-ID header → middleware generates a valid UUID."""
    response = await async_client.get("/internal/v1/health")
    returned_id = response.headers["X-Request-ID"]
    assert UUID_RE.match(returned_id), f"Expected UUID, got {returned_id!r}"


@pytest.mark.asyncio
async def test_request_id_in_success_and_error(
    async_client: AsyncClient,
    async_client_unavailable: AsyncClient,
):
    """Both success and error responses carry the same X-Request-ID."""
    # Success (200)
    success = await async_client.get(
        "/internal/v1/health",
        headers={"X-Request-ID": "success-abc"},
    )
    assert success.headers["X-Request-ID"] == "success-abc"

    # Error (503 from unavailable runtime)
    error = await async_client_unavailable.post(
        "/internal/v1/faces/recognize",
        files={"image": ("f.jpg", b"\xff", "image/jpeg")},
        headers={"X-Request-ID": "error-xyz"},
    )
    assert error.headers["X-Request-ID"] == "error-xyz"


@pytest.mark.asyncio
async def test_request_id_in_error_body(
    async_client_unavailable: AsyncClient,
):
    """Structured error body includes request_id matching the header."""
    response = await async_client_unavailable.post(
        "/internal/v1/faces/recognize",
        files={"image": ("f.jpg", b"\xff", "image/jpeg")},
        headers={"X-Request-ID": "body-789"},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["request_id"] == "body-789"
    assert response.headers["X-Request-ID"] == "body-789"
