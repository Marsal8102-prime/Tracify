import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_ok(async_client: AsyncClient):
    response = await async_client.get("/internal/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["models_loaded"] is True
    assert data["gallery_loaded"] is True
    assert data["gallery_size"] == 0

@pytest.mark.asyncio
async def test_health_unavailable(async_client_unavailable: AsyncClient):
    response = await async_client_unavailable.get("/internal/v1/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["models_loaded"] is False
    assert data["gallery_loaded"] is False
    assert "error" not in data # Ensure internal errors are not leaked
