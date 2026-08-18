import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from api.main import create_app
from tests.fakes import create_fake_runtime

@pytest.fixture
def app_with_fake_runtime():
    app = create_app(runtime_factory=create_fake_runtime)
    app.state.ml_runtime = create_fake_runtime(ready=True)
    return app

@pytest_asyncio.fixture
async def async_client(app_with_fake_runtime):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_fake_runtime),
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture
def app_unavailable():
    app = create_app(runtime_factory=create_fake_runtime)
    app.state.ml_runtime = create_fake_runtime(ready=False)
    return app

@pytest_asyncio.fixture
async def async_client_unavailable(app_unavailable):
    async with AsyncClient(
        transport=ASGITransport(app=app_unavailable),
        base_url="http://test"
    ) as client:
        yield client
