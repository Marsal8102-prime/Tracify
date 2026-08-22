import pytest
from httpx import Response
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.base import Base

from backend.tests.conftest import request_app

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from backend.app.dependencies import get_database_session

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
def app_with_db(app, db_engine):
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async def override_get_session():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
    app.dependency_overrides[get_database_session] = override_get_session
    return app

@pytest.fixture
async def sample_person(app_with_db) -> dict:
    payload = {
        "person_id": "EMP123",
        "display_name": "Test Employee",
        "department": "Engineering"
    }
    resp = await request_app(app_with_db, "POST", "/api/persons", json=payload)
    if resp.status_code == 409:
        resp = await request_app(app_with_db, "GET", "/api/persons/EMP123")
    assert resp.status_code in (200, 201)
    return resp.json()

async def test_create_person_success(app_with_db):
    payload = {
        "person_id": "NEW456",
        "display_name": "New Employee",
        "department": "HR"
    }
    response = await request_app(app_with_db, "POST", "/api/persons", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["person_id"] == "NEW456"
    assert data["display_name"] == "New Employee"
    assert data["department"] == "HR"
    assert data["status"] == "active"
    assert "id" in data
    assert "created_at" in data

async def test_create_person_duplicate(app_with_db, sample_person):
    payload = {
        "person_id": sample_person["person_id"], # Duplicate
        "display_name": "Duplicate Employee",
    }
    response = await request_app(app_with_db, "POST", "/api/persons", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONFLICT"

async def test_create_person_invalid_data(app_with_db):
    payload = {
        "person_id": "", # Invalid (min_length=1)
        "display_name": "New Employee"
    }
    response = await request_app(app_with_db, "POST", "/api/persons", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"

async def test_get_person_success(app_with_db, sample_person):
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db, "GET", f"/api/persons/{person_id}")
    assert response.status_code == 200
    assert response.json()["person_id"] == person_id

async def test_get_person_not_found(app_with_db):
    response = await request_app(app_with_db, "GET", "/api/persons/MISSING999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"

async def test_list_persons(app_with_db, sample_person):
    response = await request_app(app_with_db, "GET", "/api/persons")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any(p["person_id"] == sample_person["person_id"] for p in data)

async def test_update_person_success(app_with_db, sample_person):
    person_id = sample_person["person_id"]
    payload = {
        "display_name": "Updated Name",
        "status": "archived"
    }
    response = await request_app(app_with_db, "PATCH", f"/api/persons/{person_id}", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "Updated Name"
    assert data["status"] == "archived"
