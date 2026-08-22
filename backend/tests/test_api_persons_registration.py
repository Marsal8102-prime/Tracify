import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from backend.app.models.base import Base

from backend.tests.conftest import request_app
from backend.app.dependencies import get_database_session, get_ml_engine_client
from backend.app.schemas import MLRegistrationResponse
from backend.app.errors import MLEngineDownstreamError, MLEngineTimeoutError

pytestmark = pytest.mark.asyncio

class FakeMLEngineClient:
    def __init__(self):
        self.should_timeout = False
        self.downstream_error_code = None
        
    async def register_faces(
        self,
        *,
        person_id: str,
        display_name: str,
        images: list,
        metadata: dict | None = None,
        request_id: str,
    ) -> MLRegistrationResponse:
        if self.should_timeout:
            raise MLEngineTimeoutError("timeout")
        if self.downstream_error_code:
            raise MLEngineDownstreamError(400, self.downstream_error_code)
            
        return MLRegistrationResponse(
            person_id=person_id,
            status="success",
            accepted_count=len(images),
            rejected_count=0,
            rejection_reasons=[],
            sample_results=[],
            model_name="test_model",
            embedding_dimension=128,
            timestamp="2023-01-01T00:00:00Z"
        )

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
def fake_ml_client():
    return FakeMLEngineClient()

@pytest.fixture
def app_with_db_and_ml(app, db_engine, fake_ml_client):
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async def override_get_session():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
                
    app.dependency_overrides[get_database_session] = override_get_session
    app.dependency_overrides[get_ml_engine_client] = lambda: fake_ml_client
    return app

@pytest.fixture
async def sample_person(app_with_db_and_ml) -> dict:
    payload = {
        "person_id": "EMP123",
        "display_name": "Test Employee",
        "department": "Engineering"
    }
    resp = await request_app(app_with_db_and_ml, "POST", "/api/persons", json=payload)
    if resp.status_code == 409:
        resp = await request_app(app_with_db_and_ml, "GET", "/api/persons/EMP123")
    assert resp.status_code in (200, 201)
    return resp.json()

async def test_register_face_success(app_with_db_and_ml, sample_person):
    files = [("images", ("face.jpg", b"fake_image_data", "image/jpeg"))]
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db_and_ml, "POST", f"/api/persons/{person_id}/register", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["person_id"] == person_id
    assert data["status"] == "success"
    assert data["accepted_count"] == 1

async def test_register_face_person_not_found(app_with_db_and_ml):
    files = [("images", ("face.jpg", b"fake_image_data", "image/jpeg"))]
    response = await request_app(app_with_db_and_ml, "POST", "/api/persons/MISSING/register", files=files)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"

async def test_register_face_ml_timeout(app_with_db_and_ml, sample_person, fake_ml_client):
    fake_ml_client.should_timeout = True
    files = [("images", ("face.jpg", b"fake_image_data", "image/jpeg"))]
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db_and_ml, "POST", f"/api/persons/{person_id}/register", files=files)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

async def test_register_face_ml_downstream_error(app_with_db_and_ml, sample_person, fake_ml_client):
    fake_ml_client.downstream_error_code = "INVALID_IMAGE"
    files = [("images", ("face.jpg", b"fake_image_data", "image/jpeg"))]
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db_and_ml, "POST", f"/api/persons/{person_id}/register", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ML_DOWNSTREAM_ERROR"
    assert "INVALID_IMAGE" in response.json()["error"]["message"]

async def test_register_face_malformed_upload_type(app_with_db_and_ml, sample_person):
    files = [("images", ("doc.txt", b"text", "text/plain"))]
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db_and_ml, "POST", f"/api/persons/{person_id}/register", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "Unsupported image type" in response.json()["error"]["message"]

async def test_register_face_malformed_upload_too_many(app_with_db_and_ml, sample_person):
    files = [("images", ("face1.jpg", b"img", "image/jpeg")) for _ in range(6)]
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db_and_ml, "POST", f"/api/persons/{person_id}/register", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "Too many images" in response.json()["error"]["message"]

async def test_register_face_malformed_upload_too_large(app_with_db_and_ml, sample_person):
    large_image = b"0" * (10 * 1024 * 1024 + 1)
    files = [("images", ("huge.jpg", large_image, "image/jpeg"))]
    person_id = sample_person["person_id"]
    response = await request_app(app_with_db_and_ml, "POST", f"/api/persons/{person_id}/register", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "exceeds the 10MB limit" in response.json()["error"]["message"]
