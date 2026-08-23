import pytest
from unittest.mock import AsyncMock
from fastapi import UploadFile

from backend.tests.conftest import request_app
from backend.app.models.base import Base
from backend.app.models.person import Person
from backend.app.schemas import MLRecognitionResponse, MLFaceResult
from backend.app.dependencies import get_database_session, get_ml_engine_client

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

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
def mock_ml_client():
    mock = AsyncMock()
    return mock


@pytest.fixture
def app_with_db_and_ml(app, db_engine, mock_ml_client):
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    
    async def override_get_session():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
                
    def override_get_ml_client():
        return mock_ml_client
        
    app.dependency_overrides[get_database_session] = override_get_session
    app.dependency_overrides[get_ml_engine_client] = override_get_ml_client
    return app


@pytest.fixture
async def setup_person(db_engine) -> str:
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        p = Person(person_id="emp_001", display_name="Test Person")
        session.add(p)
        await session.commit()
    return "emp_001"


async def test_recognize_missing_image(app_with_db_and_ml):
    response = await request_app(app_with_db_and_ml, "POST", "/api/recognition/recognize")
    assert response.status_code == 422


async def test_recognize_invalid_image_type(app_with_db_and_ml):
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    response = await request_app(app_with_db_and_ml, "POST", "/api/recognition/recognize", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_recognize_no_faces(app_with_db_and_ml, mock_ml_client):
    mock_ml_client.recognize_face.return_value = MLRecognitionResponse(
        face_count=0,
        processing_time_ms=10.0,
        faces=[],
    )
    files = {"image": ("test.jpg", b"fake_image_content", "image/jpeg")}
    response = await request_app(app_with_db_and_ml, "POST", "/api/recognition/recognize", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["face_count"] == 0
    assert len(data["faces"]) == 0


async def test_recognize_known_person(app_with_db_and_ml, mock_ml_client, setup_person):
    mock_ml_client.recognize_face.return_value = MLRecognitionResponse(
        face_count=1,
        processing_time_ms=50.0,
        faces=[
            MLFaceResult(
                person_id="emp_001",
                recognition_status="known",
                similarity=0.9,
                threshold=0.6,
                detection_confidence=0.95,
                bbox=[10, 10, 100, 100],
                matched_embedding_id="emb_1",
            )
        ],
    )
    files = {"image": ("test.jpg", b"fake_image_content", "image/jpeg")}
    response = await request_app(app_with_db_and_ml, "POST", "/api/recognition/recognize", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["faces"][0]["person_id"] == "emp_001"
    assert data["faces"][0]["recognition_status"] == "known"


async def test_recognize_orphaned_person(app_with_db_and_ml, mock_ml_client):
    mock_ml_client.recognize_face.return_value = MLRecognitionResponse(
        face_count=1,
        processing_time_ms=50.0,
        faces=[
            MLFaceResult(
                person_id="emp_999",
                recognition_status="known",
                similarity=0.9,
                threshold=0.6,
                detection_confidence=0.95,
                bbox=[10, 10, 100, 100],
                matched_embedding_id="emb_999",
            )
        ],
    )
    files = {"image": ("test.jpg", b"fake_image_content", "image/jpeg")}
    response = await request_app(app_with_db_and_ml, "POST", "/api/recognition/recognize", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["faces"][0]["person_id"] == "emp_999"
    assert data["faces"][0]["recognition_status"] == "known"
