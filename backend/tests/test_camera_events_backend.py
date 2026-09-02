import pytest
from httpx import Response
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.exc import OperationalError

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.services.recognition import RecognitionService
from backend.app.schemas import MLCameraEvent
from backend.app.models.recognition_event import RecognitionEvent
from backend.app.models.person import Person
from backend.app.errors import DatabaseUnavailableError


@pytest.fixture
def ml_client():
    mock_httpx = AsyncMock()
    return MLEngineClient(mock_httpx, health_timeout_seconds=2.0)


@pytest.mark.asyncio
async def test_fetch_camera_events_success(ml_client):
    events_payload = [
        {
            "person_id": "test_person",
            "recognition_status": "known",
            "similarity": 0.95,
            "threshold": 0.60,
            "detection_confidence": 0.99,
            "bbox": [10, 10, 100, 100],
            "matched_embedding_id": "emb123",
            "timestamp": "2023-01-01T00:00:00Z",
        }
    ]
    ml_client._client.request.return_value = Response(200, json=events_payload)

    events = await ml_client.fetch_camera_events(request_id="req-123", limit=50)

    assert len(events) == 1
    assert isinstance(events[0], MLCameraEvent)
    assert events[0].person_id == "test_person"

    # Verify request path and limit
    ml_client._client.request.assert_called_once()
    args, kwargs = ml_client._client.request.call_args
    assert args[0] == "GET"
    assert args[1] == "/internal/v1/camera/events?limit=50"


@pytest.mark.asyncio
async def test_persist_camera_events_empty():
    mock_session = AsyncMock()
    mock_session.add_all = MagicMock()
    service = RecognitionService(session=mock_session, ml_client=AsyncMock())

    await service.persist_camera_events([], "req-123")

    # Should return early without hitting DB
    mock_session.execute.assert_not_called()
    mock_session.add_all.assert_not_called()


@pytest.mark.asyncio
async def test_persist_camera_events_normalizes_orphans():
    mock_session = AsyncMock()
    mock_session.add_all = MagicMock()
    # Mock DB query to return empty set (person doesn't exist)
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = []
    mock_session.execute.return_value = mock_result

    service = RecognitionService(session=mock_session, ml_client=AsyncMock())

    events = [
        MLCameraEvent(
            person_id="nonexistent_person",
            recognition_status="known",
            similarity=0.9,
            threshold=0.6,
            detection_confidence=0.9,
            bbox=[0, 0, 100, 100],
            matched_embedding_id="emb1",
            timestamp="2023-01-01T00:00:00Z"
        )
    ]

    await service.persist_camera_events(events, "req-123")

    mock_session.add_all.assert_called_once()
    inserted_events = mock_session.add_all.call_args[0][0]
    assert len(inserted_events) == 1

    # Verify it was normalized to unknown since it wasn't in DB
    event = inserted_events[0]
    assert isinstance(event, RecognitionEvent)
    assert event.person_id is None
    assert event.recognition_status == "unknown"


@pytest.mark.asyncio
async def test_persist_camera_events_known():
    mock_session = AsyncMock()
    mock_session.add_all = MagicMock()
    # Mock DB query to return the person
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = ["existing_person"]
    mock_session.execute.return_value = mock_result

    service = RecognitionService(session=mock_session, ml_client=AsyncMock())

    events = [
        MLCameraEvent(
            person_id="existing_person",
            recognition_status="known",
            similarity=0.9,
            threshold=0.6,
            detection_confidence=0.9,
            bbox=[0, 0, 100, 100],
            matched_embedding_id="emb1",
            timestamp="2023-01-01T00:00:00Z"
        )
    ]

    await service.persist_camera_events(events, "req-123")

    mock_session.add_all.assert_called_once()
    inserted_events = mock_session.add_all.call_args[0][0]
    assert len(inserted_events) == 1

    # Verify it remains known
    event = inserted_events[0]
    assert isinstance(event, RecognitionEvent)
    assert event.person_id == "existing_person"
    assert event.recognition_status == "known"
