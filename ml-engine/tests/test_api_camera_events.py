import pytest
from fastapi.testclient import TestClient
from typing import AsyncGenerator
import json
import asyncio

from api.main import create_app
from api.dependencies import get_runtime
from tests.fakes import create_fake_runtime, FakeCamera
from camera.session import CameraSession
from camera.events import CameraEvent

@pytest.fixture
def fake_runtime():
    return create_fake_runtime(ready=True)

@pytest.fixture
def client(fake_runtime) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_runtime] = lambda: fake_runtime

    # We must explicitly set app.state.ml_runtime because require_ready_runtime depends on it
    app.state.ml_runtime = fake_runtime

    with TestClient(app) as test_client:
        yield test_client

def test_camera_events_empty_when_no_session(client):
    response = client.get("/internal/v1/camera/events")
    assert response.status_code == 200
    assert response.json() == []

def test_camera_events_returns_events_and_drains(client, fake_runtime):
    # Setup session with some fake events
    camera = FakeCamera()
    session = CameraSession(
        camera=camera,
        preprocessor=fake_runtime.preprocessor,
        detector=fake_runtime.detector,
        aligner=fake_runtime.aligner,
        embedder=fake_runtime.embedder,
        recognizer=fake_runtime.recognizer,
        pipeline_lock=fake_runtime.pipeline_lock,
    )

    # Inject some events manually
    for i in range(5):
        session._event_buffer.append(
            CameraEvent(
                person_id=f"person_{i}",
                recognition_status="known",
                similarity=0.9,
                threshold=0.6,
                detection_confidence=0.99,
                bbox=[0, 0, 100, 100],
                matched_embedding_id=None,
                timestamp="2023-01-01T00:00:00Z"
            )
        )

    fake_runtime.camera_session = session

    # Fetch events
    response = client.get("/internal/v1/camera/events")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert data[0]["person_id"] == "person_0"

    # Verify drain
    response2 = client.get("/internal/v1/camera/events")
    assert response2.status_code == 200
    assert response2.json() == []

def test_camera_events_with_limit(client, fake_runtime):
    camera = FakeCamera()
    session = CameraSession(
        camera=camera,
        preprocessor=fake_runtime.preprocessor,
        detector=fake_runtime.detector,
        aligner=fake_runtime.aligner,
        embedder=fake_runtime.embedder,
        recognizer=fake_runtime.recognizer,
        pipeline_lock=fake_runtime.pipeline_lock,
    )
    for i in range(10):
        session._event_buffer.append(
            CameraEvent(
                person_id=f"person_{i}",
                recognition_status="known",
                similarity=0.9,
                threshold=0.6,
                detection_confidence=0.99,
                bbox=[0, 0, 100, 100],
                matched_embedding_id=None,
                timestamp="2023-01-01T00:00:00Z"
            )
        )
    fake_runtime.camera_session = session

    response = client.get("/internal/v1/camera/events?limit=3")
    assert response.status_code == 200
    assert len(response.json()) == 3

    assert session.events_buffered == 7
