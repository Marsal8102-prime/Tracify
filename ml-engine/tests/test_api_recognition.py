"""Recognition endpoint tests — 13 cases covering the full endpoint contract."""
import cv2
import numpy as np
import pytest
from httpx import AsyncClient

from detection.base import DetectionResult
from preprocessing.preprocessor import PreprocessedFrame
from recognition.result import RecognitionResult, RecognitionStatus


# ── Helpers ────────────────────────────────────────────────────────────


def _make_jpeg(width=100, height=100):
    """Create a minimal valid JPEG image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _assert_no_embedding_leak(data, path="root"):
    """Recursively verify no raw embedding vectors leak in response JSON.

    Safe contract fields (embedding_dimension, matched_embedding_id,
    model_name) are explicitly allowed.
    """
    FORBIDDEN = {"embedding", "embeddings", "candidates"}
    SAFE = {"embedding_dimension", "matched_embedding_id", "model_name"}
    if isinstance(data, dict):
        for key, value in data.items():
            if key in SAFE:
                continue
            assert key not in FORBIDDEN, f"Forbidden key '{key}' at {path}.{key}"
            _assert_no_embedding_leak(value, f"{path}.{key}")
    elif isinstance(data, list):
        if len(data) > 10 and all(isinstance(x, (int, float)) for x in data[:10]):
            raise AssertionError(
                f"Potential embedding array at {path} (length={len(data)})"
            )
        for i, item in enumerate(data):
            _assert_no_embedding_leak(item, f"{path}[{i}]")


# ── Success paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recognize_known_face(async_client: AsyncClient):
    jpeg = _make_jpeg()
    files = {"image": ("face.jpg", jpeg, "image/jpeg")}
    response = await async_client.post("/internal/v1/faces/recognize", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["face_count"] == 1
    assert res["faces"][0]["recognition_status"] == "known"
    assert res["faces"][0]["person_id"] == "EMP-001"
    assert res["faces"][0]["similarity"] == 0.9
    assert res["faces"][0]["threshold"] == 0.6
    assert "processing_time_ms" in res


@pytest.mark.asyncio
async def test_recognize_unknown_face(
    async_client: AsyncClient, app_with_fake_runtime
):
    runtime = app_with_fake_runtime.state.ml_runtime
    original = runtime.recognizer.recognize

    runtime.recognizer.recognize = lambda emb: RecognitionResult(
        status=RecognitionStatus.UNKNOWN,
        person_id=None,
        similarity=0.3,
        threshold=0.6,
    )
    try:
        jpeg = _make_jpeg()
        files = {"image": ("face.jpg", jpeg, "image/jpeg")}
        response = await async_client.post(
            "/internal/v1/faces/recognize", files=files
        )
        assert response.status_code == 200
        res = response.json()
        assert res["face_count"] == 1
        assert res["faces"][0]["recognition_status"] == "unknown"
        assert res["faces"][0]["person_id"] is None
    finally:
        runtime.recognizer.recognize = original


@pytest.mark.asyncio
async def test_recognize_multiple_faces(
    async_client: AsyncClient, app_with_fake_runtime
):
    runtime = app_with_fake_runtime.state.ml_runtime
    original = runtime.detector.detect

    runtime.detector.detect = lambda frame: [
        DetectionResult(
            bbox=np.array([0.0, 0.0, 50.0, 50.0]),
            landmarks=np.zeros((5, 2)),
            confidence=0.99,
        ),
        DetectionResult(
            bbox=np.array([50.0, 50.0, 100.0, 100.0]),
            landmarks=np.zeros((5, 2)),
            confidence=0.95,
        ),
    ]
    try:
        jpeg = _make_jpeg()
        files = {"image": ("face.jpg", jpeg, "image/jpeg")}
        response = await async_client.post(
            "/internal/v1/faces/recognize", files=files
        )
        assert response.status_code == 200
        res = response.json()
        assert res["face_count"] == 2
        assert len(res["faces"]) == 2
    finally:
        runtime.detector.detect = original


@pytest.mark.asyncio
async def test_recognize_no_faces(
    async_client: AsyncClient, app_with_fake_runtime
):
    runtime = app_with_fake_runtime.state.ml_runtime
    original = runtime.detector.detect
    runtime.detector.detect = lambda frame: []
    try:
        jpeg = _make_jpeg()
        files = {"image": ("face.jpg", jpeg, "image/jpeg")}
        response = await async_client.post(
            "/internal/v1/faces/recognize", files=files
        )
        assert response.status_code == 200
        res = response.json()
        assert res["face_count"] == 0
        assert res["faces"] == []
    finally:
        runtime.detector.detect = original


@pytest.mark.asyncio
async def test_recognize_alignment_failure(
    async_client: AsyncClient, app_with_fake_runtime
):
    """Aligner returns None for a detected face — face is skipped."""
    runtime = app_with_fake_runtime.state.ml_runtime
    original = runtime.aligner.align
    runtime.aligner.align = lambda frame, det: None
    try:
        jpeg = _make_jpeg()
        files = {"image": ("face.jpg", jpeg, "image/jpeg")}
        response = await async_client.post(
            "/internal/v1/faces/recognize", files=files
        )
        assert response.status_code == 200
        res = response.json()
        assert res["face_count"] == 0
    finally:
        runtime.aligner.align = original


@pytest.mark.asyncio
async def test_recognize_bbox_scaling(
    async_client: AsyncClient, app_with_fake_runtime
):
    """Detection coords on the preprocessed frame are scaled back to
    original image coordinates using the inverse of scale_factor."""
    runtime = app_with_fake_runtime.state.ml_runtime
    orig_preproc = runtime.preprocessor.process
    orig_detect = runtime.detector.detect

    # Preprocessor halved the image (scale_factor = 0.5)
    runtime.preprocessor.process = lambda frame: PreprocessedFrame(
        frame=frame,
        original_shape=(200, 200),
        scale_factor=0.5,
    )
    # Detection on the halved image returns [10, 10, 50, 50]
    runtime.detector.detect = lambda frame: [
        DetectionResult(
            bbox=np.array([10.0, 10.0, 50.0, 50.0]),
            landmarks=np.zeros((5, 2)),
            confidence=0.99,
        ),
    ]
    try:
        jpeg = _make_jpeg()
        files = {"image": ("face.jpg", jpeg, "image/jpeg")}
        response = await async_client.post(
            "/internal/v1/faces/recognize", files=files
        )
        assert response.status_code == 200
        bbox = response.json()["faces"][0]["bbox"]
        # coords / 0.5 → [20, 20, 100, 100]
        assert bbox == [20, 20, 100, 100]
    finally:
        runtime.preprocessor.process = orig_preproc
        runtime.detector.detect = orig_detect


# ── Input validation ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recognize_invalid_image_bytes(async_client: AsyncClient):
    files = {"image": ("bad.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg")}
    response = await async_client.post("/internal/v1/faces/recognize", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.asyncio
async def test_recognize_unsupported_mime(async_client: AsyncClient):
    files = {"image": ("doc.txt", b"not an image", "text/plain")}
    response = await async_client.post("/internal/v1/faces/recognize", files=files)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE_TYPE"


@pytest.mark.asyncio
async def test_recognize_empty_image(async_client: AsyncClient):
    files = {"image": ("empty.jpg", b"", "image/jpeg")}
    response = await async_client.post("/internal/v1/faces/recognize", files=files)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.asyncio
async def test_recognize_oversized_image(async_client: AsyncClient, monkeypatch):
    """Use monkeypatch to set a tiny limit instead of allocating 10 MiB."""
    monkeypatch.setenv("TRACIFY_ML_API_MAX_IMAGE_BYTES", "100")
    jpeg = _make_jpeg()
    assert len(jpeg) > 100, "Test JPEG must exceed the 100-byte limit"
    files = {"image": ("big.jpg", jpeg, "image/jpeg")}
    response = await async_client.post("/internal/v1/faces/recognize", files=files)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


# ── Runtime / processing errors ────────────────────────────────────────


@pytest.mark.asyncio
async def test_recognize_unavailable_runtime(
    async_client_unavailable: AsyncClient,
):
    jpeg = _make_jpeg()
    files = {"image": ("face.jpg", jpeg, "image/jpeg")}
    response = await async_client_unavailable.post(
        "/internal/v1/faces/recognize", files=files
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ML_ENGINE_NOT_READY"


@pytest.mark.asyncio
async def test_recognize_ml_processing_failure(
    async_client: AsyncClient, app_with_fake_runtime
):
    """Processing raises — endpoint returns sanitized 500."""
    runtime = app_with_fake_runtime.state.ml_runtime
    original = runtime.detector.detect

    def exploding_detect(frame):
        raise RuntimeError("GPU error")

    runtime.detector.detect = exploding_detect
    try:
        jpeg = _make_jpeg()
        files = {"image": ("face.jpg", jpeg, "image/jpeg")}
        response = await async_client.post(
            "/internal/v1/faces/recognize", files=files
        )
        assert response.status_code == 500
        res = response.json()
        assert res["error"]["code"] == "ML_PROCESSING_ERROR"
        assert "GPU" not in res["error"]["message"]
    finally:
        runtime.detector.detect = original


# ── Contract enforcement ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recognize_no_embeddings_in_response(async_client: AsyncClient):
    """No raw embedding vectors or candidate lists in any response field."""
    jpeg = _make_jpeg()
    files = {"image": ("face.jpg", jpeg, "image/jpeg")}
    response = await async_client.post("/internal/v1/faces/recognize", files=files)
    assert response.status_code == 200
    _assert_no_embedding_leak(response.json())
