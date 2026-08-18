"""Registration endpoint tests — 17 cases covering the full endpoint contract."""
import json

import cv2
import numpy as np
import pytest
from httpx import AsyncClient

from registration.result import RegistrationResult, RegistrationStatus, SampleResult


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
async def test_register_success(async_client: AsyncClient):
    """Happy path: status is serialised as the enum string value."""
    jpeg = _make_jpeg()
    files = [("images", ("face1.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 200
    res = response.json()
    assert res["person_id"] == "EMP-001"
    # Proves RegistrationStatus enum serialises to its string value
    assert res["status"] == "success"
    assert res["accepted_count"] == 1
    assert res["rejected_count"] == 0
    assert isinstance(res["sample_results"], list)
    assert "model_name" in res
    assert "embedding_dimension" in res
    assert "timestamp" in res


@pytest.mark.asyncio
async def test_register_rejected(async_client: AsyncClient, app_with_fake_runtime):
    runtime = app_with_fake_runtime.state.ml_runtime
    runtime.registration_service.register = lambda **kw: RegistrationResult(
        person_id=kw["person_id"],
        status=RegistrationStatus.REJECTED,
        accepted_count=0,
        rejected_count=1,
        rejection_reasons=["Face too small"],
        sample_results=[
            SampleResult(accepted=False, reason="Face too small", sample_index=0)
        ],
    )
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-002", "display_name": "Bob"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "rejected"
    assert res["rejected_count"] == 1
    assert "Face too small" in res["rejection_reasons"]


@pytest.mark.asyncio
async def test_register_duplicate(async_client: AsyncClient, app_with_fake_runtime):
    runtime = app_with_fake_runtime.state.ml_runtime
    runtime.registration_service.register = lambda **kw: RegistrationResult(
        person_id=kw["person_id"],
        status=RegistrationStatus.DUPLICATE,
        accepted_count=0,
        rejected_count=1,
        rejection_reasons=["Duplicate identity"],
        duplicate_person_id="EMP-099",
        duplicate_similarity=0.95,
    )
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-003", "display_name": "Charlie"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "duplicate"
    assert res["duplicate_person_id"] == "EMP-099"
    assert res["duplicate_similarity"] == 0.95


# ── Input validation ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_missing_images(async_client: AsyncClient):
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_too_many_images(async_client: AsyncClient):
    """Default maximum_samples is 10; sending 11 triggers 422."""
    jpeg = _make_jpeg()
    files = [("images", (f"face{i}.jpg", jpeg, "image/jpeg")) for i in range(11)]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_register_unsupported_mime(async_client: AsyncClient):
    files = [("images", ("doc.txt", b"hello world", "text/plain"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE_TYPE"


@pytest.mark.asyncio
async def test_register_empty_file(async_client: AsyncClient):
    files = [("images", ("empty.jpg", b"", "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.asyncio
async def test_register_invalid_image_bytes(async_client: AsyncClient):
    files = [("images", ("bad.jpg", b"\xff\xd8\xff\x00garbage", "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.asyncio
async def test_register_oversized_image(async_client: AsyncClient, monkeypatch):
    """Use monkeypatch to set a tiny limit instead of allocating 10 MiB."""
    monkeypatch.setenv("TRACIFY_ML_API_MAX_IMAGE_BYTES", "100")
    jpeg = _make_jpeg()
    assert len(jpeg) > 100, "Test JPEG must exceed the 100-byte limit"
    files = [("images", ("big.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


# ── Metadata validation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_invalid_metadata_json(async_client: AsyncClient):
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice", "metadata": "not-json{"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_METADATA"


@pytest.mark.asyncio
async def test_register_metadata_not_object(async_client: AsyncClient):
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice", "metadata": "[1, 2, 3]"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_METADATA"


@pytest.mark.asyncio
async def test_register_metadata_invalid_types(async_client: AsyncClient):
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {
        "person_id": "EMP-001",
        "display_name": "Alice",
        "metadata": json.dumps({"department": 123}),
    }
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_METADATA"


@pytest.mark.parametrize(
    "label, metadata_dict",
    [
        ("too_many_keys", {f"k{i}": "v" for i in range(51)}),
        ("key_too_long", {"k" * 101: "val"}),
        ("value_too_long", {"ok_key": "v" * 1001}),
    ],
    ids=["too_many_keys", "key_too_long", "value_too_long"],
)
@pytest.mark.asyncio
async def test_register_metadata_limits(async_client: AsyncClient, label, metadata_dict):
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {
        "person_id": "EMP-001",
        "display_name": "Alice",
        "metadata": json.dumps(metadata_dict),
    }
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 400, f"Expected 400 for {label}, got {response.status_code}"
    assert response.json()["error"]["code"] == "INVALID_METADATA"


# ── Runtime / processing errors ────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_unavailable_runtime(async_client_unavailable: AsyncClient):
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client_unavailable.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ML_ENGINE_NOT_READY"


@pytest.mark.asyncio
async def test_register_ml_processing_failure(
    async_client: AsyncClient, app_with_fake_runtime
):
    """service.register raises — endpoint returns sanitized 500."""
    runtime = app_with_fake_runtime.state.ml_runtime

    def failing_register(**kwargs):
        raise RuntimeError("GPU out of memory")

    runtime.registration_service.register = failing_register
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 500
    res = response.json()
    assert res["error"]["code"] == "ML_PROCESSING_ERROR"
    # Raw exception must not leak
    assert "GPU" not in res["error"]["message"]


# ── Contract enforcement ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_service_called_once(
    async_client: AsyncClient, app_with_fake_runtime
):
    """RegistrationService.register() must be called exactly once."""
    runtime = app_with_fake_runtime.state.ml_runtime
    original = runtime.registration_service.register
    call_count = {"n": 0}

    def counting_register(**kwargs):
        call_count["n"] += 1
        return original(**kwargs)

    runtime.registration_service.register = counting_register
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 200
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_register_no_embeddings_in_response(async_client: AsyncClient):
    """No raw embedding vectors or candidate lists in any response field."""
    jpeg = _make_jpeg()
    files = [("images", ("face.jpg", jpeg, "image/jpeg"))]
    data = {"person_id": "EMP-001", "display_name": "Alice"}
    response = await async_client.post(
        "/internal/v1/faces/register", data=data, files=files
    )
    assert response.status_code == 200
    _assert_no_embedding_leak(response.json())
