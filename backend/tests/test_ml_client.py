from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest

from backend.app.clients.ml_engine import MLEngineClient, UploadPart
from backend.app.errors import (
    MLEngineConnectionError,
    MLEngineDownstreamError,
    MLEngineProtocolError,
    MLEngineTimeoutError,
    MLEngineUnavailableError,
)
from backend.tests.conftest import healthy_payload


KNOWN_ERROR_CODES = [
    "INVALID_IMAGE",
    "UNSUPPORTED_IMAGE_TYPE",
    "IMAGE_TOO_LARGE",
    "INVALID_METADATA",
    "ML_ENGINE_NOT_READY",
    "ML_PROCESSING_ERROR",
    "VALIDATION_ERROR",
    "INTERNAL_ERROR",
]


def multipart_parts(request: httpx.Request):
    content_type = request.headers["Content-Type"]
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + request.content
    )
    return list(message.iter_parts())


def part_value(part) -> bytes:
    return part.get_payload(decode=True)


def all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in all_keys(item)}
    return set()


def registration_payload() -> dict[str, object]:
    return {
        "person_id": "EMP-001",
        "status": "success",
        "accepted_count": 2,
        "rejected_count": 0,
        "rejection_reasons": [],
        "sample_results": [
            {"accepted": True, "reason": "accepted", "sample_index": 0},
            {"accepted": True, "reason": "accepted", "sample_index": 1},
        ],
        "duplicate_person_id": None,
        "duplicate_similarity": None,
        "model_name": "buffalo_l",
        "embedding_dimension": 512,
        "timestamp": "2026-08-18T00:00:00+00:00",
    }


def recognition_payload() -> dict[str, object]:
    return {
        "face_count": 1,
        "processing_time_ms": 12.5,
        "faces": [
            {
                "person_id": "EMP-001",
                "recognition_status": "known",
                "similarity": 0.9,
                "threshold": 0.6,
                "detection_confidence": 0.99,
                "bbox": [1, 2, 3, 4],
                "matched_embedding_id": "emb-1",
            }
        ],
    }


@asynccontextmanager
async def client_for(handler, *, timeout=None):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="http://ml.test",
        transport=transport,
        timeout=timeout or httpx.Timeout(30.0, connect=2.0, write=30.0, pool=2.0),
    ) as raw_client:
        yield MLEngineClient(raw_client, health_timeout_seconds=3.0)


async def test_health_exact_path_method_request_id_and_dedicated_timeout():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=healthy_payload())

    async with client_for(handler) as client:
        response = await client.health(request_id=" health-id ")

    assert response.status == "ok"
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/internal/v1/health"
    assert seen[0].headers["X-Request-ID"] == "health-id"
    assert set(seen[0].extensions["timeout"].values()) == {3.0}


@pytest.mark.parametrize(
    "payload,status_code",
    [
        ({**healthy_payload(), "status": "unavailable"}, 503),
        ({**healthy_payload(), "models_loaded": False}, 200),
        ({**healthy_payload(), "gallery_loaded": False}, 200),
    ],
)
async def test_health_valid_unhealthy_responses(payload, status_code):
    async with client_for(
        lambda _request: httpx.Response(status_code, json=payload)
    ) as client:
        with pytest.raises(MLEngineUnavailableError):
            await client.health(request_id="id")


@pytest.mark.parametrize("status_code", [201, 400, 500, 502])
async def test_health_unexpected_status(status_code):
    async with client_for(
        lambda _request: httpx.Response(status_code, json=healthy_payload())
    ) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.health(request_id="id")


async def test_register_exact_multipart_with_metadata_and_repeated_images():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=registration_payload())

    images = [
        UploadPart("one.jpg", b"jpeg-one", "image/jpeg"),
        UploadPart("two.png", b"png-two", "image/png"),
    ]
    async with client_for(handler) as client:
        result = await client.register_faces(
            person_id="EMP-001",
            display_name="Alice",
            images=images,
            metadata={"department": "Engineering"},
            request_id="register-id",
        )

    assert result.status == "success"
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/internal/v1/faces/register"
    assert request.headers["X-Request-ID"] == "register-id"
    assert request.extensions["timeout"] == {
        "connect": 2.0,
        "read": 30.0,
        "write": 30.0,
        "pool": 2.0,
    }
    parts = multipart_parts(request)
    names = [part.get_param("name", header="content-disposition") for part in parts]
    assert names == ["person_id", "display_name", "metadata", "images", "images"]
    assert part_value(parts[0]) == b"EMP-001"
    assert part_value(parts[1]) == b"Alice"
    assert part_value(parts[2]) == b'{"department":"Engineering"}'
    assert parts[3].get_filename() == "one.jpg"
    assert parts[3].get_content_type() == "image/jpeg"
    assert part_value(parts[3]) == b"jpeg-one"
    assert parts[4].get_filename() == "two.png"
    assert parts[4].get_content_type() == "image/png"
    assert part_value(parts[4]) == b"png-two"


async def test_register_omits_metadata_and_makes_exactly_one_request():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=registration_payload())

    async with client_for(handler) as client:
        await client.register_faces(
            person_id="EMP-001",
            display_name="Alice",
            images=[UploadPart("face.jpg", b"jpeg", "image/jpeg")],
            request_id="id",
        )

    assert len(seen) == 1
    parts = multipart_parts(seen[0])
    names = [part.get_param("name", header="content-disposition") for part in parts]
    assert names == ["person_id", "display_name", "images"]
    assert parts[2].get_filename() == "face.jpg"
    assert parts[2].get_content_type() == "image/jpeg"
    assert part_value(parts[2]) == b"jpeg"


async def test_recognize_exact_path_multipart_and_single_call():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=recognition_payload())

    async with client_for(handler) as client:
        result = await client.recognize_face(
            image=UploadPart("face.png", b"png-data", "image/png"),
            request_id="recognize-id",
        )

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/internal/v1/faces/recognize"
    assert request.headers["X-Request-ID"] == "recognize-id"
    parts = multipart_parts(request)
    assert len(parts) == 1
    assert parts[0].get_param("name", header="content-disposition") == "image"
    assert parts[0].get_filename() == "face.png"
    assert parts[0].get_content_type() == "image/png"
    assert part_value(parts[0]) == b"png-data"
    assert result.faces[0].matched_embedding_id == "emb-1"
    assert {"embedding", "embeddings", "candidates"}.isdisjoint(
        all_keys(result.model_dump())
    )


@pytest.mark.parametrize(
    "exception_type",
    [
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    ],
)
async def test_every_httpx_timeout_maps_safely(exception_type):
    def handler(request):
        raise exception_type("raw secret timeout", request=request)

    async with client_for(handler) as client:
        with pytest.raises(MLEngineTimeoutError) as captured:
            await client.health(request_id="id")
    assert "raw secret" not in str(captured.value)


@pytest.mark.parametrize("exception_type", [httpx.ConnectError, httpx.NetworkError])
async def test_connection_and_network_errors_map_safely(exception_type):
    def handler(request):
        raise exception_type("raw secret network", request=request)

    async with client_for(handler) as client:
        with pytest.raises(MLEngineConnectionError) as captured:
            await client.health(request_id="id")
    assert "raw secret" not in str(captured.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"{"),
        httpx.Response(200, content=b""),
        httpx.Response(200, content=b"\xff"),
        httpx.Response(200, json={"status": "ok"}),
        httpx.Response(200, json={**healthy_payload(), "extra": "forbidden"}),
        httpx.Response(200, json={**healthy_payload(), "gallery_size": "0"}),
    ],
)
async def test_malformed_json_and_strict_schema_failures(response):
    async with client_for(lambda _request: response) as client:
        with pytest.raises(MLEngineProtocolError) as captured:
            await client.health(request_id="id")
    assert "validation" not in str(captured.value).lower()


@pytest.mark.parametrize("status_code", [400, 413, 415, 422])
@pytest.mark.parametrize("error_code", KNOWN_ERROR_CODES)
async def test_valid_structured_client_downstream_errors(status_code, error_code):
    payload = {
        "error": {
            "code": error_code,
            "message": "downstream secret message",
            "request_id": "downstream-id",
        }
    }
    async with client_for(
        lambda _request: httpx.Response(status_code, json=payload)
    ) as client:
        with pytest.raises(MLEngineDownstreamError) as captured:
            await client.recognize_face(
                image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
                request_id="backend-id",
            )
    assert captured.value.status_code == status_code
    assert captured.value.downstream_code == error_code
    assert "secret" not in str(captured.value)
    assert "downstream-id" not in str(captured.value)


@pytest.mark.parametrize("status_code", [500, 503])
async def test_valid_server_downstream_errors_are_unavailable(status_code):
    payload = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "downstream secret message",
            "request_id": "downstream-id",
        }
    }
    async with client_for(
        lambda _request: httpx.Response(status_code, json=payload)
    ) as client:
        with pytest.raises(MLEngineUnavailableError) as captured:
            await client.recognize_face(
                image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
                request_id="backend-id",
            )
    assert "secret" not in str(captured.value)
    assert "downstream-id" not in str(captured.value)


async def test_unknown_downstream_error_code_is_protocol_failure():
    payload = {
        "error": {
            "code": "NEW_UNKNOWN_CODE",
            "message": "raw",
            "request_id": "downstream-id",
        }
    }
    async with client_for(lambda _request: httpx.Response(400, json=payload)) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.recognize_face(
                image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
                request_id="id",
            )


async def test_malformed_structured_downstream_error_is_protocol_failure():
    async with client_for(
        lambda _request: httpx.Response(400, json={"error": {"message": "raw"}})
    ) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.register_faces(
                person_id="EMP-001",
                display_name="Alice",
                images=[UploadPart("face.jpg", b"jpeg", "image/jpeg")],
                request_id="id",
            )


async def test_unexpected_operation_status_is_protocol_failure():
    async with client_for(
        lambda _request: httpx.Response(418, json={"detail": "secret"})
    ) as client:
        with pytest.raises(MLEngineProtocolError) as captured:
            await client.recognize_face(
                image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
                request_id="id",
            )
    assert "secret" not in str(captured.value)


async def test_direct_client_call_sanitizes_request_id():
    seen = []

    def handler(request):
        seen.append(request.headers["X-Request-ID"])
        return httpx.Response(200, json=healthy_payload())

    async with client_for(handler) as client:
        await client.health(request_id="bad direct id!")
    assert seen[0] != "bad direct id!"
    assert " " not in seen[0]


async def test_registration_and_recognition_response_enums_are_strict():
    bad_registration = {**registration_payload(), "status": "partial"}
    async with client_for(
        lambda _request: httpx.Response(200, json=bad_registration)
    ) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.register_faces(
                person_id="EMP-001",
                display_name="Alice",
                images=[UploadPart("face.jpg", b"jpeg", "image/jpeg")],
                request_id="id",
            )

    bad_recognition = recognition_payload()
    bad_recognition["faces"][0]["recognition_status"] = "maybe"
    async with client_for(
        lambda _request: httpx.Response(200, json=bad_recognition)
    ) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.recognize_face(
                image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
                request_id="id",
            )


@pytest.mark.parametrize("status", ["success", "rejected", "duplicate", "invalid"])
async def test_all_registration_statuses(status):
    payload = {**registration_payload(), "status": status}
    async with client_for(lambda _request: httpx.Response(200, json=payload)) as client:
        result = await client.register_faces(
            person_id="EMP-001",
            display_name="Alice",
            images=[UploadPart("face.jpg", b"jpeg", "image/jpeg")],
            request_id="id",
        )
    assert result.status == status


@pytest.mark.parametrize("recognition_status", ["known", "unknown"])
async def test_known_and_unknown_recognition(recognition_status):
    payload = recognition_payload()
    payload["faces"][0]["recognition_status"] = recognition_status
    if recognition_status == "unknown":
        payload["faces"][0]["person_id"] = None
        payload["faces"][0]["matched_embedding_id"] = None
    async with client_for(lambda _request: httpx.Response(200, json=payload)) as client:
        result = await client.recognize_face(
            image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
            request_id="id",
        )
    assert result.faces[0].recognition_status == recognition_status


@pytest.mark.parametrize(
    "payload",
    [
        {"face_count": 0, "processing_time_ms": 0.0, "faces": []},
        {
            "face_count": 2,
            "processing_time_ms": 1.0,
            "faces": recognition_payload()["faces"] * 2,
        },
    ],
)
async def test_zero_and_multiple_faces(payload):
    async with client_for(lambda _request: httpx.Response(200, json=payload)) as client:
        result = await client.recognize_face(
            image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
            request_id="id",
        )
    assert result.face_count == len(result.faces)


@pytest.mark.parametrize("bbox", [[1, 2, 3], [1, 2, 3, 4, 5]])
async def test_bbox_requires_exactly_four_values(bbox):
    payload = recognition_payload()
    payload["faces"][0]["bbox"] = bbox
    await assert_recognition_protocol_failure(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("similarity", -1.1),
        ("similarity", 1.1),
        ("threshold", -1.1),
        ("threshold", 1.1),
        ("detection_confidence", -0.1),
        ("detection_confidence", 1.1),
    ],
)
async def test_recognition_numeric_ranges(field, value):
    payload = recognition_payload()
    payload["faces"][0][field] = value
    await assert_recognition_protocol_failure(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {**registration_payload(), "accepted_count": -1},
        {**registration_payload(), "rejected_count": -1},
        {**registration_payload(), "embedding_dimension": 0},
        {
            **registration_payload(),
            "sample_results": [
                {"accepted": True, "reason": "accepted", "sample_index": -1}
            ],
        },
    ],
)
async def test_registration_rejects_invalid_counts_and_dimensions(payload):
    async with client_for(lambda _request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.register_faces(
                person_id="EMP-001",
                display_name="Alice",
                images=[UploadPart("face.jpg", b"jpeg", "image/jpeg")],
                request_id="id",
            )


@pytest.mark.parametrize(
    "payload",
    [
        {**healthy_payload(), "gallery_size": -1},
        {**recognition_payload(), "face_count": -1},
        {**recognition_payload(), "processing_time_ms": -0.1},
        {**recognition_payload(), "face_count": 2},
        {**recognition_payload(), "unexpected": True},
    ],
)
async def test_health_and_recognition_reject_invalid_contracts(payload):
    if "gallery_size" in payload:
        async with client_for(lambda _request: httpx.Response(200, json=payload)) as client:
            with pytest.raises(MLEngineProtocolError):
                await client.health(request_id="id")
    else:
        await assert_recognition_protocol_failure(payload)


async def assert_recognition_protocol_failure(payload):
    async with client_for(lambda _request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(MLEngineProtocolError):
            await client.recognize_face(
                image=UploadPart("face.jpg", b"jpeg", "image/jpeg"),
                request_id="id",
            )


async def test_response_json_is_parsed_exactly_once():
    class CountingResponse(httpx.Response):
        json_calls = 0

        def json(self, **kwargs):
            self.json_calls += 1
            return super().json(**kwargs)

    response = CountingResponse(200, json=healthy_payload())
    async with client_for(lambda _request: response) as client:
        await client.health(request_id="id")
    assert response.json_calls == 1
