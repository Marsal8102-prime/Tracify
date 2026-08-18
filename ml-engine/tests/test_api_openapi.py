"""OpenAPI schema tests — verify endpoint documentation, multipart fields,
response schema references, and health 503 model."""
import pytest
from httpx import AsyncClient


# ── Helpers ────────────────────────────────────────────────────────────


def _resolve_ref(schema, ref_str):
    """Resolve a JSON Pointer $ref like '#/components/schemas/Foo'."""
    parts = ref_str.lstrip("#/").split("/")
    current = schema
    for part in parts:
        current = current[part]
    return current


def _resolve_schema(schema, obj):
    """Walk through $ref / allOf to reach the concrete schema properties."""
    if "$ref" in obj:
        return _resolve_schema(schema, _resolve_ref(schema, obj["$ref"]))
    if "allOf" in obj:
        merged = {}
        for item in obj["allOf"]:
            resolved = _resolve_schema(schema, item)
            merged.update(resolved.get("properties", {}))
        return {"properties": merged}
    return obj


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openapi_schema_paths(async_client: AsyncClient):
    response = await async_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})
    assert "/internal/v1/health" in paths
    assert "/internal/v1/faces/register" in paths
    assert "/internal/v1/faces/recognize" in paths


@pytest.mark.asyncio
async def test_openapi_register_multipart(async_client: AsyncClient):
    """Register endpoint documents multipart/form-data with expected fields."""
    response = await async_client.get("/openapi.json")
    schema = response.json()
    register = schema["paths"]["/internal/v1/faces/register"]["post"]
    assert "multipart/form-data" in register["requestBody"]["content"]
    body = register["requestBody"]["content"]["multipart/form-data"]["schema"]
    resolved = _resolve_schema(schema, body)
    props = resolved.get("properties", {})
    for field in ["person_id", "display_name", "metadata", "images"]:
        assert field in props, f"Missing field '{field}' in register schema"


@pytest.mark.asyncio
async def test_openapi_recognize_multipart(async_client: AsyncClient):
    """Recognize endpoint documents multipart/form-data with image field."""
    response = await async_client.get("/openapi.json")
    schema = response.json()
    recognize = schema["paths"]["/internal/v1/faces/recognize"]["post"]
    assert "multipart/form-data" in recognize["requestBody"]["content"]
    body = recognize["requestBody"]["content"]["multipart/form-data"]["schema"]
    resolved = _resolve_schema(schema, body)
    props = resolved.get("properties", {})
    assert "image" in props, "Missing field 'image' in recognize schema"


@pytest.mark.asyncio
async def test_openapi_error_response_schemas(async_client: AsyncClient):
    """Register and recognize endpoints document error responses using
    ErrorResponse (which contains an ErrorDetail with 'error' key)."""
    response = await async_client.get("/openapi.json")
    schema = response.json()

    for path in [
        "/internal/v1/faces/register",
        "/internal/v1/faces/recognize",
    ]:
        endpoint = schema["paths"][path]["post"]
        responses = endpoint["responses"]
        for code in ["400", "413", "415", "422", "500", "503"]:
            assert code in responses, (
                f"Missing response {code} in {path}"
            )
            resp_content = responses[code]["content"]["application/json"]
            resp_schema = _resolve_schema(schema, resp_content["schema"])
            assert "error" in resp_schema.get("properties", {}), (
                f"Response {code} at {path} missing 'error' property"
            )
            # Resolve 'error' $ref → ErrorDetail and verify its fields
            error_prop = resp_schema["properties"]["error"]
            error_detail = _resolve_schema(schema, error_prop)
            error_props = error_detail.get("properties", {})
            for field in ["code", "message", "request_id"]:
                assert field in error_props, (
                    f"ErrorDetail missing '{field}' in response {code} at {path}"
                )


@pytest.mark.asyncio
async def test_openapi_health_503_uses_health_response(async_client: AsyncClient):
    """Health 503 references HealthResponse (not ErrorResponse)."""
    response = await async_client.get("/openapi.json")
    schema = response.json()
    health = schema["paths"]["/internal/v1/health"]["get"]
    assert "503" in health["responses"]
    resp_content = health["responses"]["503"]["content"]["application/json"]
    resolved = _resolve_schema(schema, resp_content["schema"])
    props = resolved.get("properties", {})
    assert "status" in props
    assert "models_loaded" in props
    assert "version" in props
