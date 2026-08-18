import json
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from backend.app.errors import (
    MLEngineConnectionError,
    MLEngineDownstreamError,
    MLEngineProtocolError,
    MLEngineTimeoutError,
    MLEngineUnavailableError,
)
from backend.app.middleware import REQUEST_ID_HEADER, sanitize_request_id
from backend.app.schemas import (
    MLDownstreamErrorResponse,
    MLHealthResponse,
    MLRecognitionResponse,
    MLRegistrationResponse,
)


HEALTH_PATH = "/internal/v1/health"
REGISTER_PATH = "/internal/v1/faces/register"
RECOGNIZE_PATH = "/internal/v1/faces/recognize"
DOWNSTREAM_CLIENT_ERROR_STATUSES = {400, 413, 415, 422}
DOWNSTREAM_UNAVAILABLE_STATUSES = {500, 503}
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


@dataclass(frozen=True)
class UploadPart:
    filename: str
    content: bytes
    content_type: Literal["image/jpeg", "image/png"]


class MLEngineClient:
    def __init__(self, client: httpx.AsyncClient, health_timeout_seconds: float) -> None:
        self._client = client
        self._health_timeout_seconds = health_timeout_seconds

    async def health(self, *, request_id: str) -> MLHealthResponse:
        response = await self._request(
            "GET",
            HEALTH_PATH,
            request_id=request_id,
            timeout=self._health_timeout_seconds,
        )
        if response.status_code not in (200, 503):
            raise MLEngineProtocolError("The ML service returned an unexpected status.")

        health = self._validate_response(response, MLHealthResponse)
        if (
            response.status_code != 200
            or health.status != "ok"
            or not health.models_loaded
            or not health.gallery_loaded
        ):
            raise MLEngineUnavailableError("The ML service is unavailable.")
        return health

    async def register_faces(
        self,
        *,
        person_id: str,
        display_name: str,
        images: Sequence[UploadPart],
        metadata: Mapping[str, str] | None = None,
        request_id: str,
    ) -> MLRegistrationResponse:
        data = {"person_id": person_id, "display_name": display_name}
        if metadata is not None:
            data["metadata"] = json.dumps(metadata, separators=(",", ":"))
        files = [
            ("images", (image.filename, image.content, image.content_type))
            for image in images
        ]
        response = await self._request(
            "POST",
            REGISTER_PATH,
            request_id=request_id,
            data=data,
            files=files,
        )
        return self._parse_operation_response(response, MLRegistrationResponse)

    async def recognize_face(
        self,
        *,
        image: UploadPart,
        request_id: str,
    ) -> MLRecognitionResponse:
        response = await self._request(
            "POST",
            RECOGNIZE_PATH,
            request_id=request_id,
            files={"image": (image.filename, image.content, image.content_type)},
        )
        return self._parse_operation_response(response, MLRecognitionResponse)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        request_id: str,
        **kwargs: object,
    ) -> httpx.Response:
        headers = {REQUEST_ID_HEADER: sanitize_request_id(request_id)}
        try:
            return await self._client.request(method, path, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise MLEngineTimeoutError("The ML service request timed out.") from exc
        except httpx.ProtocolError as exc:
            raise MLEngineProtocolError("The ML service protocol failed.") from exc
        except httpx.NetworkError as exc:
            raise MLEngineConnectionError("The ML service could not be reached.") from exc
        except httpx.TransportError as exc:
            raise MLEngineConnectionError("The ML service transport failed.") from exc

    @staticmethod
    def _response_json(response: httpx.Response) -> object:
        try:
            return response.json()
        except ValueError as exc:
            raise MLEngineProtocolError("The ML service returned malformed JSON.") from exc

    @classmethod
    def _validate_response(
        cls,
        response: httpx.Response,
        model: type[ResponseModel],
    ) -> ResponseModel:
        payload = cls._response_json(response)
        try:
            return model.model_validate(payload, strict=True)
        except ValidationError as exc:
            raise MLEngineProtocolError("The ML service response violated its contract.") from exc

    @classmethod
    def _parse_operation_response(
        cls,
        response: httpx.Response,
        model: type[ResponseModel],
    ) -> ResponseModel:
        if response.status_code == 200:
            return cls._validate_response(response, model)
        if response.status_code in (
            DOWNSTREAM_CLIENT_ERROR_STATUSES | DOWNSTREAM_UNAVAILABLE_STATUSES
        ):
            error = cls._validate_response(response, MLDownstreamErrorResponse)
            if response.status_code in DOWNSTREAM_UNAVAILABLE_STATUSES:
                raise MLEngineUnavailableError("The ML service is unavailable.")
            raise MLEngineDownstreamError(response.status_code, error.error.code)
        raise MLEngineProtocolError("The ML service returned an unexpected status.")
