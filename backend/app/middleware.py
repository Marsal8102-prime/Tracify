import re
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")


def sanitize_request_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if REQUEST_ID_REGEX.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = sanitize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
