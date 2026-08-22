import logging
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logger = logging.getLogger("tracify.backend.errors")


class ErrorCode(str, Enum):
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class BackendError(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ServiceUnavailableError(BackendError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.SERVICE_UNAVAILABLE,
            "A required service is unavailable.",
            503,
        )


class MLEngineError(Exception):
    """Base class for known, sanitized ML service failures."""


class MLEngineConnectionError(MLEngineError):
    pass


class MLEngineTimeoutError(MLEngineError):
    pass


class MLEngineProtocolError(MLEngineError):
    pass


class MLEngineUnavailableError(MLEngineError):
    pass


class MLEngineDownstreamError(MLEngineError):
    def __init__(self, status_code: int, downstream_code: str) -> None:
        super().__init__("The ML service rejected the request.")
        self.status_code = status_code
        self.downstream_code = downstream_code


class DatabaseUnavailableError(Exception):
    """Known, sanitized database availability failure.

    Stores no raw SQL, URL, hostname, or driver detail.
    """


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _response(status_code: int, code: ErrorCode, message: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code.value,
                "message": message,
                "request_id": _request_id(request),
            }
        },
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError) -> JSONResponse:
        return _response(exc.status_code, exc.code, exc.message, request)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return _response(
            422,
            ErrorCode.VALIDATION_ERROR,
            "The request data is invalid.",
            request,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _exc: Exception) -> JSONResponse:
        logger.error("Unhandled backend exception")
        return _response(
            500,
            ErrorCode.INTERNAL_ERROR,
            "An internal server error occurred.",
            request,
        )
