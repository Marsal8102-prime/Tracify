from fastapi import APIRouter, Depends, Request

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.dependencies import get_ml_engine_client
from backend.app.errors import MLEngineError, ServiceUnavailableError
from backend.app.schemas import ErrorResponse, LivenessResponse, ReadinessResponse
from backend.app.version import VERSION


router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get(
    "/live",
    response_model=LivenessResponse,
    responses={200: {"model": LivenessResponse, "description": "Backend is live"}},
)
async def live() -> LivenessResponse:
    return LivenessResponse(status="ok", version=VERSION)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"model": ReadinessResponse, "description": "Backend is ready"},
        503: {"model": ErrorResponse, "description": "Required service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def ready(
    request: Request,
    ml_client: MLEngineClient = Depends(get_ml_engine_client),
) -> ReadinessResponse:
    try:
        await ml_client.health(request_id=request.state.request_id)
    except MLEngineError as exc:
        raise ServiceUnavailableError() from exc
    return ReadinessResponse(status="ready", version=VERSION)
