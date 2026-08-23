import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.clients.ml_engine import MLEngineClient, UploadPart
from backend.app.dependencies import get_database_session, get_ml_engine_client
from backend.app.errors import BackendError, ErrorCode
from backend.app.schemas import MLRecognitionResponse
from backend.app.services.recognition import RecognitionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recognition", tags=["recognition"])


def get_recognition_service(
    session: Annotated[AsyncSession, Depends(get_database_session)],
    ml_client: Annotated[MLEngineClient, Depends(get_ml_engine_client)],
) -> RecognitionService:
    return RecognitionService(session=session, ml_client=ml_client)


@router.post(
    "/recognize",
    response_model=MLRecognitionResponse,
    status_code=status.HTTP_200_OK,
)
async def recognize_face(
    request: Request,
    service: Annotated[RecognitionService, Depends(get_recognition_service)],
    image: UploadFile | None = File(None),
):
    if not image:
        raise BackendError(
            ErrorCode.VALIDATION_ERROR,
            "An image file is required.",
            422,
        )

    if image.content_type not in ("image/jpeg", "image/png"):
        raise BackendError(
            ErrorCode.VALIDATION_ERROR,
            f"Unsupported image type '{image.content_type}'. Only image/jpeg and image/png are allowed.",
            400,
        )

    content = await image.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit per image
        raise BackendError(
            ErrorCode.VALIDATION_ERROR,
            f"Image '{image.filename}' exceeds the 10MB limit.",
            400,
        )

    upload_part = UploadPart(
        filename=image.filename or "unknown",
        content=content,
        content_type=image.content_type,  # type: ignore[arg-type]
    )

    # Extract request_id from RequestState (injected by RequestIDMiddleware)
    request_id = getattr(request.state, "request_id", "unknown")

    return await service.process_recognition(
        image_part=upload_part,
        request_id=request_id,
    )
