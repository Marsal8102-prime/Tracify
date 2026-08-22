import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.clients.ml_engine import MLEngineClient, UploadPart
from backend.app.dependencies import get_database_session, get_ml_engine_client
from backend.app.errors import BackendError, DatabaseUnavailableError, ErrorCode
from backend.app.models.person import Person
from backend.app.schemas import MLRegistrationResponse, PersonCreate, PersonResponse, PersonUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/persons", tags=["persons"])


@router.post(
    "",
    response_model=PersonResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_person(
    payload: PersonCreate,
    session: Annotated[AsyncSession, Depends(get_database_session)],
):
    person = Person(
        person_id=payload.person_id,
        display_name=payload.display_name,
        department=payload.department,
        designation=payload.designation,
        profile_metadata=payload.profile_metadata,
    )
    session.add(person)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        error_msg = str(e).lower()
        if "uq_persons_person_id" in error_msg or ("unique constraint" in error_msg and "person_id" in error_msg):
            raise BackendError(
                ErrorCode.CONFLICT,
                "A person with this person_id already exists.",
                409,
            ) from e
        raise
    except OperationalError as e:
        await session.rollback()
        raise DatabaseUnavailableError() from e

    await session.refresh(person)
    return person


@router.get(
    "/{person_id}",
    response_model=PersonResponse,
)
async def get_person(
    person_id: str,
    session: Annotated[AsyncSession, Depends(get_database_session)],
):
    try:
        result = await session.execute(select(Person).where(Person.person_id == person_id))
        person = result.scalar_one_or_none()
    except OperationalError as e:
        raise DatabaseUnavailableError() from e

    if not person:
        raise BackendError(
            ErrorCode.NOT_FOUND,
            "Person not found.",
            404,
        )
    return person


@router.get(
    "",
    response_model=list[PersonResponse],
)
async def list_persons(
    session: Annotated[AsyncSession, Depends(get_database_session)],
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    try:
        result = await session.execute(
            select(Person).order_by(Person.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())
    except OperationalError as e:
        raise DatabaseUnavailableError() from e


@router.patch(
    "/{person_id}",
    response_model=PersonResponse,
)
async def update_person(
    person_id: str,
    payload: PersonUpdate,
    session: Annotated[AsyncSession, Depends(get_database_session)],
):
    try:
        result = await session.execute(select(Person).where(Person.person_id == person_id))
        person = result.scalar_one_or_none()
    except OperationalError as e:
        raise DatabaseUnavailableError() from e

    if not person:
        raise BackendError(
            ErrorCode.NOT_FOUND,
            "Person not found.",
            404,
        )

    update_data = payload.model_dump(exclude_unset=True)
    if not update_data:
        return person

    for key, value in update_data.items():
        setattr(person, key, value)

    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        error_msg = str(e).lower()
        if "uq_persons_person_id" in error_msg or ("unique constraint" in error_msg and "person_id" in error_msg):
            raise BackendError(
                ErrorCode.CONFLICT,
                "A person with this person_id already exists.",
                409,
            ) from e
        raise
    except OperationalError as e:
        await session.rollback()
        raise DatabaseUnavailableError() from e

    await session.refresh(person)
    return person


@router.post(
    "/{person_id}/register",
    response_model=MLRegistrationResponse,
)
async def register_person_faces(
    person_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_database_session)],
    ml_client: Annotated[MLEngineClient, Depends(get_ml_engine_client)],
    images: list[UploadFile] = File(...),
):
    if not images:
        raise BackendError(
            ErrorCode.VALIDATION_ERROR,
            "At least one image is required.",
            400,
        )
    if len(images) > 5:
        raise BackendError(
            ErrorCode.VALIDATION_ERROR,
            "Too many images. Maximum 5 images allowed per request.",
            400,
        )

    try:
        result = await session.execute(select(Person).where(Person.person_id == person_id))
        person = result.scalar_one_or_none()
    except OperationalError as e:
        raise DatabaseUnavailableError() from e

    if not person:
        raise BackendError(
            ErrorCode.NOT_FOUND,
            "Person not found.",
            404,
        )

    upload_parts: list[UploadPart] = []
    for image in images:
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
        upload_parts.append(
            UploadPart(
                filename=image.filename or "unknown",
                content=content,
                content_type=image.content_type,  # type: ignore[arg-type]
            )
        )

    # Use the request_id from the RequestState (middleware)
    request_id = getattr(request.state, "request_id", "unknown")

    return await ml_client.register_faces(
        person_id=person.person_id,
        display_name=person.display_name,
        images=upload_parts,
        metadata={"source": "backend_api"},
        request_id=request_id,
    )
