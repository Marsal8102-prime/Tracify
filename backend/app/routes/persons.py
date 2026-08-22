import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_database_session
from backend.app.errors import BackendError, DatabaseUnavailableError, ErrorCode
from backend.app.models.person import Person
from backend.app.schemas import PersonCreate, PersonResponse, PersonUpdate

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
