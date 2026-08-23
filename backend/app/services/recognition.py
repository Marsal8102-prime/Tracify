import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError

from backend.app.clients.ml_engine import MLEngineClient, UploadPart
from backend.app.models.person import Person
from backend.app.models.recognition_event import RecognitionEvent
from backend.app.schemas import MLRecognitionResponse
from backend.app.errors import DatabaseUnavailableError

logger = logging.getLogger(__name__)


class RecognitionService:
    def __init__(self, session: AsyncSession, ml_client: MLEngineClient):
        self.session = session
        self.ml_client = ml_client

    async def process_recognition(
        self,
        image_part: UploadPart,
        request_id: str,
    ) -> MLRecognitionResponse:
        """Process face recognition for an uploaded image.
        
        Calls the ML Engine for face recognition, verifies that any 'known' person_id
        actually exists in the database, and persists the results as RecognitionEvents.
        
        Note: The returned MLRecognitionResponse is NOT modified, even if a person_id
        is found to be orphaned (missing from the database). Only the persisted
        RecognitionEvent is normalized to 'unknown' with a null person_id in that case.
        """
        # Call the ML Engine
        ml_response = await self.ml_client.recognize_face(
            image=image_part,
            request_id=request_id,
        )

        if ml_response.face_count == 0:
            # No faces detected, nothing to persist
            return ml_response

        # Extract all 'known' person_ids from the ML response
        known_person_ids = {
            face.person_id for face in ml_response.faces
            if face.recognition_status == "known" and face.person_id is not None
        }

        # Query the database to see which ones actually exist
        existing_person_ids: set[str] = set()
        if known_person_ids:
            try:
                result = await self.session.execute(
                    select(Person.person_id).where(Person.person_id.in_(known_person_ids))
                )
                existing_person_ids = set(result.scalars().all())
            except OperationalError as e:
                raise DatabaseUnavailableError() from e

        # Prepare bulk insert of RecognitionEvents
        events_to_insert = []
        for face in ml_response.faces:
            db_person_id = face.person_id
            db_status = face.recognition_status

            # If the ML Engine claims it's known, but we don't have it in the DB,
            # normalize the persisted event to 'unknown' to avoid a foreign key constraint failure.
            if db_status == "known" and db_person_id not in existing_person_ids:
                logger.warning(
                    f"Orphaned person_id '{db_person_id}' returned by ML Engine for request {request_id}. "
                    "Normalizing persisted event to 'unknown'."
                )
                db_person_id = None
                db_status = "unknown"
            
            # Note: For naturally 'unknown' faces, face.person_id is already None from the ML engine.
            if db_status == "unknown":
                db_person_id = None

            event = RecognitionEvent(
                request_id=request_id,
                person_id=db_person_id,
                recognition_status=db_status,
                similarity=face.similarity,
                threshold=face.threshold,
                detection_confidence=face.detection_confidence,
                bbox=face.bbox,
            )
            events_to_insert.append(event)

        if events_to_insert:
            self.session.add_all(events_to_insert)
            try:
                await self.session.commit()
            except OperationalError as e:
                await self.session.rollback()
                raise DatabaseUnavailableError() from e

        # Return the original, unmodified ML response
        return ml_response
