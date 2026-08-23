"""ORM model package — import all models so Alembic always sees complete metadata."""

from backend.app.models.person import Person
from backend.app.models.recognition_event import RecognitionEvent

__all__ = ["Person", "RecognitionEvent"]
