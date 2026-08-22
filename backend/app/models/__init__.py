"""ORM model package — import all models so Alembic always sees complete metadata."""

from backend.app.models.person import Person

__all__ = ["Person"]
