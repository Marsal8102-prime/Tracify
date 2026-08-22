"""Tests for database configuration settings."""

import pytest
from pydantic import ValidationError

from backend.app.config import Settings


def test_database_defaults_with_env_file_isolation():
    settings = Settings(_env_file=None)
    raw_url = settings.database_url.get_secret_value()
    assert raw_url == "postgresql+asyncpg://tracify:change-me@127.0.0.1:5432/tracify"
    assert settings.database_pool_size == 5
    assert settings.database_max_overflow == 10
    assert settings.database_pool_timeout_seconds == 30.0
    assert settings.database_health_timeout_seconds == 3.0


def test_database_settings_accept_python_field_names():
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://u:p@h/db",
        database_pool_size=3,
        database_max_overflow=5,
        database_pool_timeout_seconds=10.0,
        database_health_timeout_seconds=1.5,
    )
    assert settings.database_url.get_secret_value() == "postgresql+asyncpg://u:p@h/db"
    assert settings.database_pool_size == 3
    assert settings.database_max_overflow == 5
    assert settings.database_pool_timeout_seconds == 10.0
    assert settings.database_health_timeout_seconds == 1.5


def test_database_settings_accept_environment_aliases(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://env:pass@host/envdb")
    monkeypatch.setenv("DATABASE_POOL_SIZE", "8")
    monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "3")
    monkeypatch.setenv("DATABASE_POOL_TIMEOUT_SECONDS", "15.0")
    monkeypatch.setenv("DATABASE_HEALTH_TIMEOUT_SECONDS", "2.0")
    settings = Settings(_env_file=None)
    assert settings.database_url.get_secret_value() == "postgresql+asyncpg://env:pass@host/envdb"
    assert settings.database_pool_size == 8
    assert settings.database_max_overflow == 3
    assert settings.database_pool_timeout_seconds == 15.0
    assert settings.database_health_timeout_seconds == 2.0


def test_database_url_requires_asyncpg_driver():
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(_env_file=None, database_url="postgresql://u:p@h/db")


def test_database_url_rejects_psycopg_driver():
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(_env_file=None, database_url="postgresql+psycopg://u:p@h/db")


def test_database_url_rejects_sqlite():
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(_env_file=None, database_url="sqlite+aiosqlite:///test.db")


def test_database_url_rejects_malformed():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="not-a-url")


def test_database_url_requires_database_name():
    with pytest.raises(ValidationError, match="database name"):
        Settings(_env_file=None, database_url="postgresql+asyncpg://u:p@h/")


def test_database_pool_size_rejects_zero():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_pool_size=0)


def test_database_pool_size_rejects_negative():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_pool_size=-1)


def test_database_max_overflow_rejects_negative():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_max_overflow=-1)


def test_database_max_overflow_accepts_zero():
    settings = Settings(_env_file=None, database_max_overflow=0)
    assert settings.database_max_overflow == 0


def test_database_pool_timeout_rejects_zero():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_pool_timeout_seconds=0)


def test_database_pool_timeout_rejects_negative():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_pool_timeout_seconds=-1.0)


def test_database_health_timeout_rejects_zero():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_health_timeout_seconds=0)


def test_database_health_timeout_rejects_negative():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_health_timeout_seconds=-1.0)


def test_database_url_repr_redacted():
    settings = Settings(_env_file=None)
    text = repr(settings)
    assert "change-me" not in text
    assert "tracify:change" not in text
    assert "127.0.0.1:5432" not in text


def test_database_url_json_dump_redacted():
    settings = Settings(_env_file=None)
    dump = settings.model_dump_json()
    assert "change-me" not in dump
    assert "tracify:change" not in dump


def test_database_url_with_percent_encoded_password():
    url = "postgresql+asyncpg://user:p%40ss%25word@host/db"
    settings = Settings(_env_file=None, database_url=url)
    assert settings.database_url.get_secret_value() == url


def test_alembic_percent_escaping_survives():
    """Percent-encoded credentials survive ConfigParser escaping."""
    url = "postgresql+asyncpg://user:p%40ss%25word@host/db"
    escaped = url.replace("%", "%%")
    assert "%%" in escaped
    # ConfigParser would un-escape %% to %
    restored = escaped.replace("%%", "%")
    assert restored == url
