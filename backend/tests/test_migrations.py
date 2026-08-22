"""Tests for Alembic configuration, migrations, and model/migration consistency."""

import os
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from backend.app.models.base import Base

# Import models to populate metadata
import backend.app.models  # noqa: F401


BACKEND_DIR = Path(__file__).resolve().parents[1]
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"


def _make_alembic_config(database_url: str) -> Config:
    """Build an Alembic Config with an injected test database URL."""
    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["database_url"] = database_url
    return cfg


def test_alembic_ini_has_no_url():
    cfg = Config(str(ALEMBIC_INI))
    url = cfg.get_main_option("sqlalchemy.url")
    assert url is None or url == ""


def test_alembic_script_location_is_relative():
    import configparser
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ALEMBIC_INI)
    script_location = parser.get("alembic", "script_location")
    assert script_location.startswith("%(here)s")
    assert not os.path.isabs(script_location.replace("%(here)s", "").lstrip("/").lstrip("\\"))


def test_model_metadata_includes_persons():
    assert "persons" in Base.metadata.tables


def test_exactly_one_head():
    cfg = _make_alembic_config("sqlite:///unused")
    from alembic.script import ScriptDirectory
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1
    assert heads[0] == "0001"


def test_exactly_one_revision():
    cfg = _make_alembic_config("sqlite:///unused")
    from alembic.script import ScriptDirectory
    script = ScriptDirectory.from_config(cfg)
    revisions = list(script.walk_revisions())
    assert len(revisions) == 1


def test_revision_parent_is_none():
    cfg = _make_alembic_config("sqlite:///unused")
    from alembic.script import ScriptDirectory
    script = ScriptDirectory.from_config(cfg)
    rev = script.get_revision("0001")
    assert rev is not None
    assert rev.down_revision is None


def test_offline_sql_generation_succeeds():
    cfg = _make_alembic_config("postgresql+asyncpg://user:pass@host/db")
    output = StringIO()
    cfg.output_buffer = output
    command.upgrade(cfg, "head", sql=True)
    sql = output.getvalue()
    assert "CREATE TABLE persons" in sql


def test_offline_sql_does_not_contain_password():
    cfg = _make_alembic_config("postgresql+asyncpg://user:secretpass@host/db")
    output = StringIO()
    cfg.output_buffer = output
    command.upgrade(cfg, "head", sql=True)
    sql = output.getvalue()
    assert "secretpass" not in sql


def test_sqlite_upgrade_creates_persons_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        url = f"sqlite+aiosqlite:///{db_path}"
        cfg = _make_alembic_config(url)
        command.upgrade(cfg, "head")

        sync_url = f"sqlite:///{db_path}"
        engine = create_engine(sync_url)
        insp = inspect(engine)
        assert "persons" in insp.get_table_names()
        engine.dispose()


def test_sqlite_downgrade_removes_persons_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        url = f"sqlite+aiosqlite:///{db_path}"
        cfg = _make_alembic_config(url)

        command.upgrade(cfg, "head")
        
        sync_url = f"sqlite:///{db_path}"
        engine = create_engine(sync_url)
        assert "persons" in inspect(engine).get_table_names()

        command.downgrade(cfg, "base")
        assert "persons" not in inspect(engine).get_table_names()
        engine.dispose()


def test_migration_columns_match_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        url = f"sqlite+aiosqlite:///{db_path}"
        cfg = _make_alembic_config(url)
        command.upgrade(cfg, "head")

        sync_url = f"sqlite:///{db_path}"
        engine = create_engine(sync_url)
        insp = inspect(engine)
        migration_columns = {c["name"] for c in insp.get_columns("persons")}
        model_columns = {c.name for c in Base.metadata.tables["persons"].columns}
        assert migration_columns == model_columns
        engine.dispose()


def test_migration_no_create_all():
    """Verify migration does not use Base.metadata.create_all()."""
    migration_file = BACKEND_DIR / "alembic" / "versions" / "0001_create_persons_table.py"
    content = migration_file.read_text()
    assert "create_all" not in content


def test_migration_no_pgvector_or_extensions():
    migration_file = BACKEND_DIR / "alembic" / "versions" / "0001_create_persons_table.py"
    content = migration_file.read_text()
    assert "pgvector" not in content
    assert "create_extension" not in content.lower()
    assert "CREATE EXTENSION" not in content


def test_migration_no_enum_or_auth_tables():
    migration_file = BACKEND_DIR / "alembic" / "versions" / "0001_create_persons_table.py"
    content = migration_file.read_text()
    assert "password" not in content.lower()
    assert "jwt" not in content.lower()
    assert "authentication" not in content.lower()


def test_migration_constraints_and_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        url = f"sqlite+aiosqlite:///{db_path}"
        cfg = _make_alembic_config(url)
        command.upgrade(cfg, "head")

        sync_url = f"sqlite:///{db_path}"
        engine = create_engine(sync_url)
        insp = inspect(engine)

        # Check indexes
        indexes = insp.get_indexes("persons")
        index_names = {idx["name"] for idx in indexes}
        assert "ix_persons_status" in index_names

        # Check unique constraints
        unique_constraints = insp.get_unique_constraints("persons")
        unique_names = {uc["name"] for uc in unique_constraints}
        assert "uq_persons_person_id" in unique_names

        engine.dispose()
