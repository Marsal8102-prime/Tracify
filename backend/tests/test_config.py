from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.config import REPOSITORY_ROOT, Settings


def test_settings_defaults_and_env_file_isolation():
    settings = Settings(_env_file=None)
    assert settings.ml_engine_base_url == "http://127.0.0.1:8001"
    assert settings.ml_engine_connect_timeout_seconds == 2.0
    assert settings.ml_engine_read_timeout_seconds == 30.0
    assert settings.ml_engine_write_timeout_seconds == 30.0
    assert settings.ml_engine_pool_timeout_seconds == 2.0
    assert settings.ml_engine_health_timeout_seconds == 3.0


def test_settings_root_env_path_is_deterministic():
    assert REPOSITORY_ROOT == Path(__file__).resolve().parents[2]
    assert Settings.model_config["env_file"] == REPOSITORY_ROOT / ".env"


def test_settings_accepts_python_field_names():
    settings = Settings(
        _env_file=None,
        ml_engine_url="https://field-name.example.test/",
        ml_engine_connect_timeout_seconds=4.0,
    )
    assert settings.ml_engine_base_url == "https://field-name.example.test"
    assert settings.ml_engine_connect_timeout_seconds == 4.0


def test_settings_accepts_environment_alias_names():
    settings = Settings(
        _env_file=None,
        ML_ENGINE_URL="https://alias.example.test/",
        ML_ENGINE_READ_TIMEOUT_SECONDS=12.0,
    )
    assert settings.ml_engine_base_url == "https://alias.example.test"
    assert settings.ml_engine_read_timeout_seconds == 12.0


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost:8001",
        "http://user:secret@localhost:8001",
        "http://localhost:8001/internal",
        "http://localhost:8001/?debug=true",
        "http://localhost:8001/#fragment",
    ],
)
def test_settings_reject_invalid_ml_urls(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ML_ENGINE_URL=url)


def test_settings_accepts_https_and_normalizes_root():
    settings = Settings(_env_file=None, ML_ENGINE_URL="https://ml.example.test:8443/")
    assert settings.ml_engine_base_url == "https://ml.example.test:8443"


@pytest.mark.parametrize(
    "name",
    [
        "ML_ENGINE_CONNECT_TIMEOUT_SECONDS",
        "ML_ENGINE_READ_TIMEOUT_SECONDS",
        "ML_ENGINE_WRITE_TIMEOUT_SECONDS",
        "ML_ENGINE_POOL_TIMEOUT_SECONDS",
        "ML_ENGINE_HEALTH_TIMEOUT_SECONDS",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_settings_reject_nonpositive_timeouts(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_settings_environment_aliases(monkeypatch):
    monkeypatch.setenv("ML_ENGINE_URL", "https://ml.example.test")
    monkeypatch.setenv("ML_ENGINE_HEALTH_TIMEOUT_SECONDS", "4.5")
    settings = Settings(_env_file=None)
    assert settings.ml_engine_base_url == "https://ml.example.test"
    assert settings.ml_engine_health_timeout_seconds == 4.5
