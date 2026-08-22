from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    ml_engine_url: AnyHttpUrl = Field(
        default="http://127.0.0.1:8001",
        validation_alias="ML_ENGINE_URL",
    )
    ml_engine_connect_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        validation_alias="ML_ENGINE_CONNECT_TIMEOUT_SECONDS",
    )
    ml_engine_read_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="ML_ENGINE_READ_TIMEOUT_SECONDS",
    )
    ml_engine_write_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="ML_ENGINE_WRITE_TIMEOUT_SECONDS",
    )
    ml_engine_pool_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        validation_alias="ML_ENGINE_POOL_TIMEOUT_SECONDS",
    )
    ml_engine_health_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        validation_alias="ML_ENGINE_HEALTH_TIMEOUT_SECONDS",
    )

    # ---- Database ----
    database_url: SecretStr = Field(
        default="postgresql+asyncpg://tracify:change-me@127.0.0.1:5432/tracify",
        validation_alias="DATABASE_URL",
    )
    database_pool_size: int = Field(
        default=5,
        gt=0,
        validation_alias="DATABASE_POOL_SIZE",
    )
    database_max_overflow: int = Field(
        default=10,
        ge=0,
        validation_alias="DATABASE_MAX_OVERFLOW",
    )
    database_pool_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="DATABASE_POOL_TIMEOUT_SECONDS",
    )
    database_health_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        validation_alias="DATABASE_HEALTH_TIMEOUT_SECONDS",
    )

    @field_validator("ml_engine_url", mode="before")
    @classmethod
    def validate_ml_engine_url(cls, value: Any) -> Any:
        text = str(value).strip()
        if not text.lower().startswith(("http://", "https://")):
            raise ValueError("ML engine URL must use HTTP or HTTPS.")
        return text

    @field_validator("ml_engine_url")
    @classmethod
    def restrict_ml_engine_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("ML engine URL must not contain credentials.")
        if value.query is not None or value.fragment is not None:
            raise ValueError("ML engine URL must not contain query parameters or fragments.")
        if value.path not in (None, "", "/"):
            raise ValueError("ML engine URL must not contain a non-root path.")
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not raw.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the postgresql+asyncpg:// driver."
            )
        # Require a database name after the last /
        after_scheme = raw.split("://", 1)[1]
        path_part = after_scheme.rsplit("/", 1)
        if len(path_part) < 2 or not path_part[1].split("?")[0]:
            raise ValueError(
                "DATABASE_URL must include a database name."
            )
        return value

    @property
    def ml_engine_base_url(self) -> str:
        return str(self.ml_engine_url).rstrip("/")
