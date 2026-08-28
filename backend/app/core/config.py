"""Centralised, validated application configuration.

Every tunable value in GeoVisionGPT is declared here and sourced from the environment.
No module is permitted to read ``os.environ`` directly or to hardcode a limit, URL or
secret — that rule is what makes the deployment profiles in ``docs/00-architecture.md``
(local-lite / local-full / production) switchable without touching code.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings, loaded once and cached for the process lifetime."""

    model_config = SettingsConfigDict(
        env_prefix="GVG_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---------------------------------------------------------
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    project_name: str = "GeoVisionGPT"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False

    # --- Security ------------------------------------------------------------
    secret_key: SecretStr = SecretStr("dev-only-insecure-secret-key-change-me-please")
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)
    refresh_token_expire_days: int = Field(default=14, ge=1, le=90)
    password_min_length: int = Field(default=10, ge=8, le=128)

    # --- CORS ----------------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Database ------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./geovision.db"
    db_echo: bool = False
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=20, ge=0)

    # --- Cache / rate limiting ----------------------------------------------
    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_enabled: bool = True
    rate_limit_login_attempts: int = Field(default=5, ge=1)
    rate_limit_login_window_seconds: int = Field(default=300, ge=1)
    rate_limit_default_requests: int = Field(default=120, ge=1)
    rate_limit_default_window_seconds: int = Field(default=60, ge=1)

    # --- Object storage (Module 2) ------------------------------------------
    storage_backend: Literal["filesystem", "minio", "s3"] = "filesystem"
    storage_root: str = "./storage"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: SecretStr = SecretStr("minioadmin")
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "geovision"

    # --- Bootstrap admin -----------------------------------------------------
    # Must be a routable domain: reserved TLDs (.local, .test, .example, .invalid) are
    # rejected by email validation, which would make the bootstrap admin unserialisable.
    first_admin_email: str = "admin@geovisiongpt.io"
    # Must not contain the local part of first_admin_email ("admin") — PasswordPolicy
    # rejects a password containing the account's own email local-part, and the bootstrap
    # path now enforces that policy (see infrastructure/bootstrap.py::ensure_first_admin).
    first_admin_password: SecretStr = SecretStr("ChangeMe_Please_2026")
    first_admin_name: str = "Platform Administrator"

    # --- Inference backend (Modules 6/7) ------------------------------------
    inference_backend: Literal["null", "openai_compatible", "local_transformers"] = "null"
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_base_url: str = "https://openrouter.ai/api/v1"
    vlm_api_key: SecretStr = SecretStr("")

    # --- Derived helpers -----------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @field_validator("secret_key")
    @classmethod
    def _reject_weak_secret(cls, value: SecretStr) -> SecretStr:
        """A short key silently weakens every token the system issues, so fail loudly."""
        if len(value.get_secret_value()) < 32:
            raise ValueError("GVG_SECRET_KEY must be at least 32 characters")
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that validation runs once; tests clear the cache via
    ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()
