"""Application settings via pydantic-settings (env-driven)."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_JWT_SECRET = "change-me-in-production"
MIN_JWT_SECRET_BYTES = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    environment: Literal["development", "staging", "production", "test"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    # NoDecode: without it pydantic-settings JSON-parses complex types straight
    # from the dotenv value, before the validator below ever sees the string.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- Auth ---
    jwt_secret: SecretStr = SecretStr(DEFAULT_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "aii"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 14
    interview_token_ttl_minutes: int = 10
    invite_ttl_hours: int = 72
    invite_max_redemptions: int = 3

    # --- Rate limiting ---
    login_max_attempts: int = 10
    login_attempt_window_seconds: int = 900
    register_max_attempts: int = 5
    register_attempt_window_seconds: int = 3600

    # --- Database ---
    # The app connects as an unprivileged role so RLS actually applies to it.
    database_url: str = "postgresql+asyncpg://app_user:app_pw@localhost:5432/ai_interviewer"
    # Alembic connects as the schema owner. The app must never use this.
    database_owner_url: str = (
        "postgresql+asyncpg://app_owner:owner_pw@localhost:5432/ai_interviewer"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 5

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def _hmac_only(cls, v: str) -> str:
        # The per-type key derivation in core.security produces symmetric keys, so an
        # asymmetric algorithm here would fail at signing time in a confusing way.
        if not v.startswith("HS"):
            raise ValueError(f"jwt_algorithm must be an HMAC algorithm (HS*), got {v!r}")
        return v

    @model_validator(mode="after")
    def _reject_weak_secret_in_production(self) -> "Settings":
        if self.environment != "production":
            return self
        secret = self.jwt_secret.get_secret_value()
        if secret == DEFAULT_JWT_SECRET:
            raise ValueError(
                "JWT_SECRET is still the example default; refusing to boot in production"
            )
        if len(secret.encode()) < MIN_JWT_SECRET_BYTES:
            raise ValueError(
                f"JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes in production"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Tests override this rather than mutating the module global."""
    return Settings()


settings = get_settings()
