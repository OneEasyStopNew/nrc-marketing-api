# Application settings, loaded from the environment (see .env.example).

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AppEnv = Literal["local", "demo", "staging", "production"]
# Environments where simulated data and locally signed tokens are allowed.
NON_PRODUCTION_ENVS: Final[frozenset[str]] = frozenset({"local", "demo"})


class Settings(BaseSettings):
    # Typed configuration. Required values have no default, so a missing one fails at startup.

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    # Defaults to the strictest value: a deployment that forgets it gets no dev shortcuts.
    app_env: AppEnv = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_docs: bool = False
    max_request_body_bytes: int = Field(default=1_048_576, ge=1024)

    database_url: SecretStr
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_timeout_seconds: float = Field(default=10, gt=0, le=60)

    cognito_region: str = Field(pattern=r"^[a-z]{2}(-[a-z]+)+-[0-9]$")
    cognito_user_pool_id: str = Field(pattern=r"^[a-z]{2}(-[a-z]+)+-[0-9]_[A-Za-z0-9]+$")
    cognito_client_ids: Annotated[tuple[str, ...], NoDecode] = Field(min_length=1)
    required_scope: str = Field(min_length=1)
    jwks_cache_ttl_seconds: float = Field(default=3600, gt=0)
    http_timeout_seconds: float = Field(default=5, gt=0, le=30)

    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()

    # Simulated validation, generation, asset and export responses so the frontend can integrate
    # before D2-D5 ship. Local and demo environments only; never in staging or production.
    demo_stubs: bool = False
    # Absolute base URL the frontend reaches this API on, used for placeholder image links.
    demo_public_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")
    # Local JWKS file (from `python -m scripts.dev_token init`) used instead of Cognito's keys.
    # Development only, until Cognito exists. Never set in staging or production.
    dev_jwks_file: Path | None = None

    @field_validator("cognito_client_ids", "cors_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @field_validator("dev_jwks_file", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        # `DEV_JWKS_FILE=` (as in .env.example) means "use Cognito", not the current directory.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _shortcuts_only_outside_production(self) -> Self:
        if self.app_env in NON_PRODUCTION_ENVS:
            return self
        enabled = [
            name
            for name, on in (("DEMO_STUBS", self.demo_stubs), ("DEV_JWKS_FILE", self.dev_jwks_file))
            if on
        ]
        if enabled:
            raise ValueError(
                f"{', '.join(enabled)} cannot be used when APP_ENV={self.app_env!r}; "
                "set APP_ENV=local (or demo) for development"
            )
        return self

    @property
    def cognito_issuer(self) -> str:
        return (
            f"https://cognito-idp.{self.cognito_region}.amazonaws.com/{self.cognito_user_pool_id}"
        )

    @property
    def cognito_jwks_url(self) -> str:
        return f"{self.cognito_issuer}/.well-known/jwks.json"


class OwnerDatabaseSettings(BaseSettings):
    # Database-owner connection for migrations and seeding. The API never uses it.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    migration_database_url: SecretStr


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Return the process-wide settings (frozen, so the cached instance cannot be mutated).
    return Settings()
