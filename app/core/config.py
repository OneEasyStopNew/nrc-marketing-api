# Application settings, loaded from the environment (see .env.example).

from decimal import Decimal
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    # Typed configuration. Required values have no default, so a missing one fails at startup.

    # .env.local (git-ignored, local only) overrides .env; it holds provider keys such as
    # HIGGSFIELD_API_KEY.
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore", frozen=True
    )

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

    # Image generation (NRC-6, NRC-10). Live mode never falls back to the simulator.
    generation_mode: Literal["simulator", "live"] = "simulator"
    generation_max_active_jobs_per_tenant: int = Field(default=2, ge=1, le=50)
    generation_daily_credit_ceiling_per_tenant: int = Field(default=20, ge=0)
    generation_credits_per_image: int = Field(default=1, ge=1)
    # A job whose provider estimate is above this fails before anything is submitted.
    generation_max_cost_usd_per_image: Decimal = Field(default=Decimal("0.10"), ge=0)
    # Cap on provider spend across all tenants (reserved estimates plus committed costs), kept per
    # provider in provider_spend. A job that would pass it fails BUDGET_EXHAUSTED before submit.
    generation_global_budget_usd: Decimal = Field(default=Decimal(40), ge=0)
    # Stored on each job. Since NRC-10 retryable provider errors are bounded by the deadline
    # instead (the agreed retry policy), so this no longer limits retries.
    generation_max_attempts: int = Field(default=3, ge=1, le=10)
    generation_deadline_seconds: int = Field(default=900, ge=60, le=86_400)
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    worker_lease_seconds: int = Field(default=60, ge=5, le=3600)
    higgsfield_base_url: str = "https://api.higgsfield.ai"
    higgsfield_model: str | None = None
    # "<key-id>:<key-secret>", the Higgsfield SDK's format. Read it via higgsfield_credentials().
    higgsfield_api_key: SecretStr | None = None
    # Staging/production: AWS Secrets Manager secret holding the key (e.g. nrc/stg/higgsfield).
    # The worker reads it at startup when HIGGSFIELD_API_KEY is not set.
    higgsfield_secret_id: str | None = None
    higgsfield_output_hosts: Annotated[tuple[str, ...], NoDecode] = ()

    # Media storage (NRC-3). The bucket comes only from configuration, never from code.
    aws_region: str = "us-east-1"
    aws_s3_bucket: str | None = None
    # Local SeaweedFS only; never set on servers, where the IAM role gives access.
    aws_endpoint_url_s3: str | None = None
    s3_force_path_style: bool = False
    aws_profile: str | None = None
    max_generated_image_bytes: int = Field(default=10_485_760, ge=1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1)
    asset_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    @field_validator("cognito_client_ids", "cors_origins", "higgsfield_output_hosts", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    def higgsfield_credentials(self) -> tuple[SecretStr, SecretStr]:
        """Split HIGGSFIELD_API_KEY into (key id, key secret) at the first ':'.

        Not a field validator, because pydantic echoes input values in its errors.

        Raises:
            ValueError: HIGGSFIELD_API_KEY is missing or malformed. The message never contains
                the value.
        """
        raw = self.higgsfield_api_key.get_secret_value() if self.higgsfield_api_key else ""
        key_id, separator, key_secret = raw.partition(":")
        if not separator or not key_id.strip() or not key_secret.strip():
            raise ValueError(
                "HIGGSFIELD_API_KEY is missing or not in the form <key-id>:<key-secret>."
            )
        return SecretStr(key_id), SecretStr(key_secret)

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
