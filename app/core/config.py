# Application settings, loaded from the environment (see .env.example).

from decimal import Decimal
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# lease_outbox_events() clamps its lease to the same range in SQL; keep the two in sync.
LEASE_MIN_SECONDS = 5
LEASE_MAX_SECONDS = 600


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
    # Failed processing attempts per outbox event (a crash, an expired lease, no handler) before
    # the worker gives up. Polls and retryable provider errors don't count: the job deadline
    # bounds those (the agreed retry policy).
    generation_max_attempts: int = Field(default=3, ge=1, le=10)
    generation_deadline_seconds: int = Field(default=900, ge=60, le=86_400)
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    # Retry and poll backoff per event: doubles from the base up to the max, with jitter.
    worker_backoff_base_seconds: float = Field(default=2.0, gt=0, le=60)
    worker_backoff_max_seconds: float = Field(default=60.0, gt=0, le=3600)
    worker_lease_seconds: int = Field(default=60, ge=LEASE_MIN_SECONDS, le=LEASE_MAX_SECONDS)
    # Events one worker leases and runs, one at a time, before it polls again.
    worker_max_events_per_run: int = Field(default=10, ge=1, le=100)
    # Retry delay for an event whose job has no handler (e.g. a kind deployed before its worker).
    worker_no_handler_retry_seconds: float = Field(default=60, gt=0, le=3600)
    higgsfield_base_url: str = "https://api.higgsfield.ai"
    higgsfield_model: str | None = None
    # None leaves it out of the request body (the model default applies).
    higgsfield_resolution: Literal["720p", "1080p"] | None = "1080p"
    # "<key-id>:<key-secret>", the Higgsfield SDK's format. Parsed by
    # app.services.providers.higgsfield.higgsfield_credentials.
    higgsfield_api_key: SecretStr | None = None
    # Staging/production: AWS Secrets Manager secret holding the key (e.g. nrc/stg/higgsfield).
    # The worker reads it at startup when HIGGSFIELD_API_KEY is not set.
    higgsfield_secret_id: str | None = None
    higgsfield_output_hosts: Annotated[tuple[str, ...], NoDecode] = ()
    # Downloading one provider output; the worker also caps it at half its lease.
    provider_output_fetch_timeout_seconds: float = Field(default=30, gt=0, le=300)

    # Media storage (NRC-3). The bucket comes only from configuration, never from code.
    aws_region: str = "us-east-1"
    aws_s3_bucket: str | None = None
    # Local SeaweedFS only; never set on servers, where the IAM role gives access.
    aws_endpoint_url_s3: str | None = None
    s3_force_path_style: bool = False
    aws_profile: str | None = None
    # S3 client: timeouts per request, and botocore's retries after the first attempt (standard
    # mode).
    s3_connect_timeout_seconds: float = Field(default=5, gt=0, le=60)
    s3_read_timeout_seconds: float = Field(default=30, gt=0, le=300)
    s3_max_retries: int = Field(default=3, ge=0, le=10)
    max_generated_image_bytes: int = Field(default=10_485_760, ge=1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1)
    asset_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    @field_validator("cognito_client_ids", "cors_origins", "higgsfield_output_hosts", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @property
    def cognito_issuer(self) -> str:
        return (
            f"https://cognito-idp.{self.cognito_region}.amazonaws.com/{self.cognito_user_pool_id}"
        )

    @property
    def cognito_jwks_url(self) -> str:
        return f"{self.cognito_issuer}/.well-known/jwks.json"


class WorkerDatabaseSettings(BaseSettings):
    # The connector worker's connection, as marketing_worker. Required: the worker refuses to
    # start without it; the API never uses it.

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore"
    )

    worker_database_url: SecretStr


class OwnerDatabaseSettings(BaseSettings):
    # Database-owner connection for migrations and seeding. The API never uses it.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    migration_database_url: SecretStr


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Return the process-wide settings (frozen, so the cached instance cannot be mutated).
    return Settings()
