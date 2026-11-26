# Application settings, loaded from the environment (see .env.example).

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.media_types import IMAGE_EXTENSIONS

AppEnv = Literal["local", "demo", "staging", "production"]
# Environments where simulated data and locally signed tokens are allowed.
NON_PRODUCTION_ENVS: Final[frozenset[str]] = frozenset({"local", "demo"})
# lease_outbox_events() clamps its lease to the same range in SQL; keep the two in sync.
LEASE_MIN_SECONDS = 5
LEASE_MAX_SECONDS = 600
# The types app.services.media can decode (app.core.media_types).
SUPPORTED_UPLOAD_IMAGE_TYPES: Final[frozenset[str]] = frozenset(IMAGE_EXTENSIONS)


class EmbeddingSettings(BaseSettings):
    # What policy embeddings need, on its own so owner scripts (seed, index build) can load it
    # without the API's settings (Cognito, DATABASE_URL). Settings inherits every field here.

    # .env.local (git-ignored, local only) overrides .env; it holds provider keys such as
    # HIGGSFIELD_API_KEY.
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    # Defaults to the strictest value: a deployment that forgets it gets no dev shortcuts.
    app_env: AppEnv = "production"
    aws_region: str = "us-east-1"
    # Policy embeddings (NRC-5). "off" builds and uses no vectors (retrieval is lexical only), so
    # staging and production start unchanged until Bedrock access exists. "live" never falls back
    # to the simulator and refuses to start without BEDROCK_EMBEDDING_MODEL_ID. The simulator's
    # vectors carry no meaning: local and demo only.
    embedding_mode: Literal["off", "simulator", "live"] = "off"
    # One of the pinned models in app.services.providers.bedrock_embeddings.PINNED_MODELS.
    bedrock_embedding_model_id: str | None = None
    # Bedrock client: connect and read timeout per request, and botocore's retries after the
    # first attempt (standard mode). The region is AWS_REGION; access comes from the IAM role.
    bedrock_timeout_seconds: float = Field(default=10, gt=0, le=60)
    bedrock_max_retries: int = Field(default=3, ge=0, le=10)

    @model_validator(mode="after")
    def _simulated_embeddings_only_outside_production(self) -> Self:
        if self.embedding_mode == "simulator" and self.app_env not in NON_PRODUCTION_ENVS:
            raise ValueError(
                f"EMBEDDING_MODE=simulator cannot be used when APP_ENV={self.app_env!r}; "
                "set APP_ENV=local (or demo) for development"
            )
        return self


class RetrievalSettings(EmbeddingSettings):
    # What policy retrieval needs (its bounds plus the embedding settings), on its own so the
    # retrieval evaluation script runs without the API's settings. Settings inherits it.

    # Policy retrieval (NRC-5): bounds on the graph walk from the selected rules: hops per rule,
    # and evidence nodes plus exception entries per manifest. Hitting either is recorded in the
    # context manifest (TRAVERSAL_TRUNCATED), never silently dropped.
    policy_retrieval_max_hops: int = Field(default=3, ge=1, le=10)
    policy_retrieval_max_nodes: int = Field(default=200, ge=10, le=2000)
    # Search: matches kept per statement (per search, and after fusing them), and statements
    # searched per request (more are recorded as STATEMENTS_TRUNCATED, not silently dropped).
    policy_retrieval_top_k: int = Field(default=5, ge=1, le=50)
    policy_retrieval_max_statements: int = Field(default=50, ge=1, le=200)
    # Limit on each database statement (Postgres statement_timeout) while building the policy
    # context inside a validation request; a run that hits it is stored as NEEDS_REVIEW (never
    # PASS) with a coverage gap. Per statement, so a build's worst case is a few times this.
    # Must be below DB_TIMEOUT_SECONDS (checked in Settings).
    # At least 1 ms: statement_timeout is in whole milliseconds and 0 would mean no limit.
    policy_context_timeout_seconds: float = Field(default=5, ge=0.001, le=30)


class Settings(RetrievalSettings):
    # Typed configuration. Required values have no default, so a missing one fails at startup.
    # app_env, aws_region, the embedding settings and the retrieval bounds come from
    # RetrievalSettings.

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

    # Image generation (NRC-6, NRC-10). Live mode never falls back to the simulator.
    generation_mode: Literal["simulator", "live"] = "simulator"
    generation_max_active_jobs_per_tenant: int = Field(default=2, ge=1, le=50)
    # Upper bounds on spend limits stop a typo (an extra zero) from lifting the cap entirely.
    generation_daily_credit_ceiling_per_tenant: int = Field(default=20, ge=0, le=1000)
    generation_credits_per_image: int = Field(default=1, ge=1, le=100)
    # A job whose provider estimate is above this fails before anything is submitted.
    # One dollar is ten times the default; a still image costing more is a misconfiguration.
    generation_max_cost_usd_per_image: Decimal = Field(default=Decimal("0.10"), ge=0, le=1)
    # Cap on provider spend across all tenants (reserved estimates plus committed costs), kept per
    # provider in provider_spend. A job that would pass it fails BUDGET_EXHAUSTED before submit.
    # 250 times the default: room to grow, but a typo cannot remove the cap.
    generation_global_budget_usd: Decimal = Field(default=Decimal(40), ge=0, le=10_000)
    # Failed processing attempts per outbox event (a crash, an expired lease, no handler) before
    # the worker gives up. Polls and retryable provider errors don't count: the job deadline
    # bounds those (the agreed retry policy) until the provider accepts the job. After that, the
    # grace window below bounds an unfinished job and output expiry bounds a completed one.
    # Stored on each job when it is accepted, so a change applies to new jobs only.
    generation_max_attempts: int = Field(default=3, ge=1, le=10)
    generation_deadline_seconds: int = Field(default=900, ge=60, le=86_400)
    # How long past its deadline an accepted job may stay queued or in progress at the provider
    # before it is parked as UNKNOWN for review. Does not apply once the provider has completed.
    generation_accepted_grace_seconds: int = Field(default=3600, ge=60, le=86_400)
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
    # Sent with every Higgsfield request; a smoke-script candidate may name its own instead.
    higgsfield_resolution: Literal["720p", "1080p"] = "1080p"
    # "<key-id>:<key-secret>", the Higgsfield SDK's format. Parsed by
    # app.services.providers.higgsfield.higgsfield_credentials.
    higgsfield_api_key: SecretStr | None = None
    # Staging/production: AWS Secrets Manager secret holding the key (e.g. nrc/stg/higgsfield).
    # The worker reads it at startup when HIGGSFIELD_API_KEY is not set.
    higgsfield_secret_id: str | None = None
    higgsfield_output_hosts: Annotated[tuple[str, ...], NoDecode] = ()
    # Downloading one provider output; the worker also caps it at half its lease.
    provider_output_fetch_timeout_seconds: float = Field(default=30, gt=0, le=300)
    # How long a completed output is assumed to stay downloadable when the provider's status
    # gives no expiry, counted from when the worker first saw it completed. Default: 7 days,
    # Higgsfield's documented minimum retention.
    provider_output_retention_fallback_seconds: int = Field(default=604_800, ge=3600, le=1_209_600)

    # Media storage (NRC-3). The bucket comes only from configuration, never from code. The
    # region (AWS_REGION) is in EmbeddingSettings.
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
    # The whole image is held in memory while it is checked and stored: 50 MiB at most.
    max_generated_image_bytes: int = Field(default=10_485_760, ge=1024, le=52_428_800)
    # Decoding allocates about 4 bytes per pixel, so 100 megapixels bounds it near 400 MB.
    max_image_pixels: int = Field(default=25_000_000, ge=1, le=100_000_000)
    # The S3 standard caps presigned links at 15 minutes.
    asset_url_ttl_seconds: int = Field(default=900, ge=60, le=900)

    # Client uploads (NRC-3). Final limits come from the product owner; these are the defaults.
    # Presigned upload URLs share the 15-minute S3 cap above.
    upload_url_ttl_seconds: int = Field(default=600, ge=60, le=900)
    # The worker holds the whole file in memory while inspecting it: 50 MiB at most.
    upload_image_max_bytes: int = Field(default=20_971_520, ge=1024, le=52_428_800)
    # Accepted MIME types, comma-separated in the environment.
    upload_image_types: Annotated[tuple[str, ...], NoDecode] = ("image/png", "image/jpeg")
    # Uploads one tenant may have in progress: waiting in quarantine or inspection, or awaiting
    # their file while the upload URL is still valid. Bounds storage growth until the cleanup job
    # (NRC-3 item 8) exists.
    upload_max_pending_per_tenant: int = Field(default=20, ge=1, le=1000)
    # Defaults to the strict value: a deployment that forgets it gets no simulator. The simulator
    # marks its verdicts as simulated; live needs a real scanner, and the media worker refuses to
    # start without one.
    # Temporary (owner Firoz, NRC-3): "simulator" exists only until Prabhu picks the scanner. When
    # the live scanner lands, keep this setting for local use but add its implementation in
    # app.services.scanning.build_scanner and remove the live-mode refusal test named there.
    malware_scan_mode: Literal["simulator", "live"] = "live"
    # Longest one malware scan may take before the upload waits and is retried
    # (SCAN_UNAVAILABLE). Bounds a scanner that never answers.
    malware_scan_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    # Failed inspection attempts (a crash, an expired lease) before an upload is rejected as
    # INSPECTION_FAILED. Retryable scanner and storage outages are bounded separately.
    media_inspection_max_attempts: int = Field(default=3, ge=1, le=10)
    # How long after completion temporary scanner or storage outages may keep an upload waiting
    # before it is rejected as INSPECTION_FAILED (the uploader can then upload again).
    media_inspection_retry_window_seconds: int = Field(default=86_400, ge=600, le=604_800)

    @field_validator(
        "cognito_client_ids",
        "cors_origins",
        "higgsfield_output_hosts",
        "upload_image_types",
        mode="before",
    )
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @field_validator("upload_image_types")
    @classmethod
    def _upload_types_are_decodable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        types = tuple(item.strip().lower() for item in value)
        if not types:
            raise ValueError("upload_image_types must list at least one MIME type")
        unsupported = sorted(set(types) - SUPPORTED_UPLOAD_IMAGE_TYPES)
        if unsupported:
            raise ValueError(
                f"upload_image_types has unsupported type(s): {', '.join(unsupported)}; "
                f"supported: {', '.join(sorted(SUPPORTED_UPLOAD_IMAGE_TYPES))}"
            )
        return types

    @field_validator("dev_jwks_file", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        # `DEV_JWKS_FILE=` (as in .env.example) means "use Cognito", not the current directory.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _context_timeout_below_db_timeout(self) -> Self:
        # Postgres's statement_timeout must fire first: asyncpg's own timeout (DB_TIMEOUT_SECONDS)
        # raises outside SQLAlchemy and would fail the request instead of recording NEEDS_REVIEW.
        if self.policy_context_timeout_seconds >= self.db_timeout_seconds:
            raise ValueError(
                "POLICY_CONTEXT_TIMEOUT_SECONDS must be below DB_TIMEOUT_SECONDS "
                f"({self.policy_context_timeout_seconds} >= {self.db_timeout_seconds})"
            )
        return self

    @model_validator(mode="after")
    def _shortcuts_only_outside_production(self) -> Self:
        if self.app_env in NON_PRODUCTION_ENVS:
            return self
        enabled = [
            name
            # EMBEDDING_MODE=simulator is refused by EmbeddingSettings' own validator.
            for name, on in (
                ("DEMO_STUBS", self.demo_stubs),
                ("DEV_JWKS_FILE", self.dev_jwks_file),
                ("MALWARE_SCAN_MODE=simulator", self.malware_scan_mode == "simulator"),
            )
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
