# Contract between the connector worker and image-generation providers (simulator, Higgsfield).

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Protocol


class ProviderJobStatus(StrEnum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    # Content rejected by the provider (Higgsfield reports "nsfw").
    REJECTED = "rejected"
    CANCELED = "canceled"


TERMINAL_PROVIDER_STATUSES = frozenset(
    {
        ProviderJobStatus.COMPLETED,
        ProviderJobStatus.FAILED,
        ProviderJobStatus.REJECTED,
        ProviderJobStatus.CANCELED,
    }
)


@dataclass(frozen=True, slots=True)
class ImageRequest:
    prompt: str
    width_px: int
    height_px: int
    aspect_ratio: str
    # Stable per job and resent unchanged on every retry, so the provider never bills twice.
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class SubmitResult:
    provider_request_id: str
    status: ProviderJobStatus


@dataclass(frozen=True, slots=True)
class StatusResult:
    provider_request_id: str
    status: ProviderJobStatus
    output_urls: tuple[str, ...] = ()
    outputs_expire_at: datetime | None = None
    # Safe, provider-supplied reason for FAILED or REJECTED. Never contains credentials.
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class FetchedImage:
    content: bytes
    content_type: str
    # Where the bytes came from (provenance). None for the simulator. Never a signed URL.
    source_host: str | None


class OutputFetcher(Protocol):
    """Downloads one provider output URL with host allowlist, size cap and SSRF protection."""

    def __call__(
        self, url: str, *, allowed_hosts: frozenset[str], max_bytes: int
    ) -> Awaitable[FetchedImage]: ...


class ImageProvider(Protocol):
    """One image model behind one provider account. Implementations hold no per-job state."""

    name: str
    model: str
    is_simulated: bool

    async def submit(self, request: ImageRequest) -> SubmitResult:
        """Submit a generation. Raises a ProviderError subclass on failure."""
        ...

    async def get_status(self, provider_request_id: str) -> StatusResult: ...

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        """Download the first output of a COMPLETED result."""
        ...


class ProviderError(Exception):
    # Base for classified provider failures. Messages are safe to store and show.

    code: ClassVar[str] = "PROVIDER_ERROR"
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class ProviderConfigError(ProviderError):
    # Bad credentials, model not enabled for the account (401/404/423). Fix config, don't retry.
    code = "PROVIDER_CONFIG"


class ProviderRejectedError(ProviderError):
    # Invalid parameters or content rejected before any billable work. Not retryable.
    code = "PROVIDER_REJECTED"


class ProviderRateLimitedError(ProviderError):
    code = "PROVIDER_RATE_LIMITED"
    retryable = True


class ProviderUnavailableError(ProviderError):
    # The request certainly did not reach the provider (connect failure, 5xx before acceptance).
    code = "PROVIDER_UNAVAILABLE"
    retryable = True


class ProviderOutcomeUnknownError(ProviderError):
    # The request may have been accepted (timeout after sending, lost response). The job becomes
    # UNKNOWN and is reconciled by resending the same request with the same idempotency key.
    code = "PROVIDER_OUTCOME_UNKNOWN"


class ProviderOutputError(ProviderError):
    # The output could not be fetched safely (host not allowed, too large, wrong type, expired).
    code = "PROVIDER_OUTPUT_INVALID"
