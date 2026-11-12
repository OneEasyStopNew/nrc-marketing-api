"""Higgsfield image-generation adapter (NRC-10).

Verified on 9 Nov 2026 against the official docs (docs.higgsfield.ai): llms.txt, quickstart,
authentication index entry, api-reference/headers, api-reference/requests/get-request-status,
concepts/requests, concepts/idempotency, concepts/errors, concepts/rate-limits and openapi.json.
The docs publish no per-model page; the model catalog lives in the account console. The
only text-to-image request schema found is openapi.json "/higgsfield-ai/soul/standard"
(prompt, num_images, resolution 2K|4K, aspect_ratio enum), while the quickstart example uses
"higgsfield-ai/soul/v2/standard" with only a prompt. The model path is therefore configuration
(HIGGSFIELD_MODEL), and the body sent is {"prompt", "aspect_ratio"} plus an optional resolution.

Live run on 9 Nov 2026 with model "higgsfield-ai/soul/v2/standard": aspect_ratio 1:1 was
accepted, and the status response carries no cost field, so the cost recorded for a job is its
/estimate price. The remaining TODO(NRC-10) markers are what a live account must still confirm.
"""

import json
import logging
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Literal
from urllib.parse import urlsplit

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

from app.core.config import Settings
from app.services.fetch import SafeFetcher, build_fetch_client
from app.services.providers.base import (
    HIGGSFIELD_NAME,
    FetchedImage,
    ImageRequest,
    OutputFetcher,
    ProviderConfigError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderOutputError,
    ProviderOutputExpiredError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
    StatusResult,
    SubmitResult,
)

# Type stubs are a dev dependency only; never import them at runtime.
if TYPE_CHECKING:
    from mypy_boto3_secretsmanager import SecretsManagerClient

logger = logging.getLogger(__name__)

# The Secrets Manager secret's JSON entry; the same name as the setting it fills.
KEY_ENTRY_NAME: Final = "HIGGSFIELD_API_KEY"

# Protocol facts from the Higgsfield docs, not tunable: header and id formats.
_IDEMPOTENCY_KEY_RE = re.compile(r"^[\x21-\x7e]{1,255}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CREDENTIAL_RE = re.compile(r"^[\x21-\x7e]+$")
# Sanity bound on the provider's Retry-After; the job deadline (a setting) bounds retries anyway.
_MAX_RETRY_AFTER_SECONDS = 3600.0
# Documented 422 detail when a key is reused with different parameters (concepts/idempotency).
_IDEMPOTENCY_CONFLICT_TEXT = "idempotency-key was already used with different request parameters"

_HiggsfieldStatus = Literal["queued", "in_progress", "completed", "failed", "nsfw", "canceled"]
_STATUS_MAP: Final[Mapping[_HiggsfieldStatus, ProviderJobStatus]] = {
    "queued": ProviderJobStatus.QUEUED,
    "in_progress": ProviderJobStatus.IN_PROGRESS,
    "completed": ProviderJobStatus.COMPLETED,
    "failed": ProviderJobStatus.FAILED,
    "nsfw": ProviderJobStatus.REJECTED,
    "canceled": ProviderJobStatus.CANCELED,
}

# Provider error text is free-form and may echo user input, so only these codes are exposed.
_FAILURE_CODES = {
    ProviderJobStatus.FAILED: "provider_failed",
    ProviderJobStatus.REJECTED: "content_rejected",
}

_CONFIG_STATUSES = frozenset({401, 404})
# concepts/errors documents 403 as "Insufficient balance"; 402 is not documented but means
# Payment Required, so it is treated the same. Both are terminal: nothing was generated.
_BALANCE_STATUSES = frozenset({402, 403})

# Failures where the request provably never left this process.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def _decimal_string(value: object) -> object:
    # /estimate answers {"usd": "<decimal>"}; a JSON number would be a change of contract.
    if not isinstance(value, str):
        raise ValueError("usd must be a decimal string")
    return value


def _utc_if_naive(value: datetime | None) -> datetime | None:
    return value if value is None or value.tzinfo is not None else value.replace(tzinfo=UTC)


def _none_if_unreadable(
    value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> object:
    # An optional field of a status response that cannot be read is dropped instead of failing
    # the whole response: a completed generation is billed and must still be fetched.
    try:
        return handler(value)
    except ValidationError:
        logger.warning("higgsfield status unreadable field=%s", info.field_name)
        return None


_RequestId = Annotated[str, StringConstraints(pattern=_REQUEST_ID_RE.pattern)]
_Usd = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class _SubmitResponse(BaseModel):
    request_id: _RequestId
    status: _HiggsfieldStatus


class _EstimateResponse(BaseModel):
    usd: Annotated[_Usd, BeforeValidator(_decimal_string)]


class _Image(BaseModel):
    url: str


class _StatusResponse(BaseModel):
    status: _HiggsfieldStatus
    images: Annotated[list[_Image] | None, WrapValidator(_none_if_unreadable)] = None
    # TODO(NRC-10): confirm with live account that timestamps always carry an offset; a naive
    # one is read as UTC.
    expires_at: Annotated[
        datetime | None, AfterValidator(_utc_if_naive), WrapValidator(_none_if_unreadable)
    ] = None
    # The live status response (9 Nov 2026) has no cost field, so the estimate is what gets
    # recorded. Should Higgsfield add one, it is read under the /estimate name.
    usd: Annotated[_Usd | None, WrapValidator(_none_if_unreadable)] = None


class HiggsfieldInsufficientBalanceError(ProviderConfigError):
    code: ClassVar[str] = "PROVIDER_INSUFFICIENT_BALANCE"


class HiggsfieldIdempotencyConflictError(ProviderOutcomeUnknownError):
    # The key was already used with a different body, so the original request may be billed.
    code: ClassVar[str] = "PROVIDER_IDEMPOTENCY_CONFLICT"


class HiggsfieldProvider:
    name = HIGGSFIELD_NAME
    is_simulated = False

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        api_key_id: SecretStr,
        api_key_secret: SecretStr,
        fetcher: OutputFetcher,
        allowed_output_hosts: frozenset[str],
        max_output_bytes: int,
        # TODO(NRC-10): confirm with live account whether the model accepts "resolution"
        # (openapi.json lists 2K|4K for soul/standard); None omits it and uses the model default.
        resolution: str | None = None,
    ) -> None:
        key_id = api_key_id.get_secret_value()
        key_secret = api_key_secret.get_secret_value()
        if not _CREDENTIAL_RE.fullmatch(key_id) or ":" in key_id:
            raise ProviderConfigError(
                "HIGGSFIELD_API_KEY key id is empty or contains invalid characters"
            )
        if not _CREDENTIAL_RE.fullmatch(key_secret):
            raise ProviderConfigError(
                "HIGGSFIELD_API_KEY key secret is empty or contains invalid characters"
            )
        parts = urlsplit(base_url)
        if parts.scheme != "https" or not parts.hostname:
            raise ProviderConfigError("HIGGSFIELD_BASE_URL must be an https URL")
        self.model = model.strip("/")
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._authorization = SecretStr(f"Key {key_id}:{key_secret}")
        self._fetcher = fetcher
        self._allowed_hosts = allowed_output_hosts
        self._max_bytes = max_output_bytes
        self._resolution = resolution

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        client: httpx.AsyncClient,
        fetcher: OutputFetcher,
        *,
        model: str | None = None,
        resolution: str | None = None,
    ) -> HiggsfieldProvider:
        # An empty HIGGSFIELD_OUTPUT_HOSTS fails closed (no output host is allowed); live mode
        # refuses to start without it (missing_live_settings).
        model = model or settings.higgsfield_model
        if not model:
            raise ProviderConfigError("Higgsfield is not configured; missing HIGGSFIELD_MODEL")
        try:
            api_key_id, api_key_secret = higgsfield_credentials(settings)
        except ValueError as error:
            raise ProviderConfigError(str(error)) from None
        return cls(
            client=client,
            base_url=settings.higgsfield_base_url,
            model=model,
            api_key_id=api_key_id,
            api_key_secret=api_key_secret,
            fetcher=fetcher,
            allowed_output_hosts=frozenset(h.lower() for h in settings.higgsfield_output_hosts),
            max_output_bytes=settings.max_generated_image_bytes,
            resolution=resolution,
        )

    def __repr__(self) -> str:
        return f"HiggsfieldProvider(model={self.model!r})"

    async def aclose(self) -> None:
        # Closes the API client and the output fetcher's client (worker shutdown).
        await self._client.aclose()
        close = getattr(self._fetcher, "aclose", None)
        if close is not None:
            await close()

    async def submit(self, request: ImageRequest) -> SubmitResult:
        if not _IDEMPOTENCY_KEY_RE.fullmatch(request.idempotency_key):
            raise ProviderRejectedError("Idempotency key must be 1-255 visible ASCII characters")
        response = await self._send(
            "POST",
            f"{self._base_url}/{self.model}",
            json=self._body(request),
            headers={"Idempotency-Key": request.idempotency_key},
            sent_unknown=True,
        )
        self._raise_for_status(response, submit=True)
        try:
            data = _SubmitResponse.model_validate_json(response.content)
        except ValidationError:
            logger.warning(
                "higgsfield submit malformed response status_code=%s", response.status_code
            )
            # Not chained: pydantic errors echo the input.
            raise ProviderOutcomeUnknownError(
                "Higgsfield returned an unreadable submit response"
            ) from None
        status = _STATUS_MAP[data.status]
        logger.info("higgsfield submitted request_id=%s status=%s", data.request_id, status.value)
        return SubmitResult(provider_request_id=data.request_id, status=status)

    def _body(self, request: ImageRequest) -> dict[str, str]:
        body = {"prompt": request.prompt, "aspect_ratio": request.aspect_ratio}
        if self._resolution is not None:
            body["resolution"] = self._resolution
        # The API takes no width/height; sizing is by aspect_ratio (and resolution).
        return body

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        # Not a submission: no Idempotency-Key, and nothing billable can happen.
        response = await self._send(
            "POST",
            f"{self._base_url}/estimate/{self.model}",
            json=self._body(request),
            sent_unknown=False,
        )
        self._raise_for_status(response, submit=False)
        try:
            return _EstimateResponse.model_validate_json(response.content).usd
        except ValidationError:
            raise ProviderUnavailableError(
                "Higgsfield returned an unreadable cost estimate"
            ) from None

    async def get_status(self, provider_request_id: str) -> StatusResult:
        if not _REQUEST_ID_RE.fullmatch(provider_request_id):
            raise ProviderOutputError("Higgsfield request id has an invalid format")
        # The URL is rebuilt from base_url rather than trusting a returned status_url.
        url = f"{self._base_url}/requests/{provider_request_id}/status"
        response = await self._send("GET", url, sent_unknown=False)
        self._raise_for_status(response, submit=False, status_lookup=True)
        try:
            data = _StatusResponse.model_validate_json(response.content)
        except ValidationError:
            logger.warning(
                "higgsfield status malformed request_id=%s status_code=%s",
                provider_request_id,
                response.status_code,
            )
            raise ProviderUnavailableError(
                "Higgsfield returned an unreadable status response"
            ) from None
        status = _STATUS_MAP[data.status]
        urls = tuple(image.url for image in data.images or ())
        if status is ProviderJobStatus.COMPLETED and not urls:
            raise ProviderOutputError("Higgsfield completed the request without an output image")
        logger.info("higgsfield status request_id=%s status=%s", provider_request_id, status.value)
        return StatusResult(
            provider_request_id=provider_request_id,
            status=status,
            output_urls=urls if status is ProviderJobStatus.COMPLETED else (),
            outputs_expire_at=data.expires_at,
            failure_code=_FAILURE_CODES.get(status),
            actual_cost_usd=data.usd,
        )

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        if not result.output_urls:
            raise ProviderOutputError("Higgsfield result has no output image")
        if result.outputs_expire_at is not None and result.outputs_expire_at <= datetime.now(UTC):
            # Billed already; the worker ends the job FAILED OUTPUT_EXPIRED with charges kept.
            raise ProviderOutputExpiredError("Higgsfield output has expired")
        return await self._fetcher(
            result.output_urls[0], allowed_hosts=self._allowed_hosts, max_bytes=self._max_bytes
        )

    async def _send(
        self,
        method: str,
        url: str,
        *,
        sent_unknown: bool,
        json: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        all_headers = {"Authorization": self._authorization.get_secret_value()}
        all_headers.update(headers or {})
        try:
            return await self._client.request(method, url, json=json, headers=all_headers)
        # `from None` drops the httpx chain, whose text can embed the URL.
        except _NOT_SENT_ERRORS:
            raise ProviderUnavailableError("Could not connect to Higgsfield") from None
        except UnicodeError, httpx.InvalidURL:
            raise ProviderConfigError(
                "Higgsfield request could not be built; check HIGGSFIELD_* settings"
            ) from None
        except httpx.TransportError:
            if sent_unknown:
                raise ProviderOutcomeUnknownError(
                    "Higgsfield connection failed after the request was sent"
                ) from None
            raise ProviderUnavailableError("Higgsfield status request failed") from None

    def _raise_for_status(
        self, response: httpx.Response, *, submit: bool, status_lookup: bool = False
    ) -> None:
        code = response.status_code
        if 200 <= code < 300:
            return
        logger.warning("higgsfield error status_code=%s", code)
        if code == 429:
            raise ProviderRateLimitedError(
                "Higgsfield rate limit reached", retry_after_seconds=_retry_after(response)
            )
        if code == 423:
            raise ProviderUnavailableError("Higgsfield model is temporarily blocked (HTTP 423)")
        if code in _BALANCE_STATUSES:
            raise HiggsfieldInsufficientBalanceError(
                f"Higgsfield account has insufficient balance or the key is disabled (HTTP {code})"
            )
        if code == 404 and status_lookup:
            raise ProviderConfigError("Higgsfield request was not found or has expired (HTTP 404)")
        if code == 422 and _IDEMPOTENCY_CONFLICT_TEXT in response.text.lower():
            raise HiggsfieldIdempotencyConflictError(
                "Higgsfield idempotency key was reused with different parameters"
            )
        if code in _CONFIG_STATUSES:
            raise ProviderConfigError(
                f"Higgsfield rejected the account or model configuration (HTTP {code})"
            )
        if code >= 500:
            if submit:
                raise ProviderOutcomeUnknownError(f"Higgsfield server error (HTTP {code})")
            raise ProviderUnavailableError(f"Higgsfield server error (HTTP {code})")
        raise ProviderRejectedError(f"Higgsfield rejected the request (HTTP {code})")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is not None:
        try:
            seconds = float(value)
        except ValueError:
            # HTTP-date form is not documented for Higgsfield; fall through to the body.
            seconds = -1.0
        if math.isfinite(seconds) and seconds >= 0:
            return min(seconds, _MAX_RETRY_AFTER_SECONDS)
    try:
        detail = response.json().get("detail")
        body_value = detail.get("retry_after_seconds") if isinstance(detail, dict) else None
    except ValueError, AttributeError:
        return None
    if (
        isinstance(body_value, int | float)
        and not isinstance(body_value, bool)
        and math.isfinite(body_value)
        and body_value >= 0
    ):
        return min(float(body_value), _MAX_RETRY_AFTER_SECONDS)
    return None


def build_api_client(settings: Settings) -> httpx.AsyncClient:
    """HTTP client for the Higgsfield API: https only, no proxies from the environment.

    Its timeout must end every call before the worker's lease on the job expires.
    """
    timeout = settings.http_timeout_seconds
    if timeout >= settings.worker_lease_seconds:
        raise ProviderConfigError("HTTP_TIMEOUT_SECONDS must be below WORKER_LEASE_SECONDS")
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        trust_env=False,
        follow_redirects=False,
        event_hooks={"request": [_require_https]},
    )


async def _require_https(request: httpx.Request) -> None:
    # Runs before anything is sent, so the key never travels over plain http.
    if request.url.scheme != "https":
        raise ProviderConfigError("Higgsfield requests must use https")


def create_provider(
    settings: Settings, *, model: str | None = None, resolution: str | None = None
) -> HiggsfieldProvider:
    """The live provider, with its own API client and SSRF-safe output fetcher.

    The worker calls this with settings only (connector.build_providers); the smoke script
    picks the model and resolution. Close it with aclose().
    """
    # Output downloads run under the worker's lease too.
    fetch_timeout = min(
        settings.provider_output_fetch_timeout_seconds, settings.worker_lease_seconds / 2
    )
    return HiggsfieldProvider.from_settings(
        settings,
        build_api_client(settings),
        SafeFetcher(build_fetch_client(), timeout_seconds=fetch_timeout),
        model=model,
        resolution=resolution,
    )


def higgsfield_credentials(settings: Settings) -> tuple[SecretStr, SecretStr]:
    """Split HIGGSFIELD_API_KEY into (key id, key secret) at the first ':'.

    Not a field validator, because pydantic echoes input values in its errors.

    Raises:
        ValueError: HIGGSFIELD_API_KEY is missing or malformed. The message never contains
            the value.
    """
    key = settings.higgsfield_api_key
    raw = key.get_secret_value() if key else ""
    key_id, separator, key_secret = raw.partition(":")
    if not separator or not key_id.strip() or not key_secret.strip():
        raise ValueError("HIGGSFIELD_API_KEY is missing or not in the form <key-id>:<key-secret>.")
    return SecretStr(key_id), SecretStr(key_secret)


def missing_live_settings(settings: Settings) -> list[str]:
    # What live mode lacks for Higgsfield; messages name settings, never values.
    problems: list[str] = []
    try:
        higgsfield_credentials(settings)
    except ValueError as error:
        problems.append(str(error))
    missing = [
        name
        for name, value in (
            ("HIGGSFIELD_MODEL", settings.higgsfield_model),
            ("HIGGSFIELD_OUTPUT_HOSTS", settings.higgsfield_output_hosts),
        )
        if not value
    ]
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    return problems


# Staging and production: the key comes from AWS Secrets Manager at worker startup.


class ProviderSecretError(RuntimeError):
    # Messages name the secret id and JSON keys, never the secret's content.
    pass


def with_higgsfield_key(settings: Settings, client: SecretsManagerClient | None = None) -> Settings:
    """Return settings that hold the Higgsfield key, reading it from Secrets Manager if needed.

    Only when HIGGSFIELD_SECRET_ID is set and HIGGSFIELD_API_KEY is not. The key stays in memory
    (a copy of the frozen settings); it is never written anywhere or logged. On servers the
    default credential chain (the pod's IAM role) authorizes the read; no access keys are used.

    Raises:
        ProviderSecretError: the secret cannot be read, is not JSON, or has no usable entry.
    """
    secret_id = settings.higgsfield_secret_id
    # An empty HIGGSFIELD_API_KEY= line in .env counts as not set.
    key = settings.higgsfield_api_key
    if not secret_id or (key is not None and key.get_secret_value()):
        return settings
    if client is None:
        client = boto3.Session().client("secretsmanager", region_name=settings.aws_region)
    try:
        response = client.get_secret_value(SecretId=secret_id)
    except BotoCoreError, ClientError:
        # Not chained: botocore messages can include request details.
        raise ProviderSecretError(f"Could not read secret {secret_id}.") from None
    try:
        data = json.loads(response.get("SecretString") or "")
    except ValueError:
        raise ProviderSecretError(f"Secret {secret_id} is not a JSON object.") from None
    if not isinstance(data, dict):
        raise ProviderSecretError(f"Secret {secret_id} is not a JSON object.")
    value = data.get(KEY_ENTRY_NAME)
    if not isinstance(value, str) or not value:
        raise ProviderSecretError(f"Secret {secret_id} has no {KEY_ENTRY_NAME} entry.")
    logger.info("higgsfield key loaded from secret=%s", secret_id)
    return settings.model_copy(update={"higgsfield_api_key": SecretStr(value)})
