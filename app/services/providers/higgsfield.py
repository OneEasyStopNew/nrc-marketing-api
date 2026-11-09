"""Higgsfield image-generation adapter (NRC-10).

Verified on 9 Nov 2026 against the official docs (docs.higgsfield.ai): llms.txt, quickstart,
authentication index entry, api-reference/headers, api-reference/requests/get-request-status,
concepts/requests, concepts/idempotency, concepts/errors, concepts/rate-limits and openapi.json.
The docs publish no per-model page; the model catalog lives in the account console. The
only text-to-image request schema found is openapi.json "/higgsfield-ai/soul/standard"
(prompt, num_images, resolution 2K|4K, aspect_ratio enum), while the quickstart example uses
"higgsfield-ai/soul/v2/standard" with only a prompt. The model path is therefore configuration
(HIGGSFIELD_MODEL), and the body sent is {"prompt", "aspect_ratio"} plus an optional resolution.
No live call has been made; see the TODO(NRC-10) markers for what a live account must confirm.
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import SecretStr

from app.core.config import Settings
from app.services.providers.base import (
    FetchedImage,
    ImageRequest,
    OutputFetcher,
    ProviderConfigError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderOutputError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
    StatusResult,
    SubmitResult,
)

logger = logging.getLogger(__name__)

_IDEMPOTENCY_KEY_RE = re.compile(r"^[\x21-\x7e]{1,255}$")

_STATUS_MAP = {
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

_CONFIG_STATUSES = frozenset({401, 403, 404, 423})

# Failures where the request provably never left this process.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class HiggsfieldProvider:
    name = "higgsfield"
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
        self.model = model.strip("/")
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._authorization = SecretStr(
            f"Key {api_key_id.get_secret_value()}:{api_key_secret.get_secret_value()}"
        )
        self._fetcher = fetcher
        self._allowed_hosts = allowed_output_hosts
        self._max_bytes = max_output_bytes
        self._resolution = resolution

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient, fetcher: OutputFetcher
    ) -> HiggsfieldProvider:
        missing = [
            env
            for env, value in (
                ("HIGGSFIELD_MODEL", settings.higgsfield_model),
                ("HIGGSFIELD_API_KEY_ID", settings.higgsfield_api_key_id),
                ("HIGGSFIELD_API_KEY_SECRET", settings.higgsfield_api_key_secret),
                ("HIGGSFIELD_OUTPUT_HOSTS", settings.higgsfield_output_hosts),
            )
            if not value
        ]
        if missing:
            raise ProviderConfigError(f"Higgsfield is not configured; missing {', '.join(missing)}")
        assert settings.higgsfield_model is not None  # noqa: S101 (narrowing for mypy)
        assert settings.higgsfield_api_key_id is not None  # noqa: S101
        assert settings.higgsfield_api_key_secret is not None  # noqa: S101
        return cls(
            client=client,
            base_url=settings.higgsfield_base_url,
            model=settings.higgsfield_model,
            api_key_id=settings.higgsfield_api_key_id,
            api_key_secret=settings.higgsfield_api_key_secret,
            fetcher=fetcher,
            allowed_output_hosts=frozenset(h.lower() for h in settings.higgsfield_output_hosts),
            max_output_bytes=settings.max_generated_image_bytes,
        )

    def __repr__(self) -> str:
        return f"HiggsfieldProvider(model={self.model!r})"

    async def submit(self, request: ImageRequest) -> SubmitResult:
        if not _IDEMPOTENCY_KEY_RE.fullmatch(request.idempotency_key):
            raise ProviderRejectedError("Idempotency key must be 1-255 visible ASCII characters")
        body: dict[str, str] = {"prompt": request.prompt, "aspect_ratio": request.aspect_ratio}
        if self._resolution is not None:
            body["resolution"] = self._resolution
        # The API takes no width/height; sizing is by aspect_ratio (and resolution).
        # TODO(NRC-10): confirm with live account that aspect_ratio values from the job are valid
        # for the pinned model (documented enum: 1:1 4:3 3:4 3:2 2:3 5:4 4:5 16:9 9:16 21:9).
        response = await self._send(
            "POST",
            f"{self._base_url}/{self.model}",
            json=body,
            headers={"Idempotency-Key": request.idempotency_key},
            sent_unknown=True,
        )
        self._raise_for_status(response, submit=True)
        data = self._json_object(response, submit=True)
        request_id = data.get("request_id")
        status = _STATUS_MAP.get(str(data.get("status")))
        if not isinstance(request_id, str) or not request_id or status is None:
            logger.warning(
                "higgsfield submit malformed response status_code=%s", response.status_code
            )
            raise ProviderOutcomeUnknownError("Higgsfield returned an unreadable submit response")
        logger.info("higgsfield submitted request_id=%s status=%s", request_id, status.value)
        return SubmitResult(provider_request_id=request_id, status=status)

    async def get_status(self, provider_request_id: str) -> StatusResult:
        # The URL is rebuilt from base_url rather than trusting a returned status_url.
        url = f"{self._base_url}/requests/{quote(provider_request_id, safe='')}/status"
        response = await self._send("GET", url, sent_unknown=False)
        self._raise_for_status(response, submit=False)
        data = self._json_object(response, submit=False)
        status = _STATUS_MAP.get(str(data.get("status")))
        if status is None:
            logger.warning(
                "higgsfield status malformed request_id=%s status_code=%s",
                provider_request_id,
                response.status_code,
            )
            raise ProviderUnavailableError("Higgsfield returned an unreadable status response")
        urls = _image_urls(data.get("images"))
        expires_at = _parse_expiry(data.get("expires_at"))
        logger.info("higgsfield status request_id=%s status=%s", provider_request_id, status.value)
        return StatusResult(
            provider_request_id=provider_request_id,
            status=status,
            output_urls=urls if status is ProviderJobStatus.COMPLETED else (),
            outputs_expire_at=expires_at,
            failure_code=_FAILURE_CODES.get(status),
        )

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        if not result.output_urls:
            raise ProviderOutputError("Higgsfield result has no output image")
        if result.outputs_expire_at is not None and result.outputs_expire_at <= datetime.now(UTC):
            raise ProviderOutputError("Higgsfield output has expired")
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
        except httpx.TransportError:
            if sent_unknown:
                raise ProviderOutcomeUnknownError(
                    "Higgsfield connection failed after the request was sent"
                ) from None
            raise ProviderUnavailableError("Higgsfield status request failed") from None

    def _raise_for_status(self, response: httpx.Response, *, submit: bool) -> None:
        code = response.status_code
        if 200 <= code < 300:
            return
        logger.warning("higgsfield error status_code=%s", code)
        if code == 429:
            raise ProviderRateLimitedError(
                "Higgsfield rate limit reached", retry_after_seconds=_retry_after(response)
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

    @staticmethod
    def _json_object(response: httpx.Response, *, submit: bool) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        if submit:
            raise ProviderOutcomeUnknownError("Higgsfield returned an unreadable submit response")
        raise ProviderUnavailableError("Higgsfield returned an unreadable status response")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is not None:
        try:
            seconds = float(value)
        except ValueError:
            # HTTP-date form is not documented for Higgsfield; fall through to the body.
            seconds = -1.0
        if seconds >= 0:
            return seconds
    try:
        detail = response.json().get("detail")
        body_value = detail.get("retry_after_seconds") if isinstance(detail, dict) else None
    except ValueError, AttributeError:
        return None
    if isinstance(body_value, int | float) and not isinstance(body_value, bool) and body_value >= 0:
        return float(body_value)
    return None


def _image_urls(images: object) -> tuple[str, ...]:
    if not isinstance(images, list):
        return ()
    return tuple(
        item["url"]
        for item in images
        if isinstance(item, dict) and isinstance(item.get("url"), str)
    )


def _parse_expiry(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    # TODO(NRC-10): confirm with live account that timestamps always carry an offset.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
