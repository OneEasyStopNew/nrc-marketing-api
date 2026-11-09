# Higgsfield adapter tests: httpx.MockTransport only, no network.

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.services.providers.base import (
    FetchedImage,
    ImageRequest,
    ProviderConfigError,
    ProviderError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderOutputError,
    ProviderRateLimitedError,
    ProviderRejectedError,
    ProviderUnavailableError,
    StatusResult,
)
from app.services.providers.higgsfield import HiggsfieldProvider

KEY_ID = "kid-SECRETID-123"
KEY_SECRET = "ksecret-SECRETVALUE-456"  # noqa: S105
PROMPT = "a confidential launch visual for Acme"
OUTPUT_URL = "https://cdn.example.com/signed/abc?sig=SIGNEDTOKEN"
REQUEST_ID = "d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"
MODEL = "higgsfield-ai/soul/standard"
Handler = Callable[[httpx.Request], httpx.Response]


class RecordingFetcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, frozenset[str], int]] = []

    async def __call__(
        self, url: str, *, allowed_hosts: frozenset[str], max_bytes: int
    ) -> FetchedImage:
        self.calls.append((url, allowed_hosts, max_bytes))
        return FetchedImage(content=b"img", content_type="image/png", source_host="cdn.example.com")


def make(handler: Handler, fetcher: RecordingFetcher | None = None) -> HiggsfieldProvider:
    return HiggsfieldProvider(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://api.higgsfield.ai/",
        model=MODEL,
        api_key_id=SecretStr(KEY_ID),
        api_key_secret=SecretStr(KEY_SECRET),
        fetcher=fetcher or RecordingFetcher(),
        allowed_output_hosts=frozenset({"cdn.example.com"}),
        max_output_bytes=1234,
    )


def request(key: str = "job-key-1") -> ImageRequest:
    return ImageRequest(
        prompt=PROMPT, width_px=1024, height_px=768, aspect_ratio="4:3", idempotency_key=key
    )


def accepted(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "queued", "request_id": REQUEST_ID})


def fixed(status: int, **kwargs: Any) -> Handler:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **kwargs)

    return handler


def raising(exc: Exception) -> Handler:
    def handler(_: httpx.Request) -> httpx.Response:
        raise exc

    return handler


async def test_submit_sends_documented_request_and_parses() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return accepted(req)

    result = await make(handler).submit(request())
    assert result.provider_request_id == REQUEST_ID
    assert result.status is ProviderJobStatus.QUEUED
    sent = seen[0]
    assert sent.method == "POST"
    assert str(sent.url) == f"https://api.higgsfield.ai/{MODEL}"
    assert sent.headers["Authorization"] == f"Key {KEY_ID}:{KEY_SECRET}"
    assert sent.headers["Idempotency-Key"] == "job-key-1"
    assert json.loads(sent.content) == {"prompt": PROMPT, "aspect_ratio": "4:3"}


async def test_same_key_resend_returns_same_request_id() -> None:
    keys: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        keys.append(req.headers["Idempotency-Key"])
        return accepted(req)

    provider = make(handler)
    first = await provider.submit(request())
    second = await provider.submit(request())
    assert first.provider_request_id == second.provider_request_id
    assert keys == ["job-key-1", "job-key-1"]


async def test_invalid_idempotency_key_is_rejected_without_a_call() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be sent")

    with pytest.raises(ProviderRejectedError):
        await make(handler).submit(request("has space"))


@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("boom"), httpx.ConnectTimeout("boom"), httpx.PoolTimeout("boom")]
)
async def test_submit_not_sent_is_unavailable(exc: Exception) -> None:
    with pytest.raises(ProviderUnavailableError):
        await make(raising(exc)).submit(request())


@pytest.mark.parametrize(
    "exc", [httpx.ReadTimeout("boom"), httpx.RemoteProtocolError("boom"), httpx.ReadError("boom")]
)
async def test_submit_lost_after_send_is_outcome_unknown(exc: Exception) -> None:
    with pytest.raises(ProviderOutcomeUnknownError):
        await make(raising(exc)).submit(request())


@pytest.mark.parametrize(
    "handler",
    [
        fixed(200, content=b"not json"),
        fixed(200, json={"status": "queued"}),
        fixed(200, json={"status": "weird", "request_id": "x"}),
        fixed(200, json=["list"]),
        fixed(500, json={"detail": "x"}),
        fixed(503, json={"detail": "x"}),
    ],
    ids=["bad-json", "no-id", "unknown-status", "list-body", "http-500", "http-503"],
)
async def test_submit_malformed_or_5xx_is_outcome_unknown(handler: Handler) -> None:
    with pytest.raises(ProviderOutcomeUnknownError):
        await make(handler).submit(request())


@pytest.mark.parametrize("code", [401, 403, 404, 423])
async def test_config_statuses(code: int) -> None:
    with pytest.raises(ProviderConfigError):
        await make(fixed(code, json={"detail": "nope"})).submit(request())
    with pytest.raises(ProviderConfigError):
        await make(fixed(code, json={"detail": "nope"})).get_status(REQUEST_ID)


@pytest.mark.parametrize("code", [400, 422])
async def test_rejected_statuses(code: int) -> None:
    detail = "Idempotency-Key was already used with different request parameters."
    with pytest.raises(ProviderRejectedError):
        await make(fixed(code, json={"detail": detail})).submit(request())


async def test_rate_limited_uses_retry_after_header() -> None:
    handler = fixed(429, headers={"Retry-After": "30"}, json={"detail": "x"})
    with pytest.raises(ProviderRateLimitedError) as info:
        await make(handler).submit(request())
    assert info.value.retry_after_seconds == 30


async def test_rate_limited_falls_back_to_body_then_none() -> None:
    body = {"detail": {"code": "account_queue_full", "retry_after_seconds": 12}}
    with pytest.raises(ProviderRateLimitedError) as info:
        await make(fixed(429, json=body)).submit(request())
    assert info.value.retry_after_seconds == 12
    with pytest.raises(ProviderRateLimitedError) as info2:
        await make(fixed(429, content=b"")).submit(request())
    assert info2.value.retry_after_seconds is None


async def test_status_5xx_and_transport_errors_are_unavailable() -> None:
    with pytest.raises(ProviderUnavailableError):
        await make(fixed(503, json={"detail": "x"})).get_status(REQUEST_ID)
    with pytest.raises(ProviderUnavailableError):
        await make(raising(httpx.ReadTimeout("boom"))).get_status(REQUEST_ID)
    with pytest.raises(ProviderUnavailableError):
        await make(fixed(200, content=b"nope")).get_status(REQUEST_ID)


async def test_status_url_is_derived_from_base_url() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "status": "in_progress",
                "request_id": REQUEST_ID,
                "status_url": "https://evil.example.net/steal",
            },
        )

    result = await make(handler).get_status(REQUEST_ID)
    assert result.status is ProviderJobStatus.IN_PROGRESS
    assert str(seen[0].url) == f"https://api.higgsfield.ai/requests/{REQUEST_ID}/status"
    assert seen[0].headers["Authorization"] == f"Key {KEY_ID}:{KEY_SECRET}"


async def test_status_request_id_cannot_escape_path() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"status": "queued", "request_id": "x"})

    await make(handler).get_status("../other?x=1")
    assert seen[0].url.host == "api.higgsfield.ai"
    assert seen[0].url.query == b""


async def test_nsfw_and_failed_mapping() -> None:
    nsfw = await make(fixed(200, json={"status": "nsfw", "request_id": "r"})).get_status("r")
    assert nsfw.status is ProviderJobStatus.REJECTED
    assert nsfw.failure_code == "content_rejected"
    body = {"status": "failed", "request_id": "r", "error": f"bad prompt {PROMPT}"}
    failed = await make(fixed(200, json=body)).get_status("r")
    assert failed.status is ProviderJobStatus.FAILED
    assert PROMPT not in (failed.failure_code or "")
    canceled = await make(fixed(200, json={"status": "canceled", "request_id": "r"})).get_status(
        "r"
    )
    assert canceled.status is ProviderJobStatus.CANCELED


async def test_completed_parses_urls_and_aware_expiry_then_fetches() -> None:
    expires = (datetime.now(UTC) + timedelta(days=7)).replace(microsecond=0)
    body = {
        "status": "completed",
        "request_id": REQUEST_ID,
        "images": [{"url": OUTPUT_URL}, {"url": "https://cdn.example.com/second.png"}],
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
    }
    fetcher = RecordingFetcher()
    provider = make(fixed(200, json=body), fetcher)
    result = await provider.get_status(REQUEST_ID)
    assert result.status is ProviderJobStatus.COMPLETED
    assert result.output_urls[0] == OUTPUT_URL
    assert result.outputs_expire_at == expires
    assert result.outputs_expire_at.tzinfo is not None
    image = await provider.fetch_output(result)
    assert image.content == b"img"
    assert fetcher.calls == [(OUTPUT_URL, frozenset({"cdn.example.com"}), 1234)]


async def test_naive_expiry_is_made_aware() -> None:
    body = {
        "status": "completed",
        "request_id": "r",
        "images": [],
        "expires_at": "2030-01-01T00:00:00",
    }
    result = await make(fixed(200, json=body)).get_status("r")
    assert result.outputs_expire_at == datetime(2030, 1, 1, tzinfo=UTC)


async def test_fetch_output_without_images_or_after_expiry_fails() -> None:
    provider = make(accepted)
    with pytest.raises(ProviderOutputError):
        await provider.fetch_output(StatusResult("r", ProviderJobStatus.COMPLETED))
    expired = StatusResult(
        "r",
        ProviderJobStatus.COMPLETED,
        output_urls=(OUTPUT_URL,),
        outputs_expire_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    with pytest.raises(ProviderOutputError):
        await provider.fetch_output(expired)


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "postgresql+asyncpg://u:p@localhost/db",
        "cognito_region": "ap-south-1",
        "cognito_user_pool_id": "ap-south-1_TESTPOOL",
        "cognito_client_ids": ("c",),
        "required_scope": "marketing/api",
        "higgsfield_model": MODEL,
        "higgsfield_api_key_id": KEY_ID,
        "higgsfield_api_key_secret": KEY_SECRET,
        "higgsfield_output_hosts": ("CDN.example.com",),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_from_settings_builds_provider() -> None:
    provider = HiggsfieldProvider.from_settings(settings(), httpx.AsyncClient(), RecordingFetcher())
    assert provider.name == "higgsfield"
    assert provider.is_simulated is False
    assert provider.model == MODEL
    assert KEY_SECRET not in repr(provider)


@pytest.mark.parametrize(
    ("override", "missing"),
    [
        ({"higgsfield_model": None}, "HIGGSFIELD_MODEL"),
        ({"higgsfield_api_key_id": None}, "HIGGSFIELD_API_KEY_ID"),
        ({"higgsfield_api_key_secret": None}, "HIGGSFIELD_API_KEY_SECRET"),
        ({"higgsfield_output_hosts": ()}, "HIGGSFIELD_OUTPUT_HOSTS"),
    ],
)
def test_from_settings_names_missing_setting(override: dict[str, Any], missing: str) -> None:
    with pytest.raises(ProviderConfigError) as info:
        HiggsfieldProvider.from_settings(
            settings(**override), httpx.AsyncClient(), RecordingFetcher()
        )
    assert missing in str(info.value)
    assert KEY_ID not in str(info.value)
    assert KEY_SECRET not in str(info.value)


async def test_secrets_prompts_and_urls_never_leak(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    forbidden = (KEY_ID, KEY_SECRET, "Key ", PROMPT, OUTPUT_URL, "SIGNEDTOKEN")
    cases: list[Handler] = [
        fixed(401, json={"detail": f"bad {KEY_SECRET} {PROMPT}"}),
        fixed(422, json={"detail": PROMPT}),
        fixed(429, headers={"Retry-After": "5"}),
        fixed(500, text=OUTPUT_URL),
        fixed(200, text=OUTPUT_URL),
        raising(httpx.ReadTimeout(f"{KEY_SECRET} {PROMPT} {OUTPUT_URL}")),
        raising(httpx.ConnectError(f"{KEY_ID} {OUTPUT_URL}")),
    ]
    errors: list[ProviderError] = []
    for handler in cases:
        with pytest.raises(ProviderError) as info:
            await make(handler).submit(request())
        errors.append(info.value)
        assert info.value.__cause__ is None
    with pytest.raises(ProviderError) as status_info:
        await make(fixed(500, text=PROMPT)).get_status(REQUEST_ID)
    errors.append(status_info.value)
    await make(accepted).submit(request())
    for exc in errors:
        for item in forbidden:
            assert item not in str(exc)
            assert item not in repr(exc)
    for item in forbidden:
        assert item not in caplog.text
