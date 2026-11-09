"""The connector worker with the real HiggsfieldProvider against a fake Higgsfield API.

The fake runs behind httpx.MockTransport: nothing leaves the process.
"""

import io
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import boto3
import httpx
import pytest
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import bind_tenant
from app.models import Asset
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState, SpendState
from app.services.assets import StoredImageSink
from app.services.fetch import SafeFetcher
from app.services.providers.base import FetchedImage, OutputFetcher, ProviderOutputError
from app.services.providers.higgsfield import HiggsfieldProvider
from app.services.storage import ObjectStorage, StoredObject
from app.workers.connector import GeneratedImageSink, run_once
from tests.conftest import AuthHeaders, World
from tests.test_job_worker import _event, _load, _requeue, _spend
from tests.test_jobs_api import _creative, _post_job
from tests.test_storage import needs_seaweedfs

Sessions = async_sessionmaker[AsyncSession]
MODEL = "higgsfield-ai/soul/standard"
ESTIMATE_USD = "0.05"
OUTPUT_HOST = "images.higgsfield.example"
OUTPUT_URL = f"https://{OUTPUT_HOST}/outputs/result.png?signature=SIGNED"
CONFLICT_DETAIL = "Idempotency-Key was already used with different request parameters."
# One scripted answer to a submit: given the fake and the request, return the response.
Answer = Callable[["FakeHiggsfield", httpx.Request], httpx.Response]


class FakeHiggsfield:
    """Stands in for api.higgsfield.ai. Like the real API, it remembers a request per key.

    `accepted` holds the keys it accepted (and so would charge for), with their request ids.
    """

    def __init__(self) -> None:
        self.submit_answers: list[Answer] = []
        self.submit_keys: list[str] = []
        self.accepted: dict[str, str] = {}
        self.status_body: Callable[[str], dict[str, Any]] = lambda request_id: {
            "status": "in_progress",
            "request_id": request_id,
        }

    def accept(self, request: httpx.Request) -> httpx.Response:
        key = request.headers["Idempotency-Key"]
        request_id = self.accepted.setdefault(key, str(uuid4()))
        return httpx.Response(200, json={"status": "queued", "request_id": request_id})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.scheme == "https"
        assert request.headers["Authorization"] == "Key kid-test:ksecret-test"
        path = request.url.path
        if request.method == "POST" and path == f"/estimate/{MODEL}":
            return httpx.Response(200, json={"usd": ESTIMATE_USD})
        if request.method == "POST" and path == f"/{MODEL}":
            self.submit_keys.append(request.headers["Idempotency-Key"])
            if self.submit_answers:
                return self.submit_answers.pop(0)(self, request)
            return self.accept(request)
        if request.method == "GET" and path.startswith("/requests/"):
            request_id = path.split("/")[2]
            assert request_id in self.accepted.values()
            return httpx.Response(200, json=self.status_body(request_id))
        return httpx.Response(404, json={"detail": "Not found"})


def status(code: int, **kwargs: Any) -> Answer:
    def answer(_: FakeHiggsfield, __: httpx.Request) -> httpx.Response:
        return httpx.Response(code, **kwargs)

    return answer


def raising(error: Exception) -> Answer:
    def answer(_: FakeHiggsfield, __: httpx.Request) -> httpx.Response:
        raise error

    return answer


def accept_then(code: int) -> Answer:
    # The request is accepted (and would be charged), but the answer is an error.
    def answer(fake: FakeHiggsfield, request: httpx.Request) -> httpx.Response:
        fake.accept(request)
        return httpx.Response(code, json={"detail": "Internal server error"})

    return answer


async def _no_fetch(url: str, *, allowed_hosts: frozenset[str], max_bytes: int) -> FetchedImage:
    raise ProviderOutputError("Not fetched in this test")


def higgsfield(
    fake: FakeHiggsfield,
    fetcher: OutputFetcher = _no_fetch,
    output_hosts: frozenset[str] = frozenset(),
) -> HiggsfieldProvider:
    return HiggsfieldProvider(
        client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
        base_url="https://api.higgsfield.ai",
        model=MODEL,
        api_key_id=SecretStr("kid-test"),
        api_key_secret=SecretStr("ksecret-test"),
        fetcher=fetcher,
        allowed_output_hosts=output_hosts,
        max_output_bytes=get_settings().max_generated_image_bytes,
    )


@pytest.fixture
async def job_id(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> UUID:
    # Accepted in simulator mode, then pointed at Higgsfield as a live-mode job would be.
    creative_id = await _creative(client, auth, world.a)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text("UPDATE jobs SET provider = 'higgsfield', model = :model"), {"model": MODEL}
    )
    return UUID(response.json()["job_id"])


async def run(
    sessions: Sessions, provider: HiggsfieldProvider, sink: GeneratedImageSink | None = None
) -> int:
    return await run_once(sessions, {"higgsfield": provider}, sink, get_settings())


FAIL_NOW = (ExecutionState.FAILED, "submitting", ReservationState.RELEASED, True)
RETRY = (ExecutionState.RUNNING, "retry_wait", ReservationState.RESERVED, False)
RECONCILE = (ExecutionState.UNKNOWN, "reconcile", ReservationState.HELD, False)
REVIEW = (ExecutionState.UNKNOWN, "reconcile", ReservationState.HELD, True)


@pytest.mark.parametrize(
    ("answer", "code", "outcome"),
    [
        (status(400, json={"detail": "Invalid"}), "PROVIDER_REJECTED", FAIL_NOW),
        (status(401, json={"detail": "No"}), "PROVIDER_CONFIG", FAIL_NOW),
        (status(402, json={"detail": "No"}), "PROVIDER_INSUFFICIENT_BALANCE", FAIL_NOW),
        (status(403, json={"detail": "No"}), "PROVIDER_INSUFFICIENT_BALANCE", FAIL_NOW),
        (status(422, json={"detail": "Invalid"}), "PROVIDER_REJECTED", FAIL_NOW),
        (status(422, json={"detail": CONFLICT_DETAIL}), "PROVIDER_IDEMPOTENCY_CONFLICT", REVIEW),
        (status(423, json={"detail": "Blocked"}), None, RETRY),
        (status(429, headers={"Retry-After": "1"}), None, RETRY),
        (raising(httpx.ConnectError("refused")), None, RETRY),
        (status(500, json={"detail": "x"}), "PROVIDER_OUTCOME_UNKNOWN", RECONCILE),
        (status(502, json={"detail": "x"}), "PROVIDER_OUTCOME_UNKNOWN", RECONCILE),
        (status(503, json={"detail": "x"}), "PROVIDER_OUTCOME_UNKNOWN", RECONCILE),
        (raising(httpx.ReadTimeout("slow")), "PROVIDER_OUTCOME_UNKNOWN", RECONCILE),
    ],
    ids=[
        "400",
        "401",
        "402",
        "403",
        "422",
        "422-idempotency-conflict",
        "423",
        "429",
        "connect-error",
        "500",
        "502",
        "503",
        "read-timeout",
    ],
)
async def test_submit_outcome_by_status(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
    answer: Answer,
    code: str | None,
    outcome: tuple[ExecutionState, str, ReservationState, bool],
) -> None:
    fake = FakeHiggsfield()
    fake.submit_answers = [answer]

    await run(sessions, higgsfield(fake))

    job, reservation = await _load(sessions, world, job_id)
    event = await _event(connection)
    state, stage, credits, finished = outcome
    assert (job.state, job.stage, job.failure_code) == (state, stage, code)
    assert reservation is credits
    assert (event.processed_at is not None) is finished
    assert fake.submit_keys == [f"nrc-job-{job_id}"]
    if outcome is FAIL_NOW:
        assert f"provider refused job={job_id}" in caplog.text
        assert f"code={code}" in caplog.text


async def test_transient_submit_errors_retry_with_the_same_key(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    fake.submit_answers = [
        status(429, headers={"Retry-After": "1"}),
        status(503, json={"detail": "x"}),
        raising(httpx.ReadTimeout("slow")),
    ]
    provider = higgsfield(fake)

    for _ in range(4):
        await run(sessions, provider)
        await _requeue(connection, world)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.WAITING_PROVIDER
    assert job.provider_request_id == fake.accepted[f"nrc-job-{job_id}"]
    assert fake.submit_keys == [f"nrc-job-{job_id}"] * 4
    assert len(fake.accepted) == 1
    assert reservation is ReservationState.HELD


async def test_refusal_reason_is_logged_without_secrets(
    sessions: Sessions, job_id: UUID, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    fake = FakeHiggsfield()
    fake.submit_answers = [status(403, json={"detail": "ksecret-test balance"})]

    await run(sessions, higgsfield(fake))

    assert "code=PROVIDER_INSUFFICIENT_BALANCE" in caplog.text
    assert "HTTP 403" in caplog.text
    assert "ksecret-test" not in caplog.text


class CountingStorage(ObjectStorage):
    # Real SeaweedFS storage that also counts its puts.

    def __init__(self) -> None:
        super().__init__(get_settings())
        self.puts = 0

    async def save_generated_image(
        self, content: bytes, *, ext: str, content_type: str
    ) -> StoredObject:
        self.puts += 1
        return await super().save_generated_image(content, ext=ext, content_type=content_type)


def _png_1080() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1080, 1080), (12, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _output_server(png: bytes) -> SafeFetcher:
    # The provider's CDN, behind MockTransport, resolved to a public address.
    def cdn(request: httpx.Request) -> httpx.Response:
        assert request.url.host == OUTPUT_HOST
        assert "Authorization" not in request.headers
        return httpx.Response(200, content=png, headers={"content-type": "image/png"})

    async def public_resolver(host: str) -> list[str]:
        return ["93.184.216.34"]

    return SafeFetcher(
        httpx.AsyncClient(transport=httpx.MockTransport(cdn)),
        timeout_seconds=get_settings().provider_output_fetch_timeout_seconds,
        resolver=public_resolver,
    )


def _versions(key: str) -> int:
    settings = get_settings()
    s3 = boto3.Session().client(
        "s3", region_name=settings.aws_region, endpoint_url=settings.aws_endpoint_url_s3
    )
    listing = s3.list_object_versions(Bucket=str(settings.aws_s3_bucket), Prefix=key)
    return sum(1 for version in listing.get("Versions", []) if version.get("Key") == key)


@needs_seaweedfs
async def test_submit_5xx_after_acceptance_reconciles_to_one_charge_and_one_image(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    fake = FakeHiggsfield()
    fake.submit_answers = [accept_then(500)]
    expires = (datetime.now(UTC) + timedelta(days=7)).isoformat()
    fake.status_body = lambda request_id: {
        "status": "completed",
        "request_id": request_id,
        "images": [{"url": OUTPUT_URL}],
        "expires_at": expires,
    }
    provider = higgsfield(fake, _output_server(_png_1080()), frozenset({OUTPUT_HOST}))
    storage = CountingStorage()
    sink = StoredImageSink(storage, get_settings())
    reserved, committed = await _spend(connection, "higgsfield")

    await run(sessions, provider, sink)
    unknown, _ = await _load(sessions, world, job_id)
    await _requeue(connection, world)
    while await run(sessions, provider, sink):
        pass

    key = f"nrc-job-{job_id}"
    job, reservation = await _load(sessions, world, job_id)
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        assets = (await db.execute(select(Asset))).scalars().all()
    assert unknown.state is ExecutionState.UNKNOWN
    # The fake charged once: the resend got the original request back.
    assert fake.accepted == {key: job.provider_request_id}
    assert fake.submit_keys == [key, key]
    assert len(assets) == 1
    asset = assets[0]
    assert (asset.width_px, asset.height_px, asset.mime_type) == (1080, 1080, "image/png")
    assert asset.provider_request_id == job.provider_request_id
    assert storage.puts == 1
    assert _versions(asset.s3_key) == 1
    assert job.state is ExecutionState.SUCCEEDED
    assert job.result_asset_id == asset.id
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + 50_000)
    assert "SIGNED" not in caplog.text
    assert OUTPUT_HOST not in caplog.text


def test_http_client_request_urls_are_never_logged() -> None:
    # Output URLs are signed; httpx would log them at INFO.
    configure_logging("DEBUG")

    for name in ("httpx", "httpcore"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


def test_aws_sdk_responses_are_never_logged_at_debug() -> None:
    # botocore logs response bodies at DEBUG, including the Secrets Manager key.
    configure_logging("DEBUG")

    for name in ("botocore", "botocore.endpoint", "boto3", "urllib3"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
