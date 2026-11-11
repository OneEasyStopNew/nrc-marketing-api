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
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import bind_tenant
from app.models import Asset
from app.models.enums import ExecutionState, JobStage
from app.models.jobs import ReservationState, SpendState
from app.services.assets import StoredImageSink
from app.services.fetch import SafeFetcher
from app.services.providers.base import (
    FetchedImage,
    OutputFetcher,
    ProviderOutputError,
    ProviderUnavailableError,
)
from app.services.providers.higgsfield import HiggsfieldProvider
from app.services.storage import ObjectStorage, StoredObject
from app.workers.connector import GeneratedImageSink, run_once
from tests.conftest import (
    AuthHeaders,
    Sessions,
    World,
    _creative,
    _event,
    _load,
    _post_job,
    _requeue,
    _spend,
    needs_seaweedfs,
)

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


FAIL_NOW = (ExecutionState.FAILED, JobStage.SUBMITTING, ReservationState.RELEASED, True)
RETRY = (ExecutionState.RUNNING, JobStage.RETRY_WAIT, ReservationState.RESERVED, False)
RECONCILE = (ExecutionState.UNKNOWN, JobStage.RECONCILE, ReservationState.HELD, False)
REVIEW = (ExecutionState.UNKNOWN, JobStage.RECONCILE, ReservationState.HELD, True)


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
    outcome: tuple[ExecutionState, JobStage, ReservationState, bool],
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


def _png_1536() -> bytes:
    # What the model returns for 1:1 at 1080p (live test, 10 Nov 2026).
    buffer = io.BytesIO()
    Image.new("RGB", (1536, 1536), (12, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


CdnAnswer = Callable[[httpx.Request], httpx.Response]


def _output_server(png: bytes, outages: list[CdnAnswer] | None = None) -> SafeFetcher:
    # The provider's CDN, behind MockTransport, resolved to a public address. Each outage
    # answers one download before the CDN serves the image.
    def cdn(request: httpx.Request) -> httpx.Response:
        assert request.url.host == OUTPUT_HOST
        assert "Authorization" not in request.headers
        if outages:
            return outages.pop(0)(request)
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
    provider = higgsfield(fake, _output_server(_png_1536()), frozenset({OUTPUT_HOST}))
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
    assert "provider_size=1536x1536 stored_size=1080x1080" in caplog.text
    assert "SIGNED" not in caplog.text
    assert OUTPUT_HOST not in caplog.text


ESTIMATE_MICROS = 50_000


def _completed(expires_at: datetime) -> Callable[[str], dict[str, Any]]:
    return lambda request_id: {
        "status": "completed",
        "request_id": request_id,
        "images": [{"url": OUTPUT_URL}],
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
    }


def _past_deadline_within_grace() -> datetime:
    # About one deadline past the deadline: well inside the default grace window.
    return datetime.now(UTC) + timedelta(seconds=get_settings().generation_deadline_seconds * 2)


def _past_grace() -> datetime:
    settings = get_settings()
    window = settings.generation_deadline_seconds + settings.generation_accepted_grace_seconds
    return datetime.now(UTC) + timedelta(seconds=window + 60)


async def _accepted_and_in_progress(
    sessions: Sessions, fake: FakeHiggsfield, connection: AsyncConnection, world: World
) -> None:
    # Higgsfield accepts the job and is still working on it when the deadline passes.
    await run(sessions, higgsfield(fake))
    await _requeue(connection, world)


async def run_at(
    sessions: Sessions,
    provider: HiggsfieldProvider,
    now: datetime,
    sink: GeneratedImageSink | None = None,
) -> int:
    return await run_once(sessions, {"higgsfield": provider}, sink, get_settings(), now=now)


def _sink() -> StoredImageSink:
    return StoredImageSink(ObjectStorage(get_settings()), get_settings())


def _in_progress(polls: list[str]) -> Callable[[str], dict[str, Any]]:
    def answer(request_id: str) -> dict[str, Any]:
        polls.append(request_id)
        return {"status": "in_progress", "request_id": request_id}

    return answer


@needs_seaweedfs
async def test_accepted_job_past_grace_still_stores_the_completed_image(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    # Once Higgsfield reports completed the grace window no longer applies: it billed.
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    later = _past_grace()
    reserved, committed = await _spend(connection, "higgsfield")
    fake.status_body = _completed(later + timedelta(days=7))
    provider = higgsfield(fake, _output_server(_png_1536()), frozenset({OUTPUT_HOST}))

    await run_at(sessions, provider, later, _sink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.stage) == (ExecutionState.SUCCEEDED, JobStage.STORED)
    assert job.result_asset_id is not None
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    spent = await _spend(connection, "higgsfield")
    assert spent == (reserved - ESTIMATE_MICROS, committed + ESTIMATE_MICROS)
    assert len(fake.submit_keys) == 1


async def test_accepted_job_within_grace_keeps_polling_while_in_progress(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    later = _past_deadline_within_grace()
    spent = await _spend(connection, "higgsfield")
    polls: list[str] = []
    fake.status_body = _in_progress(polls)
    provider = higgsfield(fake)

    for _ in range(2):
        await run_at(sessions, provider, later)
        job, reservation = await _load(sessions, world, job_id)
        assert (job.state, job.stage) == (
            ExecutionState.WAITING_PROVIDER,
            JobStage.WAITING_PROVIDER,
        )
        assert reservation is ReservationState.RESERVED
        assert job.provider_spend_state is SpendState.RESERVED
        assert (await _event(connection)).processed_at is None
        await _requeue(connection, world)

    assert polls == [job.provider_request_id] * 2
    assert await _spend(connection, "higgsfield") == spent
    assert len(fake.submit_keys) == 1


async def test_accepted_job_still_in_progress_past_grace_is_parked_with_spend_reserved(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    spent = await _spend(connection, "higgsfield")
    polls: list[str] = []
    fake.status_body = _in_progress(polls)

    with caplog.at_level(logging.ERROR, logger="app.workers.connector"):
        await run_at(sessions, higgsfield(fake), _past_grace())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.stage, job.failure_code) == (
        ExecutionState.UNKNOWN,
        JobStage.RECONCILE,
        "PROVIDER_STILL_RUNNING",
    )
    # Higgsfield may still complete and bill it: credits held, spend still reserved.
    assert reservation is ReservationState.HELD
    assert job.provider_spend_state is SpendState.RESERVED
    assert await _spend(connection, "higgsfield") == spent
    assert polls == [job.provider_request_id]
    assert (await _event(connection)).processed_at is not None
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(job.id) in errors[0]
    assert str(job.provider_request_id) in errors[0]


async def test_accepted_job_past_deadline_fails_when_higgsfield_fails(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    reserved, committed = await _spend(connection, "higgsfield")
    fake.status_body = lambda request_id: {"status": "failed", "request_id": request_id}

    await run_at(sessions, higgsfield(fake), _past_grace())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "provider_failed")
    assert reservation is ReservationState.RELEASED
    assert job.provider_spend_state is SpendState.RELEASED
    assert await _spend(connection, "higgsfield") == (reserved - ESTIMATE_MICROS, committed)


async def test_expired_output_fails_with_credits_and_spend_committed(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    later = _past_deadline_within_grace()
    reserved, committed = await _spend(connection, "higgsfield")
    # Higgsfield's expires_at: after it the files are deleted and the output URLs stop working.
    fake.status_body = _completed(later - timedelta(hours=1))
    fetched: list[str] = []

    async def fetcher(url: str, *, allowed_hosts: frozenset[str], max_bytes: int) -> FetchedImage:
        fetched.append(url)
        raise ProviderOutputError("an expired output is never fetched")

    await run_at(sessions, higgsfield(fake, fetcher, frozenset({OUTPUT_HOST})), later)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_EXPIRED")
    # Higgsfield billed the completed generation: credits and spend are committed.
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    spent = await _spend(connection, "higgsfield")
    assert spent == (reserved - ESTIMATE_MICROS, committed + ESTIMATE_MICROS)
    assert not fetched
    assert (await _event(connection)).processed_at is not None


async def test_output_expiring_at_fetch_time_ends_the_same_way(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    # The worker's clock still sees the output as live; the adapter's own check at fetch time
    # finds it expired. One rule: FAILED OUTPUT_EXPIRED, charged.
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    reserved, committed = await _spend(connection, "higgsfield")
    fake.status_body = _completed(datetime.now(UTC) - timedelta(seconds=1))
    fetched: list[str] = []

    async def fetcher(url: str, *, allowed_hosts: frozenset[str], max_bytes: int) -> FetchedImage:
        fetched.append(url)
        raise ProviderOutputError("an expired output is never fetched")

    provider = higgsfield(fake, fetcher, frozenset({OUTPUT_HOST}))
    await run_at(sessions, provider, datetime.now(UTC) - timedelta(minutes=5), _sink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_EXPIRED")
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    spent = await _spend(connection, "higgsfield")
    assert spent == (reserved - ESTIMATE_MICROS, committed + ESTIMATE_MICROS)
    assert not fetched


async def test_accepted_job_past_deadline_is_parked_once_attempts_run_out(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    await _accepted_and_in_progress(sessions, fake, connection, world)
    spent = await _spend(connection, "higgsfield")
    # As if the event had already failed GENERATION_MAX_ATTEMPTS times (crashes, lost leases).
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text("UPDATE outbox_events SET failed_attempts = :n"),
        {"n": get_settings().generation_max_attempts},
    )

    await run_at(sessions, higgsfield(fake), _past_deadline_within_grace())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.UNKNOWN, "ATTEMPTS_EXHAUSTED")
    assert reservation is ReservationState.HELD
    assert job.provider_spend_state is SpendState.RESERVED
    assert await _spend(connection, "higgsfield") == spent


@needs_seaweedfs
async def test_job_fetching_output_past_grace_keeps_fetching(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    fake.status_body = _completed(datetime.now(UTC) + timedelta(days=7))

    async def cdn_down(url: str, *, allowed_hosts: frozenset[str], max_bytes: int) -> FetchedImage:
        raise ProviderUnavailableError("CDN unavailable")

    await run(sessions, higgsfield(fake, cdn_down, frozenset({OUTPUT_HOST})), _sink())
    fetching, _ = await _load(sessions, world, job_id)
    await _requeue(connection, world)
    provider = higgsfield(fake, _output_server(_png_1536()), frozenset({OUTPUT_HOST}))

    await run_at(sessions, provider, _past_grace(), _sink())

    job, reservation = await _load(sessions, world, job_id)
    assert (fetching.state, fetching.stage) == (
        ExecutionState.WAITING_PROVIDER,
        JobStage.FETCHING_OUTPUT,
    )
    assert (job.state, job.stage) == (ExecutionState.SUCCEEDED, JobStage.STORED)
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED


UNLISTED_ONLY = frozenset({"other-cdn.example"})


@needs_seaweedfs
async def test_unlisted_output_host_keeps_the_job_waiting_and_logs_only_the_host(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeHiggsfield()
    fake.status_body = _completed(datetime.now(UTC) + timedelta(days=7))
    provider = higgsfield(fake, _output_server(_png_1536()), UNLISTED_ONLY)
    reserved, committed = await _spend(connection, "higgsfield")

    # More rounds than GENERATION_MAX_ATTEMPTS: a refused host never counts as a failure.
    with caplog.at_level(logging.ERROR, logger="app.workers.connector"):
        for _ in range(get_settings().generation_max_attempts + 1):
            await run(sessions, provider, _sink())
            await _requeue(connection, world)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.stage) == (ExecutionState.WAITING_PROVIDER, JobStage.FETCHING_OUTPUT)
    assert job.result_asset_id is None
    assert reservation is ReservationState.RESERVED
    assert job.provider_spend_state is SpendState.RESERVED
    assert await _spend(connection, "higgsfield") == (reserved + ESTIMATE_MICROS, committed)
    assert (await _event(connection)).failed_attempts == 0
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors
    for message in errors:
        assert f"host={OUTPUT_HOST}" in message
        assert f"job={job.id}" in message
        assert f"tenant={job.tenant_id}" in message
        # Never the path or the query: the signed URL is a credential.
        assert "/outputs" not in message
        assert "result.png" not in message
        assert "signature" not in message.lower()


@needs_seaweedfs
async def test_output_is_stored_once_the_host_is_added_to_the_allowlist(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    fake.status_body = _completed(datetime.now(UTC) + timedelta(days=7))
    reserved, committed = await _spend(connection, "higgsfield")
    await run(sessions, higgsfield(fake, _output_server(_png_1536()), UNLISTED_ONLY), _sink())
    waiting, _ = await _load(sessions, world, job_id)
    await _requeue(connection, world)

    # Ops add the host to HIGGSFIELD_OUTPUT_HOSTS and restart the worker.
    fixed = higgsfield(fake, _output_server(_png_1536()), UNLISTED_ONLY | {OUTPUT_HOST})
    await run(sessions, fixed, _sink())

    job, reservation = await _load(sessions, world, job_id)
    assert waiting.state is ExecutionState.WAITING_PROVIDER
    assert (job.state, job.stage) == (ExecutionState.SUCCEEDED, JobStage.STORED)
    assert job.result_asset_id is not None
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + ESTIMATE_MICROS)
    assert len(fake.submit_keys) == 1


@needs_seaweedfs
async def test_unlisted_output_host_ends_output_expired_once_the_output_expires(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    expires = datetime.now(UTC) + timedelta(hours=1)
    fake.status_body = _completed(expires)
    provider = higgsfield(fake, _output_server(_png_1536()), UNLISTED_ONLY)
    reserved, committed = await _spend(connection, "higgsfield")
    await run(sessions, provider, _sink())
    await _requeue(connection, world)

    await run_at(sessions, provider, expires + timedelta(minutes=1), _sink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_EXPIRED")
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + ESTIMATE_MICROS)
    assert (await _event(connection)).processed_at is not None


def _cdn_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _cdn_unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503)


@needs_seaweedfs
async def test_output_fetch_timeout_retries_and_stores_the_image_next_round(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    # One network blip must not lose an image Higgsfield already billed.
    fake = FakeHiggsfield()
    fake.status_body = _completed(datetime.now(UTC) + timedelta(days=7))
    reserved, committed = await _spend(connection, "higgsfield")
    server = _output_server(_png_1080(), [_cdn_timeout])
    provider = higgsfield(fake, server, frozenset({OUTPUT_HOST}))

    await run(sessions, provider, _sink())
    waiting, waiting_reservation = await _load(sessions, world, job_id)
    failed_attempts = (await _event(connection)).failed_attempts
    await _requeue(connection, world)
    await run(sessions, provider, _sink())

    assert (waiting.state, waiting.stage) == (
        ExecutionState.WAITING_PROVIDER,
        JobStage.FETCHING_OUTPUT,
    )
    assert waiting_reservation is ReservationState.RESERVED
    assert failed_attempts == 0
    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.stage) == (ExecutionState.SUCCEEDED, JobStage.STORED)
    assert job.result_asset_id is not None
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + ESTIMATE_MICROS)
    assert len(fake.submit_keys) == 1


@needs_seaweedfs
async def test_output_fetch_failing_until_expiry_ends_output_expired_and_billed(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    expires = datetime.now(UTC) + timedelta(hours=1)
    fake.status_body = _completed(expires)
    reserved, committed = await _spend(connection, "higgsfield")
    rounds = get_settings().generation_max_attempts + 1
    outages: list[CdnAnswer] = [_cdn_unavailable, _cdn_timeout] * rounds
    provider = higgsfield(fake, _output_server(_png_1080(), outages), frozenset({OUTPUT_HOST}))

    # More rounds than GENERATION_MAX_ATTEMPTS: transient fetch failures never count.
    for _ in range(rounds):
        await run(sessions, provider, _sink())
        await _requeue(connection, world)
    waiting, _ = await _load(sessions, world, job_id)
    failed_attempts = (await _event(connection)).failed_attempts
    await run_at(sessions, provider, expires + timedelta(minutes=1), _sink())

    assert (waiting.state, waiting.stage) == (
        ExecutionState.WAITING_PROVIDER,
        JobStage.FETCHING_OUTPUT,
    )
    assert failed_attempts == 0
    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_EXPIRED")
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + ESTIMATE_MICROS)
    assert (await _event(connection)).processed_at is not None


FALLBACK_SECONDS = 3600


def _completed_without_expiry(request_id: str) -> dict[str, Any]:
    return {"status": "completed", "request_id": request_id, "images": [{"url": OUTPUT_URL}]}


async def run_with_fallback(
    sessions: Sessions, provider: HiggsfieldProvider, now: datetime | None = None
) -> int:
    settings = get_settings().model_copy(
        update={"provider_output_retention_fallback_seconds": FALLBACK_SECONDS}
    )
    return await run_once(sessions, {"higgsfield": provider}, _sink(), settings, now=now)


@needs_seaweedfs
async def test_unlisted_host_without_expiry_ends_output_expired_after_the_retention_fallback(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeHiggsfield()
    fake.status_body = _completed_without_expiry
    reserved, committed = await _spend(connection, "higgsfield")
    seen_completed = datetime.now(UTC)

    with caplog.at_level(logging.WARNING, logger="app.workers.connector"):
        # Each round is a fresh provider, as after a worker restart.
        await run_with_fallback(sessions, higgsfield(fake, _output_server(_png_1536())))
        await _requeue(connection, world)
        within = seen_completed + timedelta(seconds=FALLBACK_SECONDS / 2)
        await run_with_fallback(sessions, higgsfield(fake, _output_server(_png_1536())), within)
        waiting, _ = await _load(sessions, world, job_id)
        await _requeue(connection, world)
        past = seen_completed + timedelta(seconds=FALLBACK_SECONDS + 60)
        await run_with_fallback(sessions, higgsfield(fake, _output_server(_png_1536())), past)

    job, reservation = await _load(sessions, world, job_id)
    assert (waiting.state, waiting.stage) == (
        ExecutionState.WAITING_PROVIDER,
        JobStage.FETCHING_OUTPUT,
    )
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_EXPIRED")
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + ESTIMATE_MICROS)
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "expires_at" in r.getMessage()
    ]
    assert warnings
    assert all(f"job={job.id}" in message for message in warnings)
    assert all(OUTPUT_HOST not in message for message in warnings)


@needs_seaweedfs
async def test_missing_expiry_with_an_allowed_host_still_stores_the_image(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    fake = FakeHiggsfield()
    fake.status_body = _completed_without_expiry
    reserved, committed = await _spend(connection, "higgsfield")
    provider = higgsfield(fake, _output_server(_png_1536()), frozenset({OUTPUT_HOST}))

    await run_with_fallback(sessions, provider)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.stage) == (ExecutionState.SUCCEEDED, JobStage.STORED)
    assert reservation is ReservationState.COMMITTED
    assert await _spend(connection, "higgsfield") == (reserved, committed + ESTIMATE_MICROS)


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
