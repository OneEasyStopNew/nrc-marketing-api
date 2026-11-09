"""The connector worker with the real HiggsfieldProvider against a fake Higgsfield API.

The fake runs behind httpx.MockTransport: nothing leaves the process.
"""

import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.session import TenantSession
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState
from app.services.providers.base import FetchedImage, OutputFetcher, ProviderOutputError
from app.services.providers.higgsfield import HiggsfieldProvider
from app.workers.connector import GeneratedImageSink, run_once
from tests.conftest import AuthHeaders, World
from tests.test_job_worker import _event, _load, _requeue
from tests.test_jobs_api import _creative, _post_job

Sessions = async_sessionmaker[AsyncSession]
MODEL = "higgsfield-ai/soul/standard"
ESTIMATE_USD = "0.05"
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
def sessions(connection: AsyncConnection) -> Sessions:
    return async_sessionmaker(
        bind=connection,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=TenantSession,
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
