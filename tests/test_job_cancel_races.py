# A cancel racing a worker mid-step, on committed data with separate connections (NRC-6).
#
# The cancel goes through the real POST /jobs/{id}/cancel while the worker is inside a provider
# call holding its event lease, so the row-lock order between the API and the worker really
# applies (on one shared test connection, locks never conflict). Every provider is SIMULATED.

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import get_settings
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.services.providers.base import FetchedImage, ImageRequest, StatusResult, SubmitResult
from app.services.providers.simulator import SimulatedImageProvider
from app.workers import connector
from app.workers.connector import run_once
from tests.conftest import (
    JOBS,
    AuthHeaders,
    FakeSink,
    IntegrationEnv,
    SeededTenant,
    StaticKeys,
    _creative,
)

# Far longer than a cancel needs; only a cancel stuck behind the worker's lock reaches it.
CANCEL_WAIT_SECONDS = 15
# Time a cancel gets to reach its update of the job row before the worker commits (well under
# the 2 s default JOB_CANCEL_LOCK_TIMEOUT_SECONDS).
CANCEL_REACH_SECONDS = 0.5

Hook = Callable[[], Awaitable[None]]


class HookProvider(SimulatedImageProvider):
    # SIMULATED; runs a hook once inside the submit or the output fetch, as a slow provider call.

    def __init__(self) -> None:
        super().__init__()
        self.before_submit: Hook | None = None
        self.before_fetch: Hook | None = None

    async def submit(self, request: ImageRequest) -> SubmitResult:
        if (hook := self.before_submit) is not None:
            self.before_submit = None
            await hook()
        return await super().submit(request)

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        if (hook := self.before_fetch) is not None:
            self.before_fetch = None
            await hook()
        return await super().fetch_output(result)


@pytest.fixture
async def http(
    committed_tenant: SeededTenant, signing_key: RSAPrivateKey
) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


async def _owner(test_env: IntegrationEnv, sql: str, tenant: SeededTenant) -> list[Any]:
    # As the owner on its own connection, committed at once.
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.begin() as conn:
        result = await conn.execute(text(sql), {"t": tenant.tenant_id})
        rows = list(result.all()) if result.returns_rows else []
    await engine.dispose()
    return rows


async def _queued_job(
    http: httpx.AsyncClient, auth: AuthHeaders, tenant: SeededTenant, key: str
) -> str:
    creative_id = await _creative(http, auth, tenant)
    response = await http.post(
        JOBS,
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": key}),
        json={"creative_id": creative_id, "creative_version": 1, "kind": "image_generation"},
    )
    assert response.status_code == 202
    job_id: str = response.json()["job_id"]
    return job_id


def _cancel_from_inside(
    http: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    job_id: str,
    responses: list[httpx.Response],
) -> Hook:
    async def cancel() -> None:
        async with asyncio.timeout(CANCEL_WAIT_SECONDS):
            responses.append(
                await http.post(
                    f"/v1/marketing/jobs/{job_id}/cancel",
                    headers=auth("creator-a", tenant.tenant_id),
                )
            )

    return cancel


async def _run_worker(
    test_env: IntegrationEnv, provider: HookProvider, sink: FakeSink | None, owner: str
) -> None:
    settings = get_settings()
    engine = create_engine(settings, test_env.test_worker_database_url)
    try:
        await run_once(
            create_sessionmaker(engine),
            {"simulator": provider},
            sink,
            settings,
            owner=owner,
            max_events=1,
        )
    finally:
        await engine.dispose()


async def _finish_cancel(
    test_env: IntegrationEnv, provider: HookProvider, tenant: SeededTenant
) -> list[Any]:
    # The worker kept the event for the cancel; make it due and let the worker finish it.
    await _owner(
        test_env,
        "UPDATE outbox_events SET available_at = clock_timestamp() WHERE tenant_id = :t"
        " AND processed_at IS NULL",
        tenant,
    )
    await _run_worker(test_env, provider, FakeSink(), "cancel-race-finish")
    return await _owner(
        test_env,
        "SELECT j.state, j.failure_code, r.state AS credits,"
        " (SELECT count(*) FROM assets a WHERE a.job_id = j.id) AS assets"
        " FROM jobs j JOIN credit_reservations r ON r.job_id = j.id WHERE j.tenant_id = :t",
        tenant,
    )


async def test_cancel_during_the_provider_submit_answers_at_once_and_keeps_the_request(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
) -> None:
    auth = AuthHeaders(signing_key)
    job_id = await _queued_job(http, auth, committed_tenant, "cancel-race-submit")
    provider, responses = HookProvider(), list[httpx.Response]()
    provider.before_submit = _cancel_from_inside(http, auth, committed_tenant, job_id, responses)

    await _run_worker(test_env, provider, FakeSink(), "cancel-race-submit")

    [response] = responses
    assert response.status_code == 202
    assert response.json()["state"] == "CANCEL_REQUESTED"
    [mid] = await _owner(
        test_env,
        "SELECT j.state, j.provider_request_id,"
        " (SELECT count(*) FROM outbox_events o WHERE o.job_id = j.id"
        "  AND o.processed_at IS NULL) AS pending"
        " FROM jobs j WHERE j.tenant_id = :t",
        committed_tenant,
    )
    # The worker didn't lose the provider's answer to the cancel, and kept the event.
    assert (mid.state, mid.pending) == ("CANCEL_REQUESTED", 1)
    assert mid.provider_request_id is not None

    [done] = await _finish_cancel(test_env, provider, committed_tenant)

    assert tuple(done) == ("CANCELLED", "CANCELLED_AFTER_COMPLETION", "COMMITTED", 0)
    assert sum(provider.submits.values()) == 1


async def test_cancel_during_the_output_fetch_answers_at_once_and_keeps_no_asset(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
) -> None:
    auth = AuthHeaders(signing_key)
    job_id = await _queued_job(http, auth, committed_tenant, "cancel-race-fetch")
    provider, responses = HookProvider(), list[httpx.Response]()
    provider.before_fetch = _cancel_from_inside(http, auth, committed_tenant, job_id, responses)

    await _run_worker(test_env, provider, FakeSink(), "cancel-race-fetch")

    [response] = responses
    assert response.status_code == 202
    assert response.json()["state"] == "CANCEL_REQUESTED"

    [done] = await _finish_cancel(test_env, provider, committed_tenant)

    # Billed (the provider completed it), but the image is not kept as an asset.
    assert tuple(done) == ("CANCELLED", "CANCELLED_AFTER_COMPLETION", "COMMITTED", 0)
    assert sum(provider.submits.values()) == 1


async def test_cancel_while_the_worker_closes_its_event_still_leaves_one_to_finish_it(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker's step has nothing more to do (no image storage, so the job parks at
    # output_ready) and it closes its event; the cancel lands between the worker's last job
    # update and its commit. Either the worker keeps its event or the cancel adds one, never
    # neither (or the job sits in CANCEL_REQUESTED with nothing to finish it).
    auth = AuthHeaders(signing_key)
    job_id = await _queued_job(http, auth, committed_tenant, "cancel-race-close")
    real_finish = connector._finish_event
    cancels: list[asyncio.Task[httpx.Response]] = []

    async def finish_while_a_cancel_lands(
        db: Any, run: Any, event_id: Any, retry_after: float | None, **kwargs: Any
    ) -> None:
        if not cancels:
            cancels.append(
                asyncio.create_task(
                    http.post(
                        f"/v1/marketing/jobs/{job_id}/cancel",
                        headers=auth("creator-a", committed_tenant.tenant_id),
                    )
                )
            )
            await asyncio.sleep(CANCEL_REACH_SECONDS)
        await real_finish(db, run, event_id, retry_after, **kwargs)

    monkeypatch.setattr(connector, "_finish_event", finish_while_a_cancel_lands)

    await _run_worker(test_env, HookProvider(), None, "cancel-race-close")
    async with asyncio.timeout(CANCEL_WAIT_SECONDS):
        response = await cancels[0]

    assert response.status_code == 202
    [row] = await _owner(
        test_env,
        "SELECT j.state, j.stage, (SELECT count(*) FROM outbox_events o WHERE o.job_id = j.id"
        "  AND o.event_type = 'job.run' AND o.processed_at IS NULL) AS pending"
        " FROM jobs j WHERE j.tenant_id = :t",
        committed_tenant,
    )
    assert (row.state, row.stage, row.pending) == ("CANCEL_REQUESTED", "output_ready", 1)


@pytest.fixture
def short_cancel_lock_timeout() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("JOB_CANCEL_LOCK_TIMEOUT_SECONDS", "0.5")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("short_cancel_lock_timeout")
async def test_cancel_of_a_job_row_held_by_another_transaction_is_a_retryable_503(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
) -> None:
    auth = AuthHeaders(signing_key)
    job_id = await _queued_job(http, auth, committed_tenant, "cancel-race-locked")
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with engine.connect() as holder:
            await holder.execute(
                text("SELECT id FROM jobs WHERE id = :j FOR UPDATE"), {"j": job_id}
            )
            async with asyncio.timeout(CANCEL_WAIT_SECONDS):
                response = await http.post(
                    f"/v1/marketing/jobs/{job_id}/cancel",
                    headers=auth("creator-a", committed_tenant.tenant_id),
                )
            await holder.rollback()
    finally:
        await engine.dispose()

    assert response.status_code == 503
    body = response.json()
    assert (body["code"], body["retryable"]) == ("DEPENDENCY_UNAVAILABLE", True)
    [state] = await _owner(
        test_env, "SELECT state FROM jobs WHERE tenant_id = :t", committed_tenant
    )
    assert state.state == "QUEUED"
