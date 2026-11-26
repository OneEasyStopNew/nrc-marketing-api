# Provider limits in the connector worker (NRC-6): the concurrency cap across tenants and
# workers, and the per-provider circuit breaker. Every provider here is SIMULATED.

import asyncio
from collections.abc import AsyncIterator, Mapping
from decimal import Decimal
from uuid import UUID

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import Settings, get_settings
from app.db.session import bind_tenant, create_engine, create_sessionmaker
from app.main import create_app
from app.models.enums import ExecutionState, JobStage
from app.services.providers.base import (
    ImageProvider,
    ImageRequest,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderRejectedError,
    ProviderUnavailableError,
    StatusResult,
)
from app.services.providers.simulator import SimulatedImageProvider
from app.workers import connector
from app.workers.connector import run_once
from app.workers.limits import CircuitBreaker, GuardedProvider, guard_providers, submit_slot_free
from tests.conftest import (
    APP_ROLE,
    JOBS,
    AuthHeaders,
    FakeSink,
    IntegrationEnv,
    ScriptedProvider,
    SeededTenant,
    Sessions,
    StaticKeys,
    World,
    _creative,
    _load,
    _post_job,
    _requeue,
    _spend,
    as_role,
)
from tests.ledger import assert_ledger_consistent

REQUEST = ImageRequest(
    prompt="A synthetic test prompt.",
    width_px=1080,
    height_px=1080,
    aspect_ratio="1:1",
    idempotency_key="nrc-job-limits-test",
)


def _settings(**update: object) -> Settings:
    return get_settings().model_copy(update=update)


class FakeClock:
    # Monotonic seconds the test moves by hand.

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FailingProvider(SimulatedImageProvider):
    # SIMULATED; fails estimates and submits on cue and counts the calls that reached it.

    def __init__(self) -> None:
        super().__init__()
        self.error: Exception | None = None
        self.calls = 0

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return await super().estimate_cost_usd(request)


def _guarded(
    threshold: int = 2, open_seconds: float = 30
) -> tuple[GuardedProvider, FailingProvider, FakeClock]:
    clock, inner = FakeClock(), FailingProvider()
    settings = _settings(
        provider_circuit_failure_threshold=threshold, provider_circuit_open_seconds=open_seconds
    )
    return GuardedProvider(inner, CircuitBreaker(inner.name, settings, clock)), inner, clock


async def test_breaker_opens_after_consecutive_transient_failures_and_stops_calling() -> None:
    guarded, inner, _ = _guarded(threshold=2)
    inner.error = ProviderUnavailableError("Simulated outage.")
    for _ in range(2):
        with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
            await guarded.estimate_cost_usd(REQUEST)

    with pytest.raises(ProviderUnavailableError, match="paused briefly") as raised:
        await guarded.submit(REQUEST)

    assert inner.calls == 2
    assert raised.value.retry_after_seconds == pytest.approx(30)


async def test_breaker_lets_one_probe_through_after_the_open_period_and_closes_on_success() -> None:
    guarded, inner, clock = _guarded(threshold=1, open_seconds=30)
    inner.error = ProviderUnavailableError("Simulated outage.")
    with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
        await guarded.estimate_cost_usd(REQUEST)
    clock.now += 31
    inner.error = None

    assert await guarded.estimate_cost_usd(REQUEST) is not None
    assert await guarded.estimate_cost_usd(REQUEST) is not None
    assert inner.calls == 3


async def test_a_refusal_or_a_success_resets_the_failure_count() -> None:
    guarded, inner, _ = _guarded(threshold=2)
    inner.error = ProviderUnavailableError("Simulated outage.")
    with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
        await guarded.estimate_cost_usd(REQUEST)
    # A 4xx refusal means the provider is up.
    inner.error = ProviderRejectedError("Simulated refusal.")
    with pytest.raises(ProviderRejectedError, match="Simulated refusal"):
        await guarded.estimate_cost_usd(REQUEST)
    inner.error = ProviderUnavailableError("Simulated outage.")

    with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
        await guarded.estimate_cost_usd(REQUEST)

    # Still closed: two transient failures, but not in a row.
    assert inner.calls == 3


async def test_status_and_fetch_are_never_blocked_by_an_open_breaker() -> None:
    guarded, inner, _ = _guarded(threshold=1)
    submitted = await guarded.submit(REQUEST)
    inner.error = ProviderUnavailableError("Simulated outage.")
    with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
        await guarded.estimate_cost_usd(REQUEST)

    status = await guarded.get_status(submitted.provider_request_id)
    image = await guarded.fetch_output(status)

    assert status.status is ProviderJobStatus.COMPLETED
    assert image.content.startswith(b"\x89PNG")


def test_models_of_one_provider_share_one_breaker() -> None:
    first, second = SimulatedImageProvider(), SimulatedImageProvider()

    guarded = guard_providers({"simulator:a": first, "simulator:b": second}, _settings())

    breakers = {id(provider._breaker) for provider in guarded.values()}
    assert len(breakers) == 1
    assert {provider.inner for provider in guarded.values()} == {first, second}


async def test_in_flight_count_spans_tenants_and_returns_only_a_number(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
) -> None:
    provider, sink = ScriptedProvider(), FakeSink()
    provider.pending_polls = 1
    creative_id = await _creative(client, auth, world.a, sessions)
    assert (await _post_job(client, auth, world, creative_id)).status_code == 202
    await run_once(sessions, {"simulator": provider}, sink, get_settings())

    # Bound to tenant B, whose rows don't include tenant A's job: the count still sees it.
    async with sessions() as db:
        await bind_tenant(db, world.b.tenant_id)
        in_flight = await db.scalar(text("SELECT provider_in_flight('simulator')"))
        assert await submit_slot_free(
            db, "simulator", _settings(generation_max_in_flight_per_provider=2)
        )
        assert not await submit_slot_free(
            db, "simulator", _settings(generation_max_in_flight_per_provider=1)
        )

    assert in_flight == 1


async def test_only_the_worker_role_may_count_in_flight_jobs(connection: AsyncConnection) -> None:
    savepoint = await connection.begin_nested()
    await as_role(connection, APP_ROLE)
    with pytest.raises(DBAPIError, match="permission denied"):
        await connection.execute(text("SELECT provider_in_flight('simulator')"))
    await savepoint.rollback()


async def test_job_past_the_cap_waits_queued_without_a_provider_call_then_runs(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    settings = _settings(generation_max_in_flight_per_provider=1)
    provider, sink = ScriptedProvider(), FakeSink()
    provider.pending_polls = 1
    creative_id = await _creative(client, auth, world.a, sessions)
    first = UUID((await _post_job(client, auth, world, creative_id)).json()["job_id"])
    await run_once(sessions, {"simulator": provider}, sink, settings)
    second = UUID((await _post_job(client, auth, world, creative_id)).json()["job_id"])
    await _requeue(connection, world)
    provider.pending_polls = 1

    await run_once(sessions, {"simulator": provider}, sink, settings)

    waiting, _ = await _load(sessions, world, second)
    assert (waiting.state, waiting.stage) == (ExecutionState.QUEUED, None)
    # Priced once and kept, so the wait never costs a second estimate.
    assert waiting.provider_cost_estimate_usd_micros is not None
    assert sum(provider.submits.values()) == 1
    rows = (
        await connection.execute(
            text(
                "SELECT failed_attempts, processed_at FROM outbox_events"
                " WHERE job_id = :j AND event_type = 'job.run'"
            ),
            {"j": second},
        )
    ).one()
    assert (rows.failed_attempts, rows.processed_at) == (0, None)

    # The first job finishes, which frees its slot.
    await _requeue(connection, world)
    await run_once(sessions, {"simulator": provider}, sink, settings)
    await _requeue(connection, world)
    await run_once(sessions, {"simulator": provider}, sink, settings)

    done, _ = await _load(sessions, world, first)
    ran, _ = await _load(sessions, world, second)
    assert done.state is ExecutionState.SUCCEEDED
    assert ran.state is ExecutionState.SUCCEEDED
    assert provider.estimates == 2
    assert sum(provider.submits.values()) == 2


class StuckProvider(SimulatedImageProvider):
    # SIMULATED; both workers price their job before either moves on, and every accepted
    # generation stays in progress, so the first one keeps its slot.

    def __init__(self, workers: int) -> None:
        super().__init__()
        self._barrier = asyncio.Barrier(workers)

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        async with asyncio.timeout(10):
            await self._barrier.wait()
        return await super().estimate_cost_usd(request)

    async def get_status(self, provider_request_id: str) -> StatusResult:
        return StatusResult(provider_request_id, ProviderJobStatus.IN_PROGRESS)


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


async def test_two_workers_cannot_both_take_the_last_provider_slot(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    committed_worker_sessions: Sessions,
) -> None:
    auth = AuthHeaders(signing_key)
    creative_id = await _creative(http, auth, committed_tenant, committed_worker_sessions)
    for key in ("provider-slot-1", "provider-slot-2"):
        response = await http.post(
            JOBS,
            headers=auth("creator-a", committed_tenant.tenant_id, **{"Idempotency-Key": key}),
            json={"creative_id": creative_id, "creative_version": 1, "kind": "image_generation"},
        )
        assert response.status_code == 202
    settings = _settings(generation_max_in_flight_per_provider=1)
    provider, sink = StuckProvider(2), FakeSink()
    engines = [
        create_engine(settings, test_env.test_worker_database_url),
        create_engine(settings, test_env.test_worker_database_url),
    ]

    async def worker(sessions: async_sessionmaker[AsyncSession], owner: str) -> int:
        return await run_once(
            sessions, {"simulator": provider}, sink, settings, owner=owner, max_events=1
        )

    try:
        leased = await asyncio.gather(
            *(worker(create_sessionmaker(e), f"slot-worker-{n}") for n, e in enumerate(engines))
        )
    finally:
        for engine in engines:
            await engine.dispose()

    assert leased == [1, 1]
    assert sum(provider.submits.values()) == 1
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.connect() as conn:
        states = (
            await conn.execute(
                text(
                    "SELECT state, stage FROM jobs"
                    " WHERE tenant_id = :t AND kind = 'image_generation' ORDER BY state"
                ),
                {"t": committed_tenant.tenant_id},
            )
        ).all()
    await engine.dispose()
    assert [tuple(row) for row in states] == [
        (ExecutionState.QUEUED.value, None),
        (ExecutionState.WAITING_PROVIDER.value, JobStage.WAITING_PROVIDER.value),
    ]


async def test_run_forever_serves_guarded_providers_and_closes_the_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SimulatedImageProvider()
    served: list[Mapping[str, ImageProvider]] = []
    closed: list[Mapping[str, ImageProvider]] = []

    async def fake_serve(
        settings: Settings, providers: Mapping[str, ImageProvider], database_url: SecretStr
    ) -> None:
        served.append(providers)

    async def fake_close(providers: Mapping[str, ImageProvider]) -> None:
        closed.append(providers)

    monkeypatch.setattr(connector, "worker_database_url", lambda: SecretStr("unused"))
    monkeypatch.setattr(connector, "load_provider_secrets", lambda settings: settings)
    monkeypatch.setattr(connector, "build_providers", lambda settings: {"simulator": original})
    monkeypatch.setattr(connector, "_serve", fake_serve)
    monkeypatch.setattr(connector, "close_providers", fake_close)

    await connector.run_forever(get_settings())

    [providers] = served
    assert isinstance(providers["simulator"], GuardedProvider)
    assert providers["simulator"].inner is original
    assert closed == [{"simulator": original}]


async def test_a_failed_probe_after_the_open_period_reopens_at_once() -> None:
    guarded, inner, clock = _guarded(threshold=3, open_seconds=30)
    inner.error = ProviderUnavailableError("Simulated outage.")
    for _ in range(3):
        with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
            await guarded.estimate_cost_usd(REQUEST)
    clock.now += 31

    with pytest.raises(ProviderUnavailableError, match="Simulated outage"):
        await guarded.estimate_cost_usd(REQUEST)
    with pytest.raises(ProviderUnavailableError, match="paused briefly"):
        await guarded.estimate_cost_usd(REQUEST)

    # One probe, not another full run of failures.
    assert inner.calls == 4


async def _unknown_and_in_flight(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    provider: ScriptedProvider,
) -> tuple[UUID, UUID]:
    # One job whose submit outcome is unknown (UNKNOWN, a reconcile is due later), then one
    # accepted job still in progress at the provider.
    creative_id = await _creative(client, auth, world.a, sessions)
    unknown = UUID((await _post_job(client, auth, world, creative_id)).json()["job_id"])
    provider.submit_errors = [ProviderOutcomeUnknownError("Simulated lost response.")]
    await run_once(sessions, {"simulator": provider}, FakeSink(), get_settings())
    running = UUID((await _post_job(client, auth, world, creative_id)).json()["job_id"])
    provider.pending_polls = 1
    await run_once(sessions, {"simulator": provider}, FakeSink(), get_settings())
    assert (await _load(sessions, world, unknown))[0].state is ExecutionState.UNKNOWN
    assert (await _load(sessions, world, running))[0].state is ExecutionState.WAITING_PROVIDER
    return unknown, running


async def test_a_reconcile_resend_waits_at_the_provider_cap(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    # A same-key resend starts a new billed generation when the provider never got the original,
    # so it may not pass the cap that a first submit respects.
    spend_before = await _spend(connection)
    provider = ScriptedProvider()
    unknown, _ = await _unknown_and_in_flight(client, auth, world, sessions, provider)
    await _requeue(connection, world)
    provider.pending_polls = 2

    await run_once(
        sessions,
        {"simulator": provider},
        FakeSink(),
        _settings(generation_max_in_flight_per_provider=1),
    )

    job, _ = await _load(sessions, world, unknown)
    assert job.state is ExecutionState.UNKNOWN
    assert provider.submits[f"nrc-job-{unknown}"] == 1
    pending = (
        await connection.execute(
            text(
                "SELECT failed_attempts, processed_at FROM outbox_events"
                " WHERE job_id = :j AND event_type = 'job.run'"
            ),
            {"j": unknown},
        )
    ).one()
    assert (pending.failed_attempts, pending.processed_at) == (0, None)
    await assert_ledger_consistent(
        connection, world.a.tenant_id, unknown, spend_before=spend_before
    )


async def test_a_reconcile_resend_with_room_takes_a_slot_while_it_is_sent(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
) -> None:
    spend_before = await _spend(connection)
    provider = ScriptedProvider()
    unknown, running = await _unknown_and_in_flight(client, auth, world, sessions, provider)
    during_resend: list[tuple[str, str | None]] = []

    async def record_state() -> None:
        row = (
            await connection.execute(
                text("SELECT state, stage FROM jobs WHERE id = :j"), {"j": unknown}
            )
        ).one()
        during_resend.append((row.state, row.stage))

    provider.before_submit = record_state
    await _requeue(connection, world)

    await run_once(
        sessions,
        {"simulator": provider},
        FakeSink(),
        _settings(generation_max_in_flight_per_provider=2),
    )

    # Counted in flight (RUNNING/submitting) while the resend is out, like a first submit.
    assert during_resend == [("RUNNING", "submitting")]
    assert provider.submits[f"nrc-job-{unknown}"] == 2
    assert (await _load(sessions, world, unknown))[0].state is ExecutionState.SUCCEEDED
    for job_id in (unknown, running):
        await assert_ledger_consistent(
            connection, world.a.tenant_id, job_id, spend_before=spend_before
        )
