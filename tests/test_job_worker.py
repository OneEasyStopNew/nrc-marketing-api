# Connector worker against the simulated provider: every image here is SIMULATED.

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import TenantSession, bind_tenant
from app.models import CreditReservation, Job
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState
from app.services.jobs import transition
from app.services.providers.base import FetchedImage
from app.services.providers.simulator import SimulatedImageProvider, SimulatorScenario
from app.workers.connector import WorkerConfigError, build_providers, run_once
from tests.conftest import AuthHeaders, World
from tests.test_jobs_api import _creative, _post_job

Sessions = async_sessionmaker[AsyncSession]


class FakeSink:
    # Stands in for generated-image storage (NRC-3).

    def __init__(self) -> None:
        self.stored: list[UUID] = []

    async def store(
        self, db: AsyncSession, job: Job, image: FetchedImage, provider_name: str, model: str
    ) -> UUID:
        assert image.content.startswith(b"\x89PNG")
        assert provider_name == "simulator"
        self.stored.append(uuid4())
        return self.stored[-1]


@pytest.fixture
def sessions(connection: AsyncConnection) -> Sessions:
    return async_sessionmaker(
        bind=connection,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
        sync_session_class=TenantSession,
    )


@pytest.fixture
async def job_id(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> UUID:
    creative_id = await _creative(client, auth, world.a)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    return UUID(response.json()["job_id"])


async def _load(sessions: Sessions, world: World, job_id: UUID) -> tuple[Job, ReservationState]:
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        job = await db.get(Job, job_id)
        reservation = await db.scalar(
            select(CreditReservation.state).where(CreditReservation.job_id == job_id)
        )
    assert job is not None
    assert reservation is not None
    return job, reservation


async def _requeue(connection: AsyncConnection, world: World) -> None:
    # Make every outbox event due now, as if its backoff had passed.
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text(
            "UPDATE outbox_events SET processed_at = NULL,"
            " available_at = clock_timestamp() - interval '1 second'"
        )
    )


async def _run(sessions: Sessions, provider: SimulatedImageProvider, sink: FakeSink) -> int:
    return await run_once(sessions, {"simulator": provider}, sink, get_settings())


async def test_simulated_success_stores_image_and_commits_credits(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    client: httpx.AsyncClient,
    auth: AuthHeaders,
) -> None:
    provider, sink = SimulatedImageProvider(), FakeSink()

    assert await _run(sessions, provider, sink) == 1
    assert await _run(sessions, provider, sink) == 0

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert job.result_asset_id == sink.stored[0]
    assert job.completed_at is not None
    assert reservation is ReservationState.COMMITTED
    polled = await client.get(
        f"/v1/marketing/jobs/{job_id}", headers=auth("creator-a", world.a.tenant_id)
    )
    assert polled.json()["state"] == "SUCCEEDED"
    assert polled.json()["result_asset_id"] == str(sink.stored[0])


async def test_simulated_timeout_after_accept_reconciles_with_same_request(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider, sink = SimulatedImageProvider(SimulatorScenario.TIMEOUT_AFTER_ACCEPT), FakeSink()

    await _run(sessions, provider, sink)
    unknown, held = await _load(sessions, world, job_id)
    key = str(unknown.provider_idempotency_key)
    assert unknown.state is ExecutionState.UNKNOWN
    assert held is ReservationState.HELD
    assert provider.submits == {key: 1}

    await _requeue(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert job.provider_request_id == provider._request_ids[key]
    assert provider.submits == {key: 2}
    assert reservation is ReservationState.COMMITTED


async def test_terminal_state_does_not_regress(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider, sink = SimulatedImageProvider(), FakeSink()
    await _run(sessions, provider, sink)

    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        moved = await transition(
            db, job_id, allowed_from=set(ExecutionState), to=ExecutionState.RUNNING
        )
        await db.commit()
    await _requeue(connection, world)
    await _run(sessions, provider, sink)

    job, _ = await _load(sessions, world, job_id)
    assert moved is False
    assert job.state is ExecutionState.SUCCEEDED
    assert len(sink.stored) == 1


async def test_expired_lease_is_processed_once(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider, sink = SimulatedImageProvider(), FakeSink()
    # A worker leases the event and dies; its zero-second lease has already expired.
    crashed = await connection.execute(
        text("SELECT * FROM lease_outbox_events('crashed-worker', 10, 0)")
    )
    assert len(crashed.all()) == 1

    assert await _run(sessions, provider, sink) == 1
    assert await _run(sessions, provider, sink) == 0

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert len(sink.stored) == 1


async def test_simulated_provider_failure_fails_the_job(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    await _run(sessions, SimulatedImageProvider(SimulatorScenario.PROVIDER_FAILED), FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert job.failure_code == "SIMULATED_FAILURE"
    # The provider accepted the request, so its billing applies.
    assert reservation is ReservationState.COMMITTED


async def test_simulated_rate_limit_retries_then_fails_and_releases_credits(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = SimulatedImageProvider(SimulatorScenario.RATE_LIMITED)

    for _ in range(get_settings().generation_max_attempts):
        await _run(sessions, provider, FakeSink())
        await connection.execute(
            text("UPDATE outbox_events SET available_at = clock_timestamp() - interval '1 second'")
        )

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert job.failure_code == "PROVIDER_RATE_LIMITED"
    assert sum(provider.submits.values()) == get_settings().generation_max_attempts
    assert reservation is ReservationState.RELEASED


async def test_job_past_deadline_fails_without_submitting(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    provider = SimulatedImageProvider()
    later = datetime.now(UTC) + timedelta(days=2)

    await run_once(sessions, {"simulator": provider}, FakeSink(), get_settings(), now=later)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert job.failure_code == "DEADLINE_EXCEEDED"
    assert not provider.submits
    assert reservation is ReservationState.RELEASED


async def test_simulated_cost_above_cap_fails_without_submitting(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    cap = get_settings().generation_max_cost_usd_per_image
    provider = SimulatedImageProvider(cost_usd=cap + Decimal("0.01"))

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert job.failure_code == "COST_ABOVE_LIMIT"
    assert job.provider_cost_estimate_usd_micros == int((cap + Decimal("0.01")) * 1_000_000)
    assert not provider.submits
    assert reservation is ReservationState.RELEASED


async def test_simulated_cost_estimate_is_stored_on_the_job(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    await _run(sessions, SimulatedImageProvider(cost_usd=Decimal("0.094")), FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert job.provider_cost_estimate_usd_micros == 94_000


async def test_missing_cost_estimate_never_submits(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    provider = SimulatedImageProvider(cost_usd=None)

    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert not provider.submits


def _live(**overrides: object) -> Settings:
    return get_settings().model_copy(update={"generation_mode": "live", **overrides})


def test_live_mode_without_keys_refuses_to_start() -> None:
    settings = _live(hf_key=None, higgsfield_model=None, higgsfield_output_hosts=())

    with pytest.raises(WorkerConfigError, match="HF_KEY") as refused:
        build_providers(settings)

    assert "HIGGSFIELD_MODEL, HIGGSFIELD_OUTPUT_HOSTS" in str(refused.value)
    assert "validation-PASS gate" in str(refused.value)


def test_live_mode_with_stub_validation_gate_refuses_to_start() -> None:
    settings = _live(
        hf_key=SecretStr("test-id:test-secret"),
        higgsfield_model="test/model",
        higgsfield_output_hosts=("images.example.test",),
    )

    with pytest.raises(WorkerConfigError, match=r"refused: the validation-PASS gate is still"):
        build_providers(settings)


@pytest.mark.parametrize("value", ["no-separator-value", ":only-secret", "only-id:", ""])
def test_malformed_hf_key_is_rejected_without_echoing_it(value: str) -> None:
    settings = _live(hf_key=SecretStr(value) if value else None)

    with pytest.raises(ValueError, match="HF_KEY") as rejected:
        settings.higgsfield_credentials()

    assert not value or value not in str(rejected.value)


def test_hf_key_splits_on_first_colon() -> None:
    key_id, key_secret = _live(hf_key=SecretStr("id-1:se:cret")).higgsfield_credentials()

    assert (key_id.get_secret_value(), key_secret.get_secret_value()) == ("id-1", "se:cret")
