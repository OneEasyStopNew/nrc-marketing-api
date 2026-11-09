# Connector worker against the simulated provider: every image here is SIMULATED.

import dataclasses
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import Row, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import bind_tenant
from app.models import Asset, CreditReservation, Job
from app.models.enums import ExecutionState
from app.models.jobs import ReservationState, SpendState
from app.services.jobs import jobs_awaiting_review, transition
from app.services.providers.base import (
    FetchedImage,
    ImageRequest,
    ProviderConfigError,
    ProviderError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
    ProviderOutputError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
    StatusResult,
    SubmitResult,
)
from app.services.providers.simulator import SimulatedImageProvider, SimulatorScenario
from app.workers.connector import (
    EVENT_HANDLERS,
    JobRunHandler,
    LeasedEvent,
    Step,
    WorkerConfigError,
    build_providers,
    run_once,
)
from tests.conftest import WORKER_ROLE, AuthHeaders, World, app_session, as_role
from tests.test_jobs_api import _creative, _post_job

Sessions = async_sessionmaker[AsyncSession]


class FakeSink:
    # Stands in for generated-image storage: a metadata row without any object.

    def __init__(self) -> None:
        self.stored: list[UUID] = []

    async def store(
        self, db: AsyncSession, job: Job, image: FetchedImage, provider_name: str, model: str
    ) -> UUID:
        assert image.content.startswith(b"\x89PNG")
        assert provider_name == "simulator"
        # A real row, because jobs.result_asset_id is a foreign key to assets.
        asset = Asset(
            tenant_id=job.tenant_id,
            creative_id=job.creative_id,
            creative_version=job.creative_version,
            job_id=job.id,
            origin="generated",
            bucket="fake",
            s3_key=f"fake/{uuid4()}.png",
            mime_type="image/png",
            width_px=1,
            height_px=1,
            size_bytes=len(image.content),
            checksum_sha256="0" * 64,
            created_by=job.created_by,
        )
        db.add(asset)
        await db.flush()
        self.stored.append(asset.id)
        return asset.id


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


async def _expire_leases(connection: AsyncConnection, world: World) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text("UPDATE outbox_events SET leased_until = clock_timestamp() - interval '1 second'")
    )


async def _event(connection: AsyncConnection) -> Row[Any]:
    # The single outbox event of the test (RLS: tenant A is bound by now).
    return (
        await connection.execute(
            text(
                "SELECT processed_at, lease_owner, last_error_code, attempts, failed_attempts"
                " FROM outbox_events WHERE event_type = 'job.run'"
            )
        )
    ).one()


class ScriptedProvider(SimulatedImageProvider):
    # SIMULATED provider whose next calls can fail on cue.

    def __init__(self) -> None:
        super().__init__()
        # Raised by the next submits, after the request was recorded (accepted, response lost).
        self.submit_errors: list[Exception] = []
        self.status_error: ProviderError | None = None
        self.fetch_error: ProviderError | None = None
        # Reported in the COMPLETED status, as a provider stating its actual cost would.
        self.actual_cost_usd: Decimal | None = None
        self.estimates = 0
        self.before_estimate: Callable[[], Awaitable[None]] | None = None
        self.before_submit: Callable[[], Awaitable[None]] | None = None
        self.before_fetch: Callable[[], Awaitable[None]] | None = None
        # Status checks that still report the generation as in progress.
        self.pending_polls = 0

    async def estimate_cost_usd(self, request: ImageRequest) -> Decimal | None:
        self.estimates += 1
        if self.before_estimate is not None:
            await self.before_estimate()
        return await super().estimate_cost_usd(request)

    async def submit(self, request: ImageRequest) -> SubmitResult:
        if self.before_submit is not None:
            await self.before_submit()
        result = await super().submit(request)
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        return result

    async def get_status(self, provider_request_id: str) -> StatusResult:
        if self.status_error is not None:
            raise self.status_error
        if self.pending_polls:
            self.pending_polls -= 1
            return StatusResult(provider_request_id, ProviderJobStatus.IN_PROGRESS)
        result = await super().get_status(provider_request_id)
        return dataclasses.replace(result, actual_cost_usd=self.actual_cost_usd)

    async def fetch_output(self, result: StatusResult) -> FetchedImage:
        if self.before_fetch is not None:
            await self.before_fetch()
        if self.fetch_error is not None:
            raise self.fetch_error
        return await super().fetch_output(result)


class IdempotencyConflictError(ProviderOutcomeUnknownError):
    code = "PROVIDER_IDEMPOTENCY_CONFLICT"


async def _spend(connection: AsyncConnection, provider: str = "simulator") -> tuple[int, int]:
    # (reserved, committed) micro-USD of the global spend counter; only the worker may read it.
    await as_role(connection, WORKER_ROLE)
    row = (
        await connection.execute(
            text(
                "SELECT reserved_usd_micros, committed_usd_micros FROM provider_spend"
                " WHERE provider = :p"
            ),
            {"p": provider},
        )
    ).one()
    return int(row.reserved_usd_micros), int(row.committed_usd_micros)


def _priced(cost: str = "0.094") -> ScriptedProvider:
    provider = ScriptedProvider()
    provider.cost_usd = Decimal(cost)
    return provider


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


async def test_provider_idempotency_key_is_derived_from_the_job(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    provider = SimulatedImageProvider()
    accepted, _ = await _load(sessions, world, job_id)

    await _run(sessions, provider, FakeSink())

    assert accepted.provider_idempotency_key == f"nrc-job-{job_id}"
    assert provider.submits == {f"nrc-job-{job_id}": 1}


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
    # A worker leases the event and dies; then its lease runs out.
    await as_role(connection, WORKER_ROLE)
    crashed = await connection.execute(
        text("SELECT * FROM lease_outbox_events('crashed-worker', 10, 60, ARRAY['job.run'])")
    )
    assert len(crashed.all()) == 1
    assert await _run(sessions, provider, sink) == 0
    await _expire_leases(connection, world)

    assert await _run(sessions, provider, sink) == 1
    assert await _run(sessions, provider, sink) == 0

    job, _ = await _load(sessions, world, job_id)
    event = await _event(connection)
    assert job.state is ExecutionState.SUCCEEDED
    assert len(sink.stored) == 1
    # The crashed lease is the one failure; the run that finished counts none.
    assert (event.attempts, event.failed_attempts) == (2, 1)


async def test_simulated_provider_failure_fails_the_job(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    await _run(sessions, SimulatedImageProvider(SimulatorScenario.PROVIDER_FAILED), FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert job.failure_code == "SIMULATED_FAILURE"
    # The provider bills completed generations only.
    assert reservation is ReservationState.RELEASED


async def test_simulated_rate_limit_retries_until_the_deadline_then_releases_credits(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = SimulatedImageProvider(SimulatorScenario.RATE_LIMITED)
    retries = get_settings().generation_max_attempts + 2

    for _ in range(retries):
        await _run(sessions, provider, FakeSink())
        await _requeue(connection, world)
    waiting, reserved = await _load(sessions, world, job_id)
    later = datetime.now(UTC) + timedelta(days=2)
    await run_once(sessions, {"simulator": provider}, FakeSink(), get_settings(), now=later)

    job, reservation = await _load(sessions, world, job_id)
    assert (waiting.state, waiting.stage) == (ExecutionState.RUNNING, "retry_wait")
    assert reserved is ReservationState.RESERVED
    assert provider.submits == {f"nrc-job-{job_id}": retries}
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "DEADLINE_EXCEEDED")
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


async def test_crash_after_submit_commit_reconciles_instead_of_resubmitting(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = ScriptedProvider()
    provider.submit_errors = [RuntimeError("worker died during the call")]
    await _run(sessions, provider, FakeSink())
    crashed, _ = await _load(sessions, world, job_id)
    assert (crashed.state, crashed.stage) == (ExecutionState.RUNNING, "submitting")

    await _expire_leases(connection, world)
    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    key = str(job.provider_idempotency_key)
    assert job.state is ExecutionState.SUCCEEDED
    assert job.provider_request_id == provider._request_ids[key]
    assert provider.submits == {key: 2}
    assert provider.estimates == 1


async def test_reconcile_retries_transient_errors_until_it_succeeds(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = ScriptedProvider()
    provider.submit_errors = [
        ProviderOutcomeUnknownError("Simulated lost response."),
        ProviderRateLimitedError("Simulated rate limit.", retry_after_seconds=1),
    ]
    await _run(sessions, provider, FakeSink())
    await _requeue(connection, world)

    await _run(sessions, provider, FakeSink())
    waiting, held = await _load(sessions, world, job_id)
    assert (waiting.state, waiting.failure_code) == (
        ExecutionState.UNKNOWN,
        "PROVIDER_RATE_LIMITED",
    )
    assert held is ReservationState.HELD
    assert (await _event(connection)).processed_at is None

    await _requeue(connection, world)
    await _run(sessions, provider, FakeSink())
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert provider.submits == {str(job.provider_idempotency_key): 3}


async def test_reconcile_stops_on_idempotency_conflict(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = ScriptedProvider()
    provider.submit_errors = [
        ProviderOutcomeUnknownError("Simulated lost response."),
        IdempotencyConflictError("Simulated key conflict."),
    ]
    await _run(sessions, provider, FakeSink())
    await _requeue(connection, world)

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (
        ExecutionState.UNKNOWN,
        "PROVIDER_IDEMPOTENCY_CONFLICT",
    )
    assert reservation is ReservationState.HELD
    assert (await _event(connection)).processed_at is not None


async def test_reconcile_never_resends_after_a_model_change(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = ScriptedProvider()
    provider.submit_errors = [ProviderOutcomeUnknownError("Simulated lost response.")]
    await _run(sessions, provider, FakeSink())
    provider.model = "simulated-image-v2"
    await _requeue(connection, world)

    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.UNKNOWN, "MODEL_CHANGED")
    assert sum(provider.submits.values()) == 1


async def test_status_error_after_acceptance_holds_credits(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    provider = ScriptedProvider()
    provider.status_error = ProviderConfigError("Simulated rejected key.")

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.UNKNOWN, "PROVIDER_CONFIG")
    assert job.provider_request_id is not None
    assert reservation is ReservationState.HELD


async def test_unusable_output_fails_and_keeps_credits_billed(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    provider = ScriptedProvider()
    provider.fetch_error = ProviderOutputError("Simulated oversize output.")

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_UNAVAILABLE")
    assert reservation is ReservationState.COMMITTED


async def test_lost_lease_abandons_the_step_without_touching_the_job(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = ScriptedProvider()

    async def another_worker_takes_over() -> None:
        await connection.execute(text("UPDATE outbox_events SET lease_owner = 'other-worker'"))

    provider.before_estimate = another_worker_takes_over

    await _run(sessions, provider, FakeSink())

    # The takeover shares the test's connection, so the abandoned step's rollback also undoes it;
    # the outcome checked here is that the job was left alone and nothing was submitted.
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.QUEUED
    assert not provider.submits
    assert "lease lost, event abandoned" in caplog.text


async def test_second_worker_cannot_lease_an_event_in_progress(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    # Even a misconfigured zero-second lease is clamped, so worker B finds nothing to do.
    settings = get_settings().model_copy(update={"worker_lease_seconds": 0})
    provider, sink = ScriptedProvider(), FakeSink()
    leased_by_b: list[int] = []

    async def worker_b_runs() -> None:
        leased_by_b.append(
            await run_once(sessions, {"simulator": provider}, sink, settings, owner="worker-b")
        )

    provider.before_submit = worker_b_runs

    await run_once(sessions, {"simulator": provider}, sink, settings, owner="worker-a")

    job, _ = await _load(sessions, world, job_id)
    assert leased_by_b == [0]
    assert sum(provider.submits.values()) == 1
    assert job.state is ExecutionState.SUCCEEDED


async def _fail_attempts(connection: AsyncConnection, world: World) -> None:
    # As if the event had already failed GENERATION_MAX_ATTEMPTS times.
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text("UPDATE outbox_events SET failed_attempts = :n"),
        {"n": get_settings().generation_max_attempts},
    )


async def test_event_past_attempt_cap_is_given_up(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    await _fail_attempts(connection, world)

    await _run(sessions, SimulatedImageProvider(), FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    event = await _event(connection)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "ATTEMPTS_EXHAUSTED")
    assert reservation is ReservationState.RELEASED
    assert event.processed_at is not None
    assert event.last_error_code == "ATTEMPTS_EXHAUSTED"


async def test_polls_of_a_long_generation_never_count_as_failures(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = ScriptedProvider()
    polls = get_settings().generation_max_attempts + 3
    provider.pending_polls = polls

    await _run(sessions, provider, FakeSink())
    for _ in range(polls):
        await _requeue(connection, world)
        await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    event = await _event(connection)
    assert job.state is ExecutionState.SUCCEEDED
    assert event.attempts == polls + 1
    assert event.failed_attempts == 0


async def test_event_without_a_handler_is_given_up_after_the_cap(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    settings = get_settings()

    no_kinds = {"job.run": JobRunHandler({})}
    for _ in range(settings.generation_max_attempts):
        assert await run_once(sessions, {}, FakeSink(), settings, event_handlers=no_kinds) == 1
        await connection.execute(text("UPDATE outbox_events SET available_at = clock_timestamp()"))

    event = await _event(connection)
    job, reservation = await _load(sessions, world, job_id)
    assert (event.last_error_code, event.failed_attempts) == (
        "NO_HANDLER",
        settings.generation_max_attempts,
    )
    assert event.processed_at is not None
    assert await run_once(sessions, {}, FakeSink(), settings, event_handlers=no_kinds) == 0
    # It never started, so it fails without billing and frees its slot.
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "NO_HANDLER")
    assert reservation is ReservationState.RELEASED


async def test_completed_output_never_stored_commits_the_credits_past_the_cap(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    reserved, committed = await _spend(connection)
    provider = _priced("0.094")
    provider.fetch_error = ProviderUnavailableError("Simulated storage outage.")
    await _run(sessions, provider, FakeSink())
    fetching, _ = await _load(sessions, world, job_id)
    await _fail_attempts(connection, world)
    await _requeue(connection, world)

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert (fetching.state, fetching.stage) == (ExecutionState.WAITING_PROVIDER, "fetching_output")
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "OUTPUT_UNAVAILABLE")
    assert reservation is ReservationState.COMMITTED
    assert await _spend(connection) == (reserved, committed + 94_000)
    assert await _awaiting_review(connection) == []


async def _awaiting_review(connection: AsyncConnection) -> list[tuple[UUID, UUID]]:
    # Lists across tenants, so it needs a role that reads tenants (the script uses the owner).
    async with app_session(connection) as db:
        return await jobs_awaiting_review(db)


async def test_unbilled_unknown_job_past_the_cap_is_listed_for_review(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider = ScriptedProvider()
    provider.submit_errors = [ProviderOutcomeUnknownError("Simulated lost response.")]
    await _run(sessions, provider, FakeSink())
    still_reconciling = await _awaiting_review(connection)
    await _fail_attempts(connection, world)
    await _requeue(connection, world)

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    event = await _event(connection)
    assert still_reconciling == []
    assert job.state is ExecutionState.UNKNOWN
    assert reservation is ReservationState.HELD
    assert (event.processed_at is not None, event.last_error_code) == (True, "ATTEMPTS_EXHAUSTED")
    assert await _awaiting_review(connection) == [(world.a.tenant_id, job_id)]
    # Given up: no worker leases it again.
    assert await _run(sessions, provider, FakeSink()) == 0


async def test_model_change_before_submit_fails_without_sending(
    sessions: Sessions, job_id: UUID, world: World
) -> None:
    provider = ScriptedProvider()
    provider.model = "simulated-image-v2"

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "MODEL_CHANGED")
    assert reservation is ReservationState.RELEASED
    assert provider.estimates == 0
    assert not provider.submits


class RecordingHandler:
    # A subscriber such as NRC-7's image check: records what it is given.

    def __init__(self) -> None:
        self.handled: list[LeasedEvent] = []

    async def handle(self, db: AsyncSession, event: LeasedEvent, step: Step) -> float | None:
        await step.heartbeat(db)
        self.handled.append(event)
        return None

    async def give_up(self, db: AsyncSession, event: LeasedEvent, code: str) -> None:
        raise AssertionError("not expected")


async def test_registering_a_handler_is_all_it_takes_to_receive_an_event_type(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    sink, settings = FakeSink(), get_settings()
    await _run(sessions, SimulatedImageProvider(), sink)
    checker = RecordingHandler()
    subscribed = {**EVENT_HANDLERS, "asset.generated": checker}

    without = await run_once(sessions, {"simulator": SimulatedImageProvider()}, sink, settings)
    leased = await run_once(
        sessions,
        {"simulator": SimulatedImageProvider()},
        sink,
        settings,
        event_handlers=subscribed,
    )

    assert (without, leased) == (0, 1)
    [event] = checker.handled
    assert (event.event_type, event.job_id, event.tenant_id) == (
        "asset.generated",
        job_id,
        world.a.tenant_id,
    )
    assert event.payload["event_version"] == 1
    assert event.payload["asset_id"] == str(sink.stored[0])
    processed = await connection.scalar(
        text("SELECT processed_at FROM outbox_events WHERE event_type = 'asset.generated'")
    )
    assert processed is not None


async def test_jobs_parked_without_storage_do_not_block_new_jobs(
    sessions: Sessions, client: httpx.AsyncClient, auth: AuthHeaders, world: World
) -> None:
    creative_id = await _creative(client, auth, world.a)
    for _ in range(get_settings().generation_max_active_jobs_per_tenant):
        assert (await _post_job(client, auth, world, creative_id)).status_code == 202
        # No sink until NRC-3: completed jobs park at the provider result.
        await run_once(sessions, {"simulator": SimulatedImageProvider()}, None, get_settings())

    third = await _post_job(client, auth, world, creative_id)

    assert third.status_code == 202


def test_http_timeout_must_be_below_the_lease() -> None:
    settings = get_settings().model_copy(
        update={"worker_lease_seconds": 5, "http_timeout_seconds": 5}
    )

    with pytest.raises(WorkerConfigError, match="HTTP_TIMEOUT_SECONDS"):
        build_providers(settings)


def _live(**overrides: object) -> Settings:
    return get_settings().model_copy(update={"generation_mode": "live", **overrides})


def test_live_mode_without_keys_refuses_to_start() -> None:
    settings = _live(higgsfield_api_key=None, higgsfield_model=None, higgsfield_output_hosts=())

    with pytest.raises(WorkerConfigError, match="HIGGSFIELD_API_KEY") as refused:
        build_providers(settings)

    assert "HIGGSFIELD_MODEL, HIGGSFIELD_OUTPUT_HOSTS" in str(refused.value)
    assert "validation-PASS gate" in str(refused.value)


def test_live_mode_with_stub_validation_gate_refuses_to_start() -> None:
    settings = _live(
        higgsfield_api_key=SecretStr("test-id:test-secret"),
        higgsfield_model="test/model",
        higgsfield_output_hosts=("images.example.test",),
    )

    with pytest.raises(WorkerConfigError, match=r"refused: the validation-PASS gate is still"):
        build_providers(settings)


@pytest.mark.parametrize("value", ["no-separator-value", ":only-secret", "only-id:", ""])
def test_malformed_api_key_is_rejected_without_echoing_it(value: str) -> None:
    settings = _live(higgsfield_api_key=SecretStr(value) if value else None)

    with pytest.raises(ValueError, match="HIGGSFIELD_API_KEY") as rejected:
        settings.higgsfield_credentials()

    assert not value or value not in str(rejected.value)


def test_api_key_splits_on_first_colon() -> None:
    key_id, key_secret = _live(
        higgsfield_api_key=SecretStr("id-1:se:cret")
    ).higgsfield_credentials()

    assert (key_id.get_secret_value(), key_secret.get_secret_value()) == ("id-1", "se:cret")


async def test_spend_past_the_global_budget_fails_before_submitting(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    budget = int(get_settings().generation_global_budget_usd * 1_000_000)
    # Earlier generations have used all but $0.05 of the budget.
    await as_role(connection, WORKER_ROLE)
    await connection.execute(
        text("UPDATE provider_spend SET committed_usd_micros = :c WHERE provider = 'simulator'"),
        {"c": budget - 50_000},
    )
    before = await _spend(connection)
    provider = _priced("0.094")

    await _run(sessions, provider, FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, job.failure_code) == (ExecutionState.FAILED, "BUDGET_EXHAUSTED")
    assert job.provider_spend_state is None
    assert reservation is ReservationState.RELEASED
    assert not provider.submits
    assert await _spend(connection) == before
    assert (await _event(connection)).processed_at is not None


async def test_completed_generation_commits_its_estimate_once(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    reserved, committed = await _spend(connection)

    await _run(sessions, _priced("0.094"), FakeSink())
    await _requeue(connection, world)
    await _run(sessions, _priced("0.094"), FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert job.provider_cost_actual_usd_micros is None
    assert await _spend(connection) == (reserved, committed + 94_000)


async def test_actual_cost_is_stored_and_committed_instead_of_the_estimate(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reserved, committed = await _spend(connection)
    provider = _priced("0.094")
    provider.actual_cost_usd = Decimal("0.08")

    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.SUCCEEDED
    assert job.provider_cost_actual_usd_micros == 80_000
    assert await _spend(connection) == (reserved, committed + 80_000)
    assert (
        f"provider cost job={job_id} provider_request={job.provider_request_id}"
        f" model=simulated-image-v1 estimate_usd_micros=94000 actual_usd_micros=80000"
    ) in caplog.text


async def test_provider_failure_releases_the_reservation(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    before = await _spend(connection)
    provider = SimulatedImageProvider(SimulatorScenario.PROVIDER_FAILED, cost_usd=Decimal("0.094"))

    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.FAILED
    assert job.provider_spend_state is SpendState.RELEASED
    assert await _spend(connection) == before


async def test_unknown_outcome_keeps_the_reservation_until_reconciled(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    reserved, committed = await _spend(connection)
    provider = SimulatedImageProvider(
        SimulatorScenario.TIMEOUT_AFTER_ACCEPT, cost_usd=Decimal("0.094")
    )

    await _run(sessions, provider, FakeSink())
    unknown, _ = await _load(sessions, world, job_id)
    held = await _spend(connection)
    await _requeue(connection, world)
    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert (unknown.state, unknown.provider_spend_state) == (
        ExecutionState.UNKNOWN,
        SpendState.RESERVED,
    )
    assert held == (reserved + 94_000, committed)
    assert job.state is ExecutionState.SUCCEEDED
    assert await _spend(connection) == (reserved, committed + 94_000)


async def test_retry_before_acceptance_keeps_one_reservation(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    reserved, committed = await _spend(connection)
    provider = _priced("0.094")
    provider.submit_errors = [ProviderRateLimitedError("Simulated rate limit.")]

    await _run(sessions, provider, FakeSink())
    waiting = await _spend(connection)
    await _requeue(connection, world)
    await _run(sessions, provider, FakeSink())

    job, _ = await _load(sessions, world, job_id)
    assert waiting == (reserved + 94_000, committed)
    assert job.state is ExecutionState.SUCCEEDED
    assert await _spend(connection) == (reserved, committed + 94_000)


async def test_asset_is_rolled_back_when_the_job_is_already_terminal(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    provider, sink = ScriptedProvider(), FakeSink()

    async def finished_elsewhere() -> None:
        # E.g. an operator or a racing webhook finished the job while the image was fetched.
        await connection.execute(text("UPDATE jobs SET state = 'CANCELLED'"))

    provider.before_fetch = finished_elsewhere

    await _run(sessions, provider, sink)

    job, _ = await _load(sessions, world, job_id)
    assets = await connection.scalar(text("SELECT count(*) FROM assets"))
    handoffs = await connection.scalar(
        text("SELECT count(*) FROM outbox_events WHERE event_type = 'asset.generated'")
    )
    assert len(sink.stored) == 1
    assert (job.state, job.result_asset_id) == (ExecutionState.CANCELLED, None)
    assert assets == 0
    assert handoffs == 0
    assert (await _event(connection)).processed_at is not None


@pytest.mark.parametrize("field", ["creative_id", "creative_version"])
async def test_asset_must_record_its_jobs_creative_version(
    sessions: Sessions, job_id: UUID, world: World, field: str
) -> None:
    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        job = await db.get(Job, job_id)
        assert job is not None
        lineage: dict[str, object] = {
            "creative_id": job.creative_id,
            "creative_version": job.creative_version,
        }
        lineage[field] = uuid4() if field == "creative_id" else job.creative_version + 1
        db.add(
            Asset(
                tenant_id=job.tenant_id,
                job_id=job.id,
                origin="generated",
                bucket="fake",
                s3_key="fake/lineage.png",
                mime_type="image/png",
                width_px=1,
                height_px=1,
                size_bytes=1,
                checksum_sha256="0" * 64,
                created_by=job.created_by,
                **lineage,
            )
        )
        with pytest.raises(IntegrityError, match="foreign key"):
            await db.flush()
