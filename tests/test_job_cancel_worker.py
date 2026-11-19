# Job cancellation, worker side (NRC-6): what the connector worker does with CANCEL_REQUESTED.
#
# A job the provider never got ends CANCELLED with everything released. A job the provider
# accepted is only polled for its outcome: a late result is recorded (cost, credits committed if
# billed) but never fetched or stored. A job that may have been sent but has no request id is
# never resent: it ends CANCELLED with its credits held for an operator. Images are SIMULATED.

from datetime import timedelta
from decimal import Decimal
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.models.enums import ExecutionState, JobStage
from app.models.jobs import ReservationState, SpendState
from app.services.jobs import jobs_awaiting_review
from app.services.providers.base import (
    ProviderConfigError,
    ProviderOutcomeUnknownError,
    ProviderUnavailableError,
)
from app.services.providers.simulator import SimulatorScenario
from app.workers.connector import run_once
from tests.conftest import (
    AuthHeaders,
    FakeSink,
    ScriptedProvider,
    Sessions,
    World,
    _creative,
    _event,
    _load,
    _post_job,
    _requeue,
    _spend,
    app_session,
)
from tests.ledger import assert_ledger_consistent

ESTIMATE_MICROS = 94_000
ACTUAL_MICROS = 81_000


@pytest.fixture
async def job_id(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> UUID:
    creative_id = await _creative(client, auth, world.a)
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    return UUID(response.json()["job_id"])


def _priced() -> ScriptedProvider:
    # A price, so a wrong charge or release would show in provider_spend.
    provider = ScriptedProvider()
    provider.cost_usd = Decimal("0.094")
    return provider


async def _run(sessions: Sessions, provider: ScriptedProvider, sink: FakeSink) -> int:
    return await run_once(sessions, {"simulator": provider}, sink, get_settings())


async def _cancel(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, job_id: UUID
) -> httpx.Response:
    return await client.post(
        f"/v1/marketing/jobs/{job_id}/cancel", headers=auth("creator-a", world.a.tenant_id)
    )


async def _cancel_in_flight(connection: AsyncConnection, job_id: UUID) -> None:
    # The cancel request lands while a worker is mid-step (as the API's guarded update would).
    await connection.execute(
        text("UPDATE jobs SET state = 'CANCEL_REQUESTED' WHERE id = :j AND state <> 'CANCELLED'"),
        {"j": job_id},
    )


async def _awaiting_review(connection: AsyncConnection) -> list[tuple[UUID, UUID]]:
    # Lists across tenants, so it needs a role that reads tenants (the script uses the owner).
    async with app_session(connection) as db:
        return await jobs_awaiting_review(db)


async def _accepted(
    sessions: Sessions, provider: ScriptedProvider, sink: FakeSink, world: World, job_id: UUID
) -> None:
    # Submitted and accepted; the first status check still reports it in progress.
    provider.pending_polls = 1
    await _run(sessions, provider, sink)
    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.WAITING_PROVIDER
    assert job.provider_request_id is not None


async def test_queued_job_is_cancelled_without_any_provider_call(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()

    assert (await _cancel(client, auth, world, job_id)).status_code == 202
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED"
    assert job.completed_at is not None
    assert reservation is ReservationState.RELEASED
    assert (provider.estimates, sum(provider.submits.values())) == (0, 0)
    assert (await _event(connection)).processed_at is not None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_late_completion_after_cancel_is_billed_but_never_fetched_or_stored(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    provider.actual_cost_usd = Decimal("0.081")
    await _accepted(sessions, provider, sink, world, job_id)

    assert (await _cancel(client, auth, world, job_id)).status_code == 202
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_AFTER_COMPLETION"
    assert job.provider_cost_actual_usd_micros == ACTUAL_MICROS
    assert reservation is ReservationState.COMMITTED
    assert job.provider_spend_state is SpendState.COMMITTED
    assert sink.stored == []
    assert sum(provider.submits.values()) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancelled_job_keeps_polling_while_the_provider_is_still_running(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)
    provider.pending_polls = 1

    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCEL_REQUESTED
    assert reservation is ReservationState.RESERVED
    assert (await _event(connection)).processed_at is None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)

    # The provider then reports a failed generation: nothing was billed.
    provider.scenario = SimulatorScenario.PROVIDER_FAILED
    await _requeue(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED"
    assert reservation is ReservationState.RELEASED
    assert sink.stored == []
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancelled_job_still_running_past_grace_ends_with_credits_held_for_an_operator(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)
    job, _ = await _load(sessions, world, job_id)
    settings = get_settings()
    later = job.deadline_at + timedelta(seconds=settings.generation_accepted_grace_seconds)
    provider.pending_polls = 1

    await run_once(sessions, {"simulator": provider}, sink, settings, now=later)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_OUTCOME_UNKNOWN"
    assert reservation is ReservationState.HELD
    assert job.provider_spend_state is SpendState.RESERVED
    assert (world.a.tenant_id, job_id) in await _awaiting_review(connection)
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_unknown_job_is_cancelled_with_credits_held_and_never_resent(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    provider.submit_errors = [ProviderOutcomeUnknownError("Simulated lost response.")]
    await _run(sessions, provider, sink)
    job, _ = await _load(sessions, world, job_id)
    assert (job.state, job.provider_request_id) == (ExecutionState.UNKNOWN, None)

    assert (await _cancel(client, auth, world, job_id)).status_code == 202
    await _requeue(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_OUTCOME_UNKNOWN"
    assert reservation is ReservationState.HELD
    assert job.provider_spend_state is SpendState.RESERVED
    # The same-key resend would bill a generation the user cancelled if the provider never got it.
    assert sum(provider.submits.values()) == 1
    assert (world.a.tenant_id, job_id) in await _awaiting_review(connection)
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancel_landing_during_submit_keeps_the_request_id_and_records_the_outcome(
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()

    async def cancel_now() -> None:
        await _cancel_in_flight(connection, job_id)

    provider.before_submit = cancel_now
    await _run(sessions, provider, sink)

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCEL_REQUESTED
    assert job.provider_request_id is not None
    # The event stays due, so the cancel is finished instead of the job being left behind.
    assert (await _event(connection)).processed_at is None

    provider.before_submit = None
    await _requeue(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_AFTER_COMPLETION"
    assert reservation is ReservationState.COMMITTED
    assert sink.stored == []
    assert sum(provider.submits.values()) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancel_landing_during_fetch_discards_the_image_and_bills_once(
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()

    async def cancel_now() -> None:
        await _cancel_in_flight(connection, job_id)

    provider.before_fetch = cancel_now
    await _run(sessions, provider, sink)

    job, _ = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCEL_REQUESTED
    assert job.stage is JobStage.FETCHING_OUTPUT
    assert (await _event(connection)).processed_at is None

    provider.before_fetch = None
    await _requeue(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_AFTER_COMPLETION"
    assert reservation is ReservationState.COMMITTED
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancel_given_up_after_the_attempt_cap_holds_credits_of_an_accepted_job(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)
    await connection.execute(text("UPDATE outbox_events SET failed_attempts = 99"))

    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    # Accepted, so possibly billed: never released by giving up.
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_OUTCOME_UNKNOWN"
    assert reservation is ReservationState.HELD
    assert (await _event(connection)).last_error_code == "ATTEMPTS_EXHAUSTED"
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancel_given_up_before_any_submit_releases_everything(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    await _cancel(client, auth, world, job_id)
    await connection.execute(text("UPDATE outbox_events SET failed_attempts = 99"))

    await _run(sessions, _priced(), FakeSink())

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED"
    assert reservation is ReservationState.RELEASED
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_status_outage_on_a_cancelled_job_retries_within_grace(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)
    provider.status_error = ProviderUnavailableError("Simulated status outage.")

    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCEL_REQUESTED
    assert reservation is ReservationState.RESERVED
    assert (await _event(connection)).processed_at is None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_status_outage_past_grace_ends_the_cancel_with_credits_held(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    # A returned retry never counts as a failed attempt, so only the grace window ends these.
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)
    provider.status_error = ProviderUnavailableError("Simulated status outage.")
    job, _ = await _load(sessions, world, job_id)
    settings = get_settings()
    later = job.deadline_at + timedelta(seconds=settings.generation_accepted_grace_seconds)

    await run_once(sessions, {"simulator": provider}, sink, settings, now=later)

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_OUTCOME_UNKNOWN"
    assert reservation is ReservationState.HELD
    assert (await _event(connection)).processed_at is not None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_status_refused_on_a_cancelled_job_holds_its_credits(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)
    provider.status_error = ProviderConfigError("Simulated rejected key.")

    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    # Accepted, so possibly billed: an unreadable outcome is never released.
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_OUTCOME_UNKNOWN"
    assert reservation is ReservationState.HELD
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_cancelled_job_whose_provider_is_not_configured_holds_its_credits(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    sessions: Sessions,
    connection: AsyncConnection,
    job_id: UUID,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    await _accepted(sessions, provider, sink, world, job_id)
    await _cancel(client, auth, world, job_id)

    # The worker now runs without the job's provider (live mode never uses the simulator).
    await run_once(sessions, {}, sink, get_settings())

    job, reservation = await _load(sessions, world, job_id)
    assert job.state is ExecutionState.CANCELLED
    assert job.failure_code == "CANCELLED_OUTCOME_UNKNOWN"
    assert reservation is ReservationState.HELD
    assert sum(provider.submits.values()) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)
