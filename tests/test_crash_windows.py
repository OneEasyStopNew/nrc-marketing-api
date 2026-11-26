# NRC-6: a crash at any commit boundary of a generation job loses no work and never double charges.
#
# Every test injects one crash on the narrowest seam (a patched call raising a builtin, which the
# worker treats as a crash: the lease stays until it expires), reruns the worker, and ends with
# assert_ledger_consistent (credits, provider spend, asset and hand-off rows) plus a count of
# provider submits. Images are SIMULATED.
#
# Window -> test (names are prefixes):
#   a. accept transaction fails before commit
#        test_accept_failing_before_commit_*
#   b. worker dies after leasing, before the handler runs
#        test_worker_dying_after_leasing_*
#        (also test_job_worker.py::test_expired_lease_is_processed_once, no ledger check)
#   c. crash after the RUNNING/submitting commit
#        test_crash_after_the_submitting_commit_* (the provider saw the request, or not)
#        (also test_job_worker.py::test_crash_after_submit_commit_reconciles_instead_of_*)
#   d. provider accepted, crash before the WAITING_PROVIDER commit
#        test_crash_after_the_provider_accepted_*
#        (also test_higgsfield_worker.py::test_submit_5xx_after_acceptance_reconciles_to_one_*)
#   e. image stored, or SUCCEEDED written, crash before the commit
#        test_crash_after_storing_the_image_*, test_crash_after_the_succeeded_write_*
#   f. callback/ack: SUCCEEDED and the event's processed_at are one commit, so a crash in the
#      acknowledgement rolls both back; an event replayed after success changes nothing
#        test_crash_while_acknowledging_*, test_event_replayed_after_success_*

from collections.abc import Collection
from decimal import Decimal
from uuid import UUID

import httpx
import pytest
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.core.config import get_settings
from app.models import Job
from app.models.enums import ExecutionState, JobStage
from app.models.jobs import ReservationState, SpendState
from app.services import idempotency
from app.services.assets import emit_asset_generated
from app.services.jobs import CREATE_ROUTE, transition
from app.services.providers.base import FetchedImage
from app.workers import connector
from app.workers.connector import run_once
from tests.conftest import (
    APP_ROLE,
    WORKER_ROLE,
    AuthHeaders,
    FakeSink,
    ScriptedProvider,
    Sessions,
    World,
    _creative,
    _event,
    _load,
    _post_job,
    _spend,
    as_role,
)
from tests.ledger import assert_ledger_consistent

# Under the $0.09 per-image cap (NRC-10), so jobs reach submit.
ESTIMATE_MICROS = 84_000
IDEMPOTENCY_KEY = "crash-window-accept-0001"


@pytest.fixture
async def creative_id(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> str:
    return await _creative(client, auth, world.a)


@pytest.fixture
async def job_id(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, creative_id: str
) -> UUID:
    response = await _post_job(client, auth, world, creative_id)
    assert response.status_code == 202
    return UUID(response.json()["job_id"])


def _priced() -> ScriptedProvider:
    # A price, so a double charge would show in provider_spend.
    provider = ScriptedProvider()
    provider.cost_usd = Decimal("0.084")
    return provider


async def _run(sessions: Sessions, provider: ScriptedProvider, sink: FakeSink) -> int:
    return await run_once(sessions, {"simulator": provider}, sink, get_settings())


async def _expire_leases(connection: AsyncConnection, world: World) -> None:
    # The crashed worker's lease runs out, so the event is leased again.
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text("UPDATE outbox_events SET leased_until = clock_timestamp() - interval '1 second'")
    )


class StoreThenCrashSink(FakeSink):
    # The image and its asset row are stored, then the worker dies before anything is committed.

    def __init__(self) -> None:
        super().__init__()
        self.crashes_left = 1

    async def store(
        self, db: AsyncSession, job: Job, image: FetchedImage, provider_name: str, model: str
    ) -> UUID:
        asset_id = await super().store(db, job, image, provider_name, model)
        if self.crashes_left:
            self.crashes_left -= 1
            raise RuntimeError("worker died after storing the image")
        return asset_id


async def test_accept_failing_before_commit_leaves_nothing_and_the_retry_creates_one_job(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    creative_id: str,
    connection: AsyncConnection,
    sessions: Sessions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spend_before = await _spend(connection)
    real_complete = idempotency.complete
    attempts: list[str] = []

    async def complete_then_lose_the_connection(
        db: AsyncSession, tenant_id: UUID, route: str, key: str, resource_id: UUID
    ) -> None:
        await real_complete(db, tenant_id, route, key, resource_id)
        if not attempts:
            attempts.append(key)
            raise RuntimeError("database connection lost before commit")

    monkeypatch.setattr(idempotency, "complete", complete_then_lose_the_connection)

    with pytest.raises(RuntimeError, match="connection lost before commit"):
        await _post_job(client, auth, world, creative_id, key=IDEMPOTENCY_KEY)

    async def rows() -> tuple[int, int, int, int]:
        # Worker role for the job tables, the API's role for the idempotency record.
        await as_role(connection, WORKER_ROLE)
        await connection.execute(
            text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
        )
        counts = (
            await connection.execute(
                text(
                    "SELECT (SELECT count(*) FROM jobs),"
                    " (SELECT count(*) FROM credit_reservations),"
                    " (SELECT count(*) FROM outbox_events WHERE event_type = 'job.run')"
                )
            )
        ).one()
        await as_role(connection, APP_ROLE)
        keys = await connection.scalar(
            text("SELECT count(*) FROM idempotency_records WHERE route = :r AND key = :k"),
            {"r": CREATE_ROUTE, "k": IDEMPOTENCY_KEY},
        )
        return counts[0], counts[1], counts[2], int(keys or 0)

    # Nothing of the failed accept survived: no job, reservation, event or claimed key.
    assert await rows() == (0, 0, 0, 0)
    assert await _spend(connection) == spend_before

    retry = await _post_job(client, auth, world, creative_id, key=IDEMPOTENCY_KEY)

    assert retry.status_code == 202
    assert "Idempotent-Replayed" not in retry.headers
    assert await rows() == (1, 1, 1, 1)
    # The one job then runs normally, so the accepted work was not lost.
    provider, sink = _priced(), FakeSink()
    assert await _run(sessions, provider, sink) == 1
    job, _ = await _load(sessions, world, UUID(retry.json()["job_id"]))
    assert job.state is ExecutionState.SUCCEEDED
    assert sum(provider.submits.values()) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job.id, spend_before=spend_before)


async def test_worker_dying_after_leasing_before_the_handler_runs_submits_once_after_expiry(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    real_run_handler = connector._run_handler
    crashes: list[UUID] = []

    async def die_before_the_handler(
        run: connector._Run, event: Row[tuple[UUID, UUID, UUID, str]]
    ) -> None:
        if not crashes:
            crashes.append(event.event_id)
            raise RuntimeError("worker died after leasing the event")
        await real_run_handler(run, event)

    monkeypatch.setattr(connector, "_run_handler", die_before_the_handler)

    assert await _run(sessions, provider, sink) == 1

    leased = await _event(connection)
    untouched, _ = await _load(sessions, world, job_id)
    assert untouched.state is ExecutionState.QUEUED
    assert not provider.submits
    assert leased.processed_at is None
    assert leased.lease_owner is not None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)
    # While the lease is valid nobody else may take the event.
    assert await _run(sessions, provider, sink) == 0

    await _expire_leases(connection, world)
    assert await _run(sessions, provider, sink) == 1
    assert await _run(sessions, provider, sink) == 0

    job, reservation = await _load(sessions, world, job_id)
    event = await _event(connection)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    assert provider.submits == {f"nrc-job-{job_id}": 1}
    assert provider.estimates == 1
    assert len(sink.stored) == 1
    # The dead lease is the one failure; the run that finished counts none.
    assert (event.attempts, event.failed_attempts) == (2, 1)
    assert event.processed_at is not None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


@pytest.mark.parametrize("provider_saw_the_request", [True, False])
async def test_crash_after_the_submitting_commit_reconciles_with_the_same_key_and_charges_once(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    provider_saw_the_request: bool,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    if provider_saw_the_request:
        # The provider recorded the request, but the worker never got the answer.
        provider.submit_errors = [RuntimeError("worker died during the submit call")]
    else:

        async def die_before_the_call() -> None:
            raise RuntimeError("worker died before the submit call")

        provider.before_submit = die_before_the_call
    key = f"nrc-job-{job_id}"

    await _run(sessions, provider, sink)

    crashed, _ = await _load(sessions, world, job_id)
    assert (crashed.state, crashed.stage) == (ExecutionState.RUNNING, JobStage.SUBMITTING)
    # The spend reservation was committed with the submitting stage, before the call.
    assert crashed.provider_spend_state is SpendState.RESERVED
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)
    if not provider_saw_the_request:
        provider.before_submit = None

    await _expire_leases(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    # Only the job's own key ever reached the provider, so one request id and one charge.
    assert set(provider.submits) == {key}
    assert provider.submits[key] == (2 if provider_saw_the_request else 1)
    assert list(provider._request_ids) == [key]
    assert job.provider_request_id == provider._request_ids[key]
    assert provider.estimates == 1
    assert len(sink.stored) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_crash_after_the_provider_accepted_reconciles_to_the_same_request(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    key = f"nrc-job-{job_id}"
    real_transition = transition
    crashes: list[str] = []

    async def write_waiting_provider_then_die(
        db: AsyncSession,
        job: UUID,
        *,
        allowed_from: Collection[ExecutionState],
        to: ExecutionState,
        **values: object,
    ) -> bool:
        moved = await real_transition(db, job, allowed_from=allowed_from, to=to, **values)
        if "provider_request_id" in values and not crashes:
            # The UPDATE ran in this transaction, but the worker dies before it commits.
            crashes.append(str(values["provider_request_id"]))
            raise RuntimeError("worker died before committing WAITING_PROVIDER")
        return moved

    # The handler calls the name imported into the worker module, so that is the one replaced.
    monkeypatch.setattr("app.workers.connector.transition", write_waiting_provider_then_die)

    await _run(sessions, provider, sink)

    crashed, _ = await _load(sessions, world, job_id)
    accepted_id = provider._request_ids[key]
    assert crashes == [accepted_id]
    assert (crashed.state, crashed.stage) == (ExecutionState.RUNNING, JobStage.SUBMITTING)
    assert crashed.provider_request_id is None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)

    await _expire_leases(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    # The resend with the same key got the original request back: one charge, one image.
    assert provider.submits == {key: 2}
    assert list(provider._request_ids.values()) == [accepted_id]
    assert job.provider_request_id == accepted_id
    assert len(sink.stored) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_crash_after_storing_the_image_keeps_one_asset_and_commits_once(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), StoreThenCrashSink()
    key = f"nrc-job-{job_id}"

    await _run(sessions, provider, sink)

    crashed, reservation = await _load(sessions, world, job_id)
    # The provider completed (billed) but nothing of the stored image was committed.
    assert (crashed.state, crashed.stage) == (
        ExecutionState.WAITING_PROVIDER,
        JobStage.FETCHING_OUTPUT,
    )
    assert reservation is ReservationState.RESERVED
    assert crashed.result_asset_id is None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)

    await _expire_leases(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    # The crashed attempt's asset row was rolled back; the second one is the job's result.
    assert len(sink.stored) == 2
    assert job.result_asset_id == sink.stored[-1]
    assert provider.submits == {key: 1}
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_crash_after_the_succeeded_write_rolls_everything_back_and_commits_once(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    key = f"nrc-job-{job_id}"
    real_emit = emit_asset_generated
    crashes: list[UUID] = []

    async def die_after_the_ledger_writes(db: AsyncSession, asset_id: UUID) -> None:
        # Reached after SUCCEEDED, the credit commit and the spend commit were written, but
        # before the one commit that makes them visible.
        if not crashes:
            crashes.append(asset_id)
            raise RuntimeError("worker died before committing SUCCEEDED")
        await real_emit(db, asset_id)

    monkeypatch.setattr("app.workers.connector.emit_asset_generated", die_after_the_ledger_writes)

    await _run(sessions, provider, sink)

    crashed, reservation = await _load(sessions, world, job_id)
    assert (crashed.state, crashed.stage) == (
        ExecutionState.WAITING_PROVIDER,
        JobStage.FETCHING_OUTPUT,
    )
    assert (reservation, crashed.provider_spend_state) == (
        ReservationState.RESERVED,
        SpendState.RESERVED,
    )
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)

    await _expire_leases(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    assert job.result_asset_id == sink.stored[-1]
    assert provider.submits == {key: 1}
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_crash_while_acknowledging_the_event_rolls_back_succeeded_with_it(
    sessions: Sessions,
    job_id: UUID,
    world: World,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SUCCEEDED and the event's processed_at are written in one transaction, so a crash in the
    # acknowledgement can never leave a finished job with an unprocessed event, or the reverse.
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    key = f"nrc-job-{job_id}"
    real_finish = connector._finish_event
    crashes: list[UUID] = []

    async def die_while_acknowledging(
        db: AsyncSession,
        run: connector._Run,
        event_id: UUID,
        retry_after: float | None,
        *,
        error_code: str | None = None,
        failed: bool = False,
    ) -> None:
        if not crashes:
            crashes.append(event_id)
            raise RuntimeError("worker died while acknowledging the event")
        await real_finish(db, run, event_id, retry_after, error_code=error_code, failed=failed)

    monkeypatch.setattr(connector, "_finish_event", die_while_acknowledging)

    await _run(sessions, provider, sink)

    crashed, _ = await _load(sessions, world, job_id)
    event = await _event(connection)
    assert crashed.state is ExecutionState.WAITING_PROVIDER
    assert event.processed_at is None
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)

    await _expire_leases(connection, world)
    await _run(sessions, provider, sink)

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    assert (await _event(connection)).processed_at is not None
    assert provider.submits == {key: 1}
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)


async def test_event_replayed_after_success_neither_charges_nor_stores_again(
    sessions: Sessions, job_id: UUID, world: World, connection: AsyncConnection
) -> None:
    spend_before = await _spend(connection)
    provider, sink = _priced(), FakeSink()
    fetches: list[int] = []

    async def count_fetch() -> None:
        fetches.append(1)

    provider.before_fetch = count_fetch
    key = f"nrc-job-{job_id}"
    await _run(sessions, provider, sink)
    done, _ = await _load(sessions, world, job_id)
    assert done.state is ExecutionState.SUCCEEDED

    # As if the acknowledgement had been lost: the finished job's event is due again.
    await connection.execute(
        text(
            "UPDATE outbox_events SET processed_at = NULL, leased_until = NULL,"
            " lease_owner = NULL, available_at = clock_timestamp() - interval '1 second'"
            " WHERE event_type = 'job.run'"
        )
    )
    assert await _run(sessions, provider, sink) == 1

    job, reservation = await _load(sessions, world, job_id)
    assert (job.state, reservation) == (ExecutionState.SUCCEEDED, ReservationState.COMMITTED)
    assert job.result_asset_id == done.result_asset_id
    assert job.completed_at == done.completed_at
    assert (await _event(connection)).processed_at is not None
    assert provider.submits == {key: 1}
    assert provider.estimates == 1
    assert fetches == [1]
    assert len(sink.stored) == 1
    await assert_ledger_consistent(connection, world.a.tenant_id, job_id, spend_before=spend_before)
