"""Ledger invariants of one generation job: credits, provider spend, assets and hand-off events.

Every crash-window test ends with `assert_ledger_consistent`, so a window that lost work or
charged twice fails on the same rules everywhere. The rules mirror what the code guarantees
(app.services.jobs.settle_reservation, app.services.spend, app.workers.connector._fail_with and
_park_unknown), not what would be nice to have.
"""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models.enums import ExecutionState
from app.models.jobs import ReservationState, SpendState
from app.services.assets import ASSET_GENERATED_EVENT
from app.services.jobs import RUN_EVENT
from tests.conftest import WORKER_ROLE, _spend, as_role

# Non-terminal states a job can be in. NEEDS_REVIEW is set by nothing in the worker, so no
# ledger rule is promised for it. CANCEL_REQUESTED keeps whatever the job held before the cancel.
_IN_FLIGHT = frozenset(
    {
        ExecutionState.QUEUED,
        ExecutionState.RUNNING,
        ExecutionState.WAITING_PROVIDER,
        ExecutionState.UNKNOWN,
        ExecutionState.CANCEL_REQUESTED,
    }
)
_CANCEL_STATES = frozenset({ExecutionState.CANCEL_REQUESTED, ExecutionState.CANCELLED})


async def assert_ledger_consistent(
    connection: AsyncConnection,
    tenant_id: UUID,
    job_id: UUID,
    *,
    spend_before: tuple[int, int],
) -> None:
    """Check one job's credits, spend, asset and event rows against its state.

    `spend_before` is (reserved, committed) of the job's provider counter, read with `_spend`
    before the scenario started, so "moved exactly once" is a comparison with that baseline.
    """
    await as_role(connection, WORKER_ROLE)
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )
    job = (
        await connection.execute(
            text(
                "SELECT state, provider, provider_spend_state, result_asset_id,"
                " provider_cost_estimate_usd_micros AS estimate,"
                " provider_cost_actual_usd_micros AS actual"
                " FROM jobs WHERE id = :j"
            ),
            {"j": job_id},
        )
    ).one()
    state = ExecutionState(job.state)
    spend_state = None if job.provider_spend_state is None else SpendState(job.provider_spend_state)

    # Exactly one reservation per job, never created twice by a retry or a replay.
    reservations = (
        await connection.execute(
            text(
                "SELECT state, units_reserved, units_committed"
                " FROM credit_reservations WHERE job_id = :j"
            ),
            {"j": job_id},
        )
    ).all()
    assert len(reservations) == 1, f"{len(reservations)} reservations for one job"
    reservation = ReservationState(reservations[0].state)
    where = f"job state={state.value} reservation={reservation.value} spend={spend_state}"

    # Credits: committed in full or not at all.
    expected_units = (
        reservations[0].units_reserved if reservation is ReservationState.COMMITTED else 0
    )
    assert reservations[0].units_committed == expected_units, where

    if state is ExecutionState.SUCCEEDED:
        assert reservation is ReservationState.COMMITTED, where
        assert spend_state is SpendState.COMMITTED, where
    elif state is ExecutionState.FAILED:
        # Billed (the provider completed it) commits both; unbilled releases credits and spend.
        # Never reserved or held: a failed job holds nothing.
        assert reservation in (ReservationState.COMMITTED, ReservationState.RELEASED), where
        if reservation is ReservationState.COMMITTED:
            assert spend_state is SpendState.COMMITTED, where
        else:
            # None: it failed before any spend was reserved (price too high, budget, content).
            assert spend_state in (SpendState.RELEASED, None), where
    elif state is ExecutionState.CANCELLED:
        # Billed (completed before the cancel) commits both; never sent or failed releases both;
        # outcome unknown holds the credits and keeps the spend reserved for an operator.
        if reservation is ReservationState.COMMITTED:
            assert spend_state is SpendState.COMMITTED, where
        elif reservation is ReservationState.HELD:
            assert spend_state is SpendState.RESERVED, where
        else:
            assert reservation is ReservationState.RELEASED, where
            assert spend_state in (SpendState.RELEASED, None), where
    elif state is ExecutionState.UNKNOWN:
        assert reservation is ReservationState.HELD, where
        assert spend_state is SpendState.RESERVED, where
    elif state in _IN_FLIGHT:
        # HELD is possible here too: a reconciled job that left UNKNOWN keeps its hold until it
        # settles.
        assert reservation in (ReservationState.RESERVED, ReservationState.HELD), where
        assert spend_state in (SpendState.RESERVED, None), where
        if state in (ExecutionState.WAITING_PROVIDER,):
            assert spend_state is SpendState.RESERVED, where
    else:
        raise AssertionError(f"no ledger rule for state {state.value}")

    # Provider spend moved once, by the amount this job's own state calls for.
    estimate = job.estimate or 0
    reserved_before, committed_before = spend_before
    expected = {
        None: (reserved_before, committed_before),
        SpendState.RELEASED: (reserved_before, committed_before),
        SpendState.RESERVED: (reserved_before + estimate, committed_before),
        SpendState.COMMITTED: (
            reserved_before,
            committed_before + (estimate if job.actual is None else job.actual),
        ),
    }[spend_state]
    assert await _spend(connection, job.provider) == expected, where

    # The generated image exists once, and only together with the SUCCEEDED job that names it.
    asset_ids = (
        (await connection.execute(text("SELECT id FROM assets WHERE job_id = :j"), {"j": job_id}))
        .scalars()
        .all()
    )
    if state is ExecutionState.SUCCEEDED:
        assert asset_ids == [job.result_asset_id], where
    else:
        assert asset_ids == [], where
        assert job.result_asset_id is None, where

    # One hand-off to image validation per stored image, one run event for the whole job.
    event_rows = await connection.execute(
        text(
            "SELECT event_type, count(*) AS n FROM outbox_events WHERE job_id = :j"
            " GROUP BY event_type"
        ),
        {"j": job_id},
    )
    events = {row.event_type: row.n for row in event_rows}
    # One run event per job, except that cancelling a job the worker had given up on adds one
    # (app.services.jobs._wake_worker); never more than one is pending.
    runs = events.get(RUN_EVENT, 0)
    assert runs == 1 or (state in _CANCEL_STATES and runs == 2), events
    pending = await connection.scalar(
        text(
            "SELECT count(*) FROM outbox_events WHERE job_id = :j AND event_type = :t"
            " AND processed_at IS NULL"
        ),
        {"j": job_id, "t": RUN_EVENT},
    )
    assert pending <= 1, f"{pending} pending run events"
    handoffs = 1 if state is ExecutionState.SUCCEEDED else 0
    assert events.get(ASSET_GENERATED_EVENT, 0) == handoffs, events
