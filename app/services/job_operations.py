"""Operator replay and settlement of jobs that wait for an operator (NRC-6), with an audit row.

Two kinds of job end up here with their credits HELD, because the provider may have billed them:
- UNKNOWN jobs the worker gave up on (no unprocessed job.run event is left);
- CANCELLED jobs whose outcome was unknown when they were cancelled (failure_code
  CANCELLED_OUTCOME_UNKNOWN).
`replay` hands the first kind back to the worker. `settle` ends either kind once the operator has
checked the outcome with the provider. Both write one job_operator_actions row in the same
transaction as the change, and commit once. Callers run as marketing_worker.
"""

import logging
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import exists, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import JobOperationNotAllowedError, NotFoundError
from app.db.session import bind_tenant
from app.models import CreditReservation, Job, JobOperatorAction, OutboxEvent
from app.models.enums import ExecutionState
from app.models.jobs import FAILURE_MESSAGE_MAX_CHARS, OperatorAction, ReservationState
from app.services import spend
from app.services.jobs import RUN_EVENT, TERMINAL_STATES, settle_reservation, transition

logger = logging.getLogger(__name__)

# failure_code of a job an operator settled (a CANCELLED job keeps its own code).
OPERATOR_SETTLED_BILLED: Final = "OPERATOR_SETTLED_BILLED"
OPERATOR_SETTLED_NOT_BILLED: Final = "OPERATOR_SETTLED_NOT_BILLED"
_SETTLED_MESSAGES: Final = MappingProxyType(
    {
        True: "An operator checked the outcome with the provider: it billed this job.",
        False: "An operator checked the outcome with the provider: it did not bill this job.",
    }
)

# details.reason of JobOperationNotAllowedError.
JOB_FINISHED: Final = "JOB_FINISHED"
NOT_UNKNOWN: Final = "NOT_UNKNOWN"
NOT_AWAITING_OPERATOR: Final = "NOT_AWAITING_OPERATOR"
RUN_EVENT_PENDING: Final = "RUN_EVENT_PENDING"
NOTHING_HELD: Final = "NOTHING_HELD"
STATE_CHANGED: Final = "STATE_CHANGED"
INVALID_ACTUAL_COST: Final = "INVALID_ACTUAL_COST"


def _refuse(job: Job | None, reason: str) -> JobOperationNotAllowedError:
    state = None if job is None else job.state.value
    return JobOperationNotAllowedError(
        "This operation is not allowed for the job in its current state.",
        details={"state": state, "reason": reason},
    )


async def _locked_job(db: AsyncSession, tenant_id: UUID, job_id: UUID) -> Job:
    # The row lock makes two operators acting on one job take turns (the second sees the result).
    job = await db.scalar(
        select(Job)
        .where(Job.tenant_id == tenant_id, Job.id == job_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if job is None:
        raise NotFoundError("Job not found.")
    return job


async def _locked_reservation_state(db: AsyncSession, job_id: UUID) -> ReservationState:
    state = await db.scalar(
        select(CreditReservation.state).where(CreditReservation.job_id == job_id).with_for_update()
    )
    if state is None:
        # Every accepted job has one reservation; a job without one is not ours to settle.
        raise NotFoundError("Job not found.")
    return state


async def _run_event_pending(db: AsyncSession, tenant_id: UUID, job_id: UUID) -> bool:
    return bool(
        await db.scalar(
            select(
                exists().where(
                    OutboxEvent.tenant_id == tenant_id,
                    OutboxEvent.job_id == job_id,
                    OutboxEvent.event_type == RUN_EVENT,
                    OutboxEvent.processed_at.is_(None),
                )
            )
        )
    )


async def _require_stopped_unknown(db: AsyncSession, job: Job) -> None:
    # UNKNOWN and no unprocessed run event: the worker has stopped working on it.
    if job.state in TERMINAL_STATES:
        raise _refuse(job, JOB_FINISHED)
    if job.state is not ExecutionState.UNKNOWN:
        raise _refuse(job, NOT_UNKNOWN)
    if await _run_event_pending(db, job.tenant_id, job.id):
        raise _refuse(job, RUN_EVENT_PENDING)


async def replay(
    db: AsyncSession,
    tenant_id: UUID,
    job_id: UUID,
    *,
    operator: str,
    reason: str,
    settings: Settings,
) -> UUID:
    """Give an UNKNOWN job the worker stopped on a new run event. Returns that event's id.

    The worker then reconciles with the job's own provider idempotency key (never a new one), so
    the provider returns the original request instead of billing a second generation. The
    deadline restarts, or the worker would give up again at once. Credits stay HELD.

    Raises:
        NotFoundError: no such job under this tenant.
        JobOperationNotAllowedError: the job is finished, not UNKNOWN, or a run event is pending.
    """
    await bind_tenant(db, tenant_id)
    job = await _locked_job(db, tenant_id, job_id)
    await _require_stopped_unknown(db, job)
    reservation = await _locked_reservation_state(db, job_id)

    event_id = uuid4()
    deadline_at = datetime.now(UTC) + timedelta(seconds=settings.generation_deadline_seconds)
    moved = await transition(
        db,
        job_id,
        allowed_from={ExecutionState.UNKNOWN},
        to=ExecutionState.UNKNOWN,
        deadline_at=deadline_at,
    )
    if not moved:
        raise _refuse(job, STATE_CHANGED)
    db.add(
        OutboxEvent(
            id=event_id,
            tenant_id=tenant_id,
            job_id=job_id,
            event_type=RUN_EVENT,
            payload={"job_id": str(job_id)},
            attempts=0,
        )
    )
    # The event row must exist before the audit row names it.
    await db.flush()
    db.add(
        JobOperatorAction(
            tenant_id=tenant_id,
            job_id=job_id,
            action=OperatorAction.REPLAY,
            operator=operator,
            reason=reason,
            job_state_before=ExecutionState.UNKNOWN,
            job_state_after=ExecutionState.UNKNOWN,
            reservation_state_before=reservation.value,
            reservation_state_after=reservation.value,
            outbox_event_id=event_id,
        )
    )
    await db.commit()
    logger.info(
        "operator action=%s job=%s tenant=%s event=%s",
        OperatorAction.REPLAY.value,
        job_id,
        tenant_id,
        event_id,
    )
    return event_id


async def settle(
    db: AsyncSession,
    tenant_id: UUID,
    job_id: UUID,
    *,
    billed: bool,
    actual_usd_micros: int | None,
    operator: str,
    reason: str,
) -> None:
    """End a job whose credits are HELD, after the operator checked the outcome with the provider.

    Billed commits the credits and the provider spend (the actual cost when given, else the
    estimate); not billed releases both. An UNKNOWN job the worker stopped on becomes FAILED with
    failure_code OPERATOR_SETTLED_BILLED or OPERATOR_SETTLED_NOT_BILLED. A CANCELLED job with
    HELD credits stays CANCELLED (a terminal state never changes). Once settled nothing is HELD,
    so a second settle is refused.

    Raises:
        NotFoundError: no such job under this tenant.
        JobOperationNotAllowedError: the job is not one of the two kinds above, a run event is
            still pending, nothing is HELD, or actual_usd_micros is negative or given unbilled.
    """
    if actual_usd_micros is not None and (not billed or actual_usd_micros < 0):
        raise _refuse(None, INVALID_ACTUAL_COST)
    await bind_tenant(db, tenant_id)
    job = await _locked_job(db, tenant_id, job_id)
    reservation = await _locked_reservation_state(db, job_id)
    state_before = job.state
    if state_before is ExecutionState.UNKNOWN:
        await _require_stopped_unknown(db, job)
    elif state_before is ExecutionState.CANCELLED:
        pass
    elif state_before in TERMINAL_STATES:
        raise _refuse(job, JOB_FINISHED)
    else:
        raise _refuse(job, NOT_AWAITING_OPERATOR)
    if reservation is not ReservationState.HELD:
        raise _refuse(job, NOTHING_HELD)

    if state_before is ExecutionState.UNKNOWN:
        code = OPERATOR_SETTLED_BILLED if billed else OPERATOR_SETTLED_NOT_BILLED
        values: dict[str, object] = {
            "failure_code": code,
            "failure_message": _SETTLED_MESSAGES[billed][:FAILURE_MESSAGE_MAX_CHARS],
        }
        if actual_usd_micros is not None:
            values["provider_cost_actual_usd_micros"] = actual_usd_micros
        moved = await transition(
            db,
            job_id,
            allowed_from={ExecutionState.UNKNOWN},
            to=ExecutionState.FAILED,
            **values,
        )
        if not moved:
            raise _refuse(job, STATE_CHANGED)
        state_after = ExecutionState.FAILED
    else:
        # transition() never leaves a terminal state, so the cost is recorded directly.
        if actual_usd_micros is not None:
            await db.execute(
                update(Job)
                .where(Job.id == job_id, Job.state == ExecutionState.CANCELLED)
                .values(provider_cost_actual_usd_micros=actual_usd_micros, updated_at=func.now())
                .execution_options(synchronize_session=False)
            )
        state_after = ExecutionState.CANCELLED

    reservation_after = ReservationState.COMMITTED if billed else ReservationState.RELEASED
    await settle_reservation(db, job_id, reservation_after)
    await spend.settle(db, job_id, billed=billed, actual_usd_micros=actual_usd_micros)
    action = OperatorAction.SETTLE_BILLED if billed else OperatorAction.SETTLE_NOT_BILLED
    db.add(
        JobOperatorAction(
            tenant_id=tenant_id,
            job_id=job_id,
            action=action,
            operator=operator,
            reason=reason,
            job_state_before=state_before,
            job_state_after=state_after,
            reservation_state_before=reservation.value,
            reservation_state_after=reservation_after.value,
        )
    )
    await db.commit()
    logger.info(
        "operator action=%s job=%s tenant=%s event=%s", action.value, job_id, tenant_id, None
    )
