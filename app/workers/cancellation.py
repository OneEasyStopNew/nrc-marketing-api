"""What the connector worker does with a CANCEL_REQUESTED job (NRC-6).

The API records the request (app.services.jobs.request_cancel); the worker finishes it on the
job's next "job.run" event, before any other step of the job:
- never sent to the provider: CANCELLED, credits and provider spend released;
- accepted (the provider request id is known): only the status is polled, never a resubmit or
  a fetch. Completed means billed: CANCELLED with the cost recorded and credits and spend
  committed, but the image is never stored or published. Failed releases them. Still running
  keeps polling until the accepted grace window ends;
- possibly sent with no request id (the submit's outcome is unknown), or no way left to learn
  the outcome: CANCELLED with credits HELD and spend reserved, for an operator
  (scripts/list_unknown_jobs.py lists them). No same-key resend: it would bill a generation the
  user cancelled if the provider never got the request.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import Job
from app.models.enums import ExecutionState, JobStage
from app.models.jobs import ReservationState
from app.services import spend
from app.services.jobs import settle_reservation, transition
from app.services.providers.base import (
    TERMINAL_PROVIDER_STATUSES,
    ImageProvider,
    ProviderError,
    ProviderJobStatus,
    ProviderOutcomeUnknownError,
)
from app.workers.common import backoff_seconds

logger = logging.getLogger(__name__)

# failure_code of a CANCELLED job: what the cancel cost.
CANCELLED: Final = "CANCELLED"
CANCELLED_AFTER_COMPLETION: Final = "CANCELLED_AFTER_COMPLETION"
CANCELLED_OUTCOME_UNKNOWN: Final = "CANCELLED_OUTCOME_UNKNOWN"

_MESSAGES: Final = {
    CANCELLED: "The job was cancelled before the provider produced an image.",
    CANCELLED_AFTER_COMPLETION: "The job was cancelled after the provider produced and billed "
    "an image; the image was not kept.",
    CANCELLED_OUTCOME_UNKNOWN: "The job was cancelled before the provider's outcome was known; "
    "its credits stay held until it is reviewed.",
}
# Stages where the provider may have the request although no request id was recorded.
_MAYBE_SENT: Final = frozenset({JobStage.SUBMITTING, JobStage.RECONCILE})
# Stages where the provider is known to have completed (billed) the generation.
_COMPLETED: Final = frozenset({JobStage.FETCHING_OUTPUT, JobStage.OUTPUT_READY})


class _Settled(Enum):
    BILLED = "billed"
    RELEASED = "released"
    HELD = "held"


async def finish_cancel(
    db: AsyncSession,
    job: Job,
    provider: ImageProvider | None,
    *,
    settings: Settings,
    now: datetime,
    event_attempts: int,
    heartbeat: Callable[[AsyncSession], Awaitable[None]],
) -> float | None:
    """Finish one CANCEL_REQUESTED job. Returns None when done, else seconds until the next try.

    Raises LeaseLostError (through heartbeat) like every other step.
    """
    request_id = job.provider_request_id
    if request_id is None:
        if job.stage in _MAYBE_SENT:
            await _end(db, job, CANCELLED_OUTCOME_UNKNOWN, _Settled.HELD)
        else:
            await _end(db, job, CANCELLED, _Settled.RELEASED)
        return None
    if provider is None:
        # Live mode never falls back to the simulator, so the outcome can't be checked here.
        await _end(db, job, CANCELLED_OUTCOME_UNKNOWN, _Settled.HELD)
        return None
    await heartbeat(db)
    try:
        status = await provider.get_status(request_id)
    except ProviderError as error:
        if error.retryable or isinstance(error, ProviderOutcomeUnknownError):
            # The attempt cap ends these (give_up_cancel), never a release.
            return max(error.retry_after_seconds or 0.0, backoff_seconds(event_attempts, settings))
        await _end(db, job, CANCELLED_OUTCOME_UNKNOWN, _Settled.HELD)
        return None
    if status.status not in TERMINAL_PROVIDER_STATUSES:
        grace = timedelta(seconds=settings.generation_accepted_grace_seconds)
        if now >= job.deadline_at + grace:
            # Same rule as an uncancelled job: it may still complete and bill.
            await _end(db, job, CANCELLED_OUTCOME_UNKNOWN, _Settled.HELD)
            return None
        return backoff_seconds(event_attempts, settings)
    if status.status is ProviderJobStatus.COMPLETED:
        actual = (
            None if status.actual_cost_usd is None else spend.usd_micros(status.actual_cost_usd)
        )
        await _end(db, job, CANCELLED_AFTER_COMPLETION, _Settled.BILLED, actual_usd_micros=actual)
    else:
        # The provider bills completed generations only.
        await _end(db, job, CANCELLED, _Settled.RELEASED)
    return None


async def give_up_cancel(db: AsyncSession, job: Job) -> None:
    # The cancel failed past the attempt cap: settle from what is known, never release credits
    # of a job the provider may have.
    if job.stage in _COMPLETED:
        await _end(db, job, CANCELLED_AFTER_COMPLETION, _Settled.BILLED)
    elif job.provider_request_id is not None or job.stage in _MAYBE_SENT:
        await _end(db, job, CANCELLED_OUTCOME_UNKNOWN, _Settled.HELD)
    else:
        await _end(db, job, CANCELLED, _Settled.RELEASED)


async def _end(
    db: AsyncSession,
    job: Job,
    code: str,
    settled: _Settled,
    *,
    actual_usd_micros: int | None = None,
) -> None:
    values: dict[str, object] = {"failure_code": code, "failure_message": _MESSAGES[code]}
    if actual_usd_micros is not None:
        values["provider_cost_actual_usd_micros"] = actual_usd_micros
    moved = await transition(
        db,
        job.id,
        allowed_from={ExecutionState.CANCEL_REQUESTED},
        to=ExecutionState.CANCELLED,
        **values,
    )
    if not moved:
        return
    if settled is _Settled.HELD:
        # Spend stays reserved with the credits, until an operator settles the job.
        await settle_reservation(db, job.id, ReservationState.HELD)
        logger.error(
            "job cancelled with outcome unknown job=%s tenant=%s provider_request=%s stage=%s",
            job.id,
            job.tenant_id,
            job.provider_request_id,
            job.stage,
        )
        return
    billed = settled is _Settled.BILLED
    await settle_reservation(
        db, job.id, ReservationState.COMMITTED if billed else ReservationState.RELEASED
    )
    await spend.settle(db, job.id, billed=billed, actual_usd_micros=actual_usd_micros)
    logger.info("job cancelled job=%s tenant=%s code=%s", job.id, job.tenant_id, code)
