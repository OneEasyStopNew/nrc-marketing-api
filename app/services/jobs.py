# Generation jobs: admission, persisted reads and guarded state transitions.

import logging
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.errors import (
    DependencyUnavailableError,
    GenerationLimitError,
    JobNotCancellableError,
    NotFoundError,
)
from app.db.session import bind_tenant
from app.models import Creative, CreativeVersion, CreditReservation, Job, OutboxEvent, Tenant
from app.models.enums import FORMAT_SPECS, ExecutionState, FormatSpec, JobStage
from app.models.jobs import ReservationState
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobKind, JobRead
from app.services import generation_checks, idempotency
from app.services.providers.base import HIGGSFIELD_NAME
from app.services.providers.simulator import SIMULATOR_MODEL, SIMULATOR_NAME

logger = logging.getLogger(__name__)

CREATE_ROUTE: Final = "POST /creative-jobs"
RUN_EVENT: Final = "job.run"
TERMINAL_STATES: Final = frozenset(
    {ExecutionState.SUCCEEDED, ExecutionState.FAILED, ExecutionState.CANCELLED}
)
NON_TERMINAL_STATES: Final = frozenset(ExecutionState) - TERMINAL_STATES
# States a cancel request may move a job out of: every non-terminal one but CANCEL_REQUESTED.
_CANCELLABLE_FROM: Final = NON_TERMINAL_STATES - {ExecutionState.CANCEL_REQUESTED}
# SQLSTATE lock_not_available: a lock_timeout ran out.
_LOCK_NOT_AVAILABLE: Final = "55P03"
# Jobs that take an active-generation slot. Parked and UNKNOWN jobs don't: UNKNOWN credits
# stay HELD, so the daily credit ceiling still bounds them. CANCEL_REQUESTED does: the provider
# may still be running until the worker confirms the cancel, so a request must not free a slot.
_TAKES_SLOT: Final = or_(
    Job.state.in_((ExecutionState.QUEUED, ExecutionState.RUNNING)),
    # A cancel of a parked (output_ready) or UNKNOWN (reconcile) job takes no slot it didn't hold.
    and_(
        Job.state == ExecutionState.CANCEL_REQUESTED,
        or_(
            Job.stage.is_(None),
            Job.stage.not_in((JobStage.OUTPUT_READY, JobStage.RECONCILE)),
        ),
    ),
    and_(
        Job.state == ExecutionState.WAITING_PROVIDER,
        Job.stage.is_distinct_from(JobStage.OUTPUT_READY),
    ),
)
# The "daily" in GENERATION_DAILY_CREDIT_CEILING_PER_TENANT: part of that setting's meaning.
_CREDIT_WINDOW: Final = timedelta(hours=24)


async def accept_creative_job(
    db: AsyncSession,
    ctx: TenantContext,
    data: CreativeJobCreate,
    idempotency_key: str,
    settings: Settings,
) -> tuple[JobAccepted, bool]:
    """Queue one generation job in a single transaction. Returns the job and whether it's a replay.

    Raises:
        IdempotencyKeyReusedError: the key was used before with a different payload.
        NotFoundError, VersionConflictError, GenerationNotEligibleError and
            ValidationNotPassedError: see generation_checks.
        GenerationLimitError: too many active jobs, or the daily credit ceiling would be passed.
    """
    fingerprint = idempotency.request_hash(ctx.principal.sub, data)
    existing_id = await idempotency.claim(
        db, ctx.tenant_id, CREATE_ROUTE, idempotency_key, fingerprint
    )
    if existing_id is not None:
        state = (
            await db.execute(
                select(Job.state).where(Job.tenant_id == ctx.tenant_id, Job.id == existing_id)
            )
        ).scalar_one()
        return JobAccepted(job_id=existing_id, state=state), True

    provider, model = _provider_for(settings)
    eligible = await generation_checks.assert_generation_eligible(
        db, ctx, data.creative_id, data.creative_version
    )
    # Serialise admission per tenant, so two requests cannot both take the last slot.
    await db.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(f"generation:{ctx.tenant_id}", 0)))
    )
    await _check_limits(db, ctx.tenant_id, settings)

    job_id = uuid4()
    db.add(
        Job(
            id=job_id,
            tenant_id=ctx.tenant_id,
            kind=data.kind.value,
            creative_id=data.creative_id,
            creative_version=data.creative_version,
            study_snapshot_id=eligible.study_snapshot_id,
            study_snapshot_version=eligible.study_snapshot_version,
            state=ExecutionState.QUEUED,
            attempt_count=0,
            max_attempts=settings.generation_max_attempts,
            deadline_at=datetime.now(UTC) + timedelta(seconds=settings.generation_deadline_seconds),
            provider=provider,
            model=model,
            provider_idempotency_key=provider_idempotency_key(job_id),
            created_by=ctx.principal.sub,
        )
    )
    # Flush the parent first: there is no ORM relationship to order the inserts.
    await db.flush()
    db.add_all(
        [
            CreditReservation(
                tenant_id=ctx.tenant_id,
                job_id=job_id,
                units_reserved=settings.generation_credits_per_image,
                units_committed=0,
                state=ReservationState.RESERVED,
            ),
            OutboxEvent(
                tenant_id=ctx.tenant_id,
                job_id=job_id,
                event_type=RUN_EVENT,
                payload={"job_id": str(job_id)},
                attempts=0,
            ),
        ]
    )
    await idempotency.complete(db, ctx.tenant_id, CREATE_ROUTE, idempotency_key, job_id)
    await db.commit()
    logger.info("job accepted job=%s tenant=%s", job_id, ctx.tenant_id)
    return JobAccepted(job_id=job_id, state=ExecutionState.QUEUED), False


@dataclass(frozen=True, slots=True)
class JobContent:
    # What a generation job was accepted for: its creative version's prompt and format.
    prompt: str
    spec: FormatSpec


class JobContentMissingError(Exception):
    # The job's creative version is gone or not visible: no retry can fix it, so the worker
    # ends the job at once instead of treating it as a crash and retrying to the attempt cap.
    pass


async def job_content(db: AsyncSession, job: Job) -> JobContent:
    # Runs under the job's tenant (RLS); the version is fixed on the job when it is accepted.
    row = (
        await db.execute(
            select(CreativeVersion.prompt, CreativeVersion.post_format).where(
                CreativeVersion.tenant_id == job.tenant_id,
                CreativeVersion.creative_id == job.creative_id,
                CreativeVersion.version == job.creative_version,
            )
        )
    ).one_or_none()
    if row is None:
        raise JobContentMissingError("The job's creative version could not be found.")
    return JobContent(prompt=row.prompt, spec=FORMAT_SPECS[row.post_format])


def provider_idempotency_key(job_id: UUID) -> str:
    # Derived from the job, so Higgsfield support can trace a request back to it.
    return f"nrc-job-{job_id}"


async def get_job(db: AsyncSession, ctx: TenantContext, job_id: UUID) -> JobRead:
    # Persisted state only; reading a job never calls a provider.
    row = (
        await db.execute(
            select(
                Job.id.label("job_id"),
                Job.kind,
                Job.creative_id,
                Job.creative_version,
                Job.state,
                Job.stage,
                Job.provider_request_id,
                Job.failure_code,
                Job.failure_message,
                Job.result_asset_id,
                Job.created_at,
                Job.updated_at,
                Job.completed_at,
            )
            .join(
                Creative, (Creative.tenant_id == Job.tenant_id) & (Creative.id == Job.creative_id)
            )
            .where(
                Job.tenant_id == ctx.tenant_id,
                Job.id == job_id,
                Creative.study_id.in_(ctx.study_ids),
            )
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Job not found.")
    return JobRead.model_validate(row)


async def request_cancel(
    db: AsyncSession, ctx: TenantContext, job_id: UUID, settings: Settings
) -> JobRead:
    """Record a durable cancel request; the worker acts on it (stops, settles credits).

    State-idempotent: a job already CANCEL_REQUESTED or CANCELLED is returned unchanged. This
    never touches the credit reservation or the provider.

    Raises:
        NotFoundError: no such job, or not visible to the caller.
        JobNotCancellableError: the job already SUCCEEDED or FAILED.
        DependencyUnavailableError: the job row stayed locked past
            JOB_CANCEL_LOCK_TIMEOUT_SECONDS (retryable).
    """
    job = await get_job(db, ctx, job_id)
    if job.state not in TERMINAL_STATES:
        try:
            moved = await _record_cancel(db, ctx.tenant_id, job_id, settings)
        except DBAPIError as error:
            sqlstate = getattr(getattr(error, "orig", None), "sqlstate", None)
            if sqlstate != _LOCK_NOT_AVAILABLE:
                raise
            await db.rollback()
            logger.exception(
                "job cancel lock timeout job=%s tenant=%s sqlstate=%s",
                job_id,
                ctx.tenant_id,
                sqlstate,
            )
            raise DependencyUnavailableError("The job is busy. Try again in a moment.") from None
        if moved:
            await db.commit()
            logger.info(
                "job cancel requested job=%s tenant=%s by=%s",
                job_id,
                ctx.tenant_id,
                ctx.principal.sub,
            )
            return await get_job(db, ctx, job_id)
    # Not moved: every other non-terminal state is accepted, so the job is already
    # CANCEL_REQUESTED or terminal (possibly since it was read).
    job = await get_job(db, ctx, job_id)
    if job.state in (ExecutionState.CANCEL_REQUESTED, ExecutionState.CANCELLED):
        return job
    raise JobNotCancellableError(
        "This job has already finished.", details={"state": job.state.value}
    )


async def _record_cancel(
    db: AsyncSession, tenant_id: UUID, job_id: UUID, settings: Settings
) -> bool:
    """Move the job to CANCEL_REQUESTED and make sure a run event will finish the cancel.

    Locks in the worker's order, the run event first and the job second, or a cancel landing
    while a worker is inside a provider call deadlocks with it (the worker holds its event lock
    through the call, then updates the job). An event a worker holds is skipped, never waited on.

    Whether a run event is still pending is decided after the job update, not before: the
    worker's post-step check reads the job with FOR SHARE (JobRunHandler.handle), so either it
    sees this cancel and keeps its event, or it has committed (event processed) before this
    update can go through, and the check below adds a new event. Returns whether the job moved.
    """
    timeout_ms = round(settings.job_cancel_lock_timeout_seconds * 1000)
    await db.execute(select(func.set_config("lock_timeout", f"{timeout_ms}ms", True)))
    pending = select(OutboxEvent.id).where(
        OutboxEvent.tenant_id == tenant_id,
        OutboxEvent.job_id == job_id,
        OutboxEvent.event_type == RUN_EVENT,
        OutboxEvent.processed_at.is_(None),
    )
    claimed = await db.scalar(pending.limit(1).with_for_update(skip_locked=True))
    if not await transition(
        db, job_id, allowed_from=_CANCELLABLE_FROM, to=ExecutionState.CANCEL_REQUESTED
    ):
        return False
    if claimed is not None:
        # Due now; the lease columns are not ours to touch.
        await db.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == claimed)
            .values(available_at=func.now())
            .execution_options(synchronize_session=False)
        )
    elif not await db.scalar(select(pending.exists())):
        # None pending (e.g. an UNKNOWN job the worker gave up on, or a worker that closed its
        # event just before this update): add one to finish the cancel. A pending one a worker
        # holds is kept by that worker.
        db.add(
            OutboxEvent(
                tenant_id=tenant_id,
                job_id=job_id,
                event_type=RUN_EVENT,
                payload={"job_id": str(job_id)},
                attempts=0,
            )
        )
    return True


async def transition(
    db: AsyncSession,
    job_id: UUID,
    *,
    allowed_from: Collection[ExecutionState],
    to: ExecutionState,
    **values: object,
) -> bool:
    """Move a job to `to` only if it is still in one of `allowed_from`. Returns whether it moved.

    Terminal states are never accepted as a source, so a finished job cannot regress and a late
    or duplicate event changes nothing.
    """
    sources = set(allowed_from) - TERMINAL_STATES
    if not sources:
        return False
    if to in TERMINAL_STATES:
        values["completed_at"] = func.now()
    moved = await db.scalar(
        update(Job)
        .where(Job.id == job_id, Job.state.in_(sources))
        .values(state=to, updated_at=func.now(), **values)
        .returning(Job.id)
        .execution_options(synchronize_session=False)
    )
    if moved is not None:
        logger.info("job state changed job=%s state=%s", job_id, to.value)
    return moved is not None


async def settle_reservation(db: AsyncSession, job_id: UUID, to: ReservationState) -> None:
    # HELD only from RESERVED; COMMITTED and RELEASED are final and never change again.
    sources = (
        {ReservationState.RESERVED}
        if to is ReservationState.HELD
        else {ReservationState.RESERVED, ReservationState.HELD}
    )
    values: dict[str, object] = {"state": to, "updated_at": func.now()}
    if to is ReservationState.COMMITTED:
        values["units_committed"] = CreditReservation.units_reserved
    await db.execute(
        update(CreditReservation)
        .where(CreditReservation.job_id == job_id, CreditReservation.state.in_(sources))
        .values(**values)
        .execution_options(synchronize_session=False)
    )


@dataclass(frozen=True, slots=True)
class AwaitingReview:
    # A job whose credits wait for an operator. failure_code says why (e.g. ATTEMPTS_EXHAUSTED,
    # DEADLINE_EXCEEDED, CANCELLED_OUTCOME_UNKNOWN); ids and codes only, never provider text.
    tenant_id: UUID
    job_id: UUID
    state: ExecutionState
    failure_code: str | None


async def awaiting_review_details(db: AsyncSession) -> list[AwaitingReview]:
    """Jobs whose credits wait for an operator, across tenants, with the reason they stopped.

    UNKNOWN jobs the worker has stopped working on (no unprocessed outbox event is left), and
    CANCELLED jobs whose outcome was unknown at cancel time (credits HELD): the provider may have
    billed either, so their credits stay HELD until an operator settles each one
    (app.services.job_operations). Binds each tenant in turn, so it works under row-level
    security with any role that can read tenants, jobs and credit reservations.
    """
    pending = (
        select(OutboxEvent.id)
        .where(
            OutboxEvent.tenant_id == Job.tenant_id,
            OutboxEvent.job_id == Job.id,
            OutboxEvent.processed_at.is_(None),
        )
        .correlate(Job)
    )
    held = (
        select(CreditReservation.id)
        .where(
            CreditReservation.tenant_id == Job.tenant_id,
            CreditReservation.job_id == Job.id,
            CreditReservation.state == ReservationState.HELD,
        )
        .correlate(Job)
    )
    found: list[AwaitingReview] = []
    for tenant_id in (await db.scalars(select(Tenant.id).order_by(Tenant.id))).all():
        await bind_tenant(db, tenant_id)
        rows = await db.execute(
            select(Job.id, Job.state, Job.failure_code)
            .where(
                Job.tenant_id == tenant_id,
                or_(
                    and_(Job.state == ExecutionState.UNKNOWN, ~exists(pending)),
                    and_(Job.state == ExecutionState.CANCELLED, exists(held)),
                ),
            )
            .order_by(Job.updated_at)
        )
        found.extend(AwaitingReview(tenant_id, row.id, row.state, row.failure_code) for row in rows)
    return found


def _provider_for(settings: Settings) -> tuple[str, str]:
    # Live mode never falls back to the simulator.
    if settings.generation_mode == "simulator":
        return SIMULATOR_NAME, SIMULATOR_MODEL
    if settings.higgsfield_model is None:
        raise DependencyUnavailableError("Image generation is not configured.")
    return HIGGSFIELD_NAME, settings.higgsfield_model


async def _check_limits(db: AsyncSession, tenant_id: UUID, settings: Settings) -> None:
    active = await db.scalar(
        select(func.count())
        .select_from(Job)
        .where(
            Job.tenant_id == tenant_id,
            Job.kind == JobKind.IMAGE_GENERATION.value,
            _TAKES_SLOT,
        )
    )
    if (active or 0) >= settings.generation_max_active_jobs_per_tenant:
        raise GenerationLimitError(
            "Too many image generations are running. Try again when one finishes.",
            details={"reason": "ACTIVE_JOB_LIMIT"},
        )
    used = await db.scalar(
        select(func.coalesce(func.sum(CreditReservation.units_reserved), 0)).where(
            CreditReservation.tenant_id == tenant_id,
            CreditReservation.state != ReservationState.RELEASED,
            CreditReservation.created_at >= func.now() - _CREDIT_WINDOW,
        )
    )
    if (used or 0) + settings.generation_credits_per_image > (
        settings.generation_daily_credit_ceiling_per_tenant
    ):
        raise GenerationLimitError(
            "The daily image generation allowance is used up.",
            details={"reason": "CREDIT_CEILING"},
        )
