# Generation jobs: admission, persisted reads and guarded state transitions.

import logging
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.errors import DependencyUnavailableError, GenerationLimitError, NotFoundError
from app.db.session import bind_tenant
from app.models import Creative, CreditReservation, Job, OutboxEvent, Tenant
from app.models.enums import ExecutionState, JobStage
from app.models.jobs import ReservationState
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobKind, JobRead
from app.services import eligibility, idempotency
from app.services.providers.base import HIGGSFIELD_NAME
from app.services.providers.simulator import SIMULATOR_MODEL, SIMULATOR_NAME

logger = logging.getLogger(__name__)

CREATE_ROUTE: Final = "POST /creative-jobs"
RUN_EVENT: Final = "job.run"
TERMINAL_STATES: Final = frozenset(
    {ExecutionState.SUCCEEDED, ExecutionState.FAILED, ExecutionState.CANCELLED}
)
NON_TERMINAL_STATES: Final = frozenset(ExecutionState) - TERMINAL_STATES
# Jobs that take an active-generation slot. Parked and UNKNOWN jobs don't: UNKNOWN credits
# stay HELD, so the daily credit ceiling still bounds them.
_TAKES_SLOT: Final = or_(
    Job.state.in_((ExecutionState.QUEUED, ExecutionState.RUNNING)),
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
        NotFoundError / VersionConflictError / GenerationNotEligibleError: see eligibility.
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
    eligible = await eligibility.assert_generation_eligible(
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


async def jobs_awaiting_review(db: AsyncSession) -> list[tuple[UUID, UUID]]:
    """(tenant id, job id) of UNKNOWN jobs the worker has stopped working on, across tenants.

    The worker gives up on them (no unprocessed outbox event is left) while the provider may have
    billed them, so their credits stay HELD until an operator settles each one. Binds each tenant
    in turn, so it works under row-level security with any role that can read tenants and jobs.
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
    found: list[tuple[UUID, UUID]] = []
    for tenant_id in (await db.scalars(select(Tenant.id).order_by(Tenant.id))).all():
        await bind_tenant(db, tenant_id)
        job_ids = await db.scalars(
            select(Job.id)
            .where(
                Job.tenant_id == tenant_id,
                Job.state == ExecutionState.UNKNOWN,
                ~exists(pending),
            )
            .order_by(Job.updated_at)
        )
        found.extend((tenant_id, job_id) for job_id in job_ids)
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
