"""Global provider spend guard (GENERATION_GLOBAL_BUDGET_USD) across all tenants.

A job reserves its cost estimate in provider_spend in the same transaction that moves it to its
submitting stage, so nothing reaches the provider past the budget. It is settled once, in the
transaction whose guarded job transition applied: released when the provider produced nothing
(failed, rejected, canceled, or never submitted), committed when it completed. UNKNOWN jobs keep
their reservation. jobs.provider_spend_state makes each step happen at most once per job.
"""

import logging
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Job, ProviderSpend
from app.models.jobs import SpendState

logger = logging.getLogger(__name__)


async def reserve(
    db: AsyncSession, job_id: UUID, provider: str, usd_micros: int, budget_usd_micros: int
) -> bool:
    """Reserve the job's estimate. Returns False when the budget would be passed.

    One conditional UPDATE: the row lock it takes serialises concurrent workers, and the WHERE
    is evaluated again on the committed row, so two jobs can never both take the last slot.
    A job that already holds a reservation (a retry before acceptance) keeps it.
    """
    state = await db.scalar(select(Job.provider_spend_state).where(Job.id == job_id))
    if state is SpendState.RESERVED:
        return True
    counters = (
        await db.execute(
            update(ProviderSpend)
            .where(
                ProviderSpend.provider == provider,
                ProviderSpend.reserved_usd_micros + ProviderSpend.committed_usd_micros + usd_micros
                <= budget_usd_micros,
            )
            .values(
                reserved_usd_micros=ProviderSpend.reserved_usd_micros + usd_micros,
                updated_at=func.now(),
            )
            .returning(ProviderSpend.reserved_usd_micros, ProviderSpend.committed_usd_micros)
            .execution_options(synchronize_session=False)
        )
    ).one_or_none()
    if counters is None:
        logger.warning(
            "provider budget exhausted job=%s provider=%s estimate_usd_micros=%s",
            job_id,
            provider,
            usd_micros,
        )
        return False
    await db.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(provider_spend_state=SpendState.RESERVED)
        .execution_options(synchronize_session=False)
    )
    logger.info(
        "provider spend reserved job=%s provider=%s estimate_usd_micros=%s "
        "reserved_usd_micros=%s committed_usd_micros=%s",
        job_id,
        provider,
        usd_micros,
        counters.reserved_usd_micros,
        counters.committed_usd_micros,
    )
    return True


async def settle(
    db: AsyncSession, job_id: UUID, *, billed: bool, actual_usd_micros: int | None = None
) -> None:
    # Commit (actual cost if known, else the estimate) or release the job's reservation, once.
    to = SpendState.COMMITTED if billed else SpendState.RELEASED
    job = (
        await db.execute(
            update(Job)
            .where(Job.id == job_id, Job.provider_spend_state == SpendState.RESERVED)
            .values(provider_spend_state=to)
            .returning(Job.provider, Job.provider_cost_estimate_usd_micros)
            .execution_options(synchronize_session=False)
        )
    ).one_or_none()
    if job is None:
        return
    reserved = job.provider_cost_estimate_usd_micros or 0
    values: dict[str, object] = {
        "reserved_usd_micros": ProviderSpend.reserved_usd_micros - reserved,
        "updated_at": func.now(),
    }
    if billed:
        cost = reserved if actual_usd_micros is None else actual_usd_micros
        values["committed_usd_micros"] = ProviderSpend.committed_usd_micros + cost
    await db.execute(
        update(ProviderSpend)
        .where(ProviderSpend.provider == job.provider)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    logger.info("provider spend %s job=%s provider=%s", to.value.lower(), job_id, job.provider)
