# The tenant's active policy bundle: the highest approved version.

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import NotFoundError
from app.models import PolicyBundle
from app.schemas.policies import PolicyBundleRead, PolicyRule, PolicySummary


async def get_active_policy(db: AsyncSession, ctx: TenantContext) -> PolicyBundleRead:
    """Return the active policy with all its rules.

    Raises:
        NotFoundError: the tenant has no policy yet.
    """
    bundle = await db.scalar(
        select(PolicyBundle)
        .where(PolicyBundle.tenant_id == ctx.tenant_id)
        .order_by(PolicyBundle.version.desc())
        .limit(1)
    )
    if bundle is None:
        raise NotFoundError("This tenant has no active policy yet.")
    return PolicyBundleRead(
        policy_bundle_id=bundle.id,
        version=bundle.version,
        name=bundle.name,
        source_ref=bundle.source_ref,
        is_synthetic=bundle.is_synthetic,
        approved_by=bundle.approved_by,
        approved_at=bundle.approved_at,
        rules=[PolicyRule.model_validate(rule) for rule in bundle.rules],
    )


async def get_active_policy_summary(db: AsyncSession, tenant_id: UUID) -> PolicySummary | None:
    # Version and rule count only, so /context does not load every rule.
    row = (
        await db.execute(
            select(
                PolicyBundle.id.label("policy_bundle_id"),
                PolicyBundle.version,
                PolicyBundle.name,
                PolicyBundle.is_synthetic,
                func.jsonb_array_length(PolicyBundle.rules).label("rule_count"),
            )
            .where(PolicyBundle.tenant_id == tenant_id)
            .order_by(PolicyBundle.version.desc())
            .limit(1)
        )
    ).one_or_none()
    return None if row is None else PolicySummary.model_validate(row)
