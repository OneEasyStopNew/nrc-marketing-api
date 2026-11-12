# Memberships, tenant context and the studies a member may use.

from collections.abc import Collection
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext, capabilities_for, parse_roles
from app.auth.tokens import Principal
from app.core.errors import PermissionDeniedError
from app.db.session import bind_tenant
from app.models import Membership, MembershipStudy, Study, StudySnapshot, Tenant
from app.models.enums import FORMAT_SPECS, Language
from app.schemas.tenancy import (
    ContextRead,
    MeRead,
    PostFormatOption,
    StudyFacts,
    StudySnapshotRead,
    TenantMembershipRead,
)
from app.services import policies


async def get_me(db: AsyncSession, principal: Principal) -> MeRead:
    rows = await db.execute(
        select(Membership.tenant_id, Tenant.name, Membership.roles)
        .join(Tenant, Tenant.id == Membership.tenant_id)
        .where(Membership.user_sub == principal.sub, Membership.is_active.is_(True))
        .order_by(Tenant.name)
    )
    memberships = [
        _membership_read(tenant_id, tenant_name, role_names)
        for tenant_id, tenant_name, role_names in rows
    ]
    return MeRead(sub=principal.sub, username=principal.username, memberships=memberships)


def _membership_read(
    tenant_id: UUID, tenant_name: str, role_names: list[str]
) -> TenantMembershipRead:
    roles = parse_roles(role_names)
    return TenantMembershipRead(
        tenant_id=tenant_id,
        tenant_name=tenant_name,
        roles=sorted(roles),
        capabilities=sorted(capabilities_for(roles)),
    )


async def load_tenant_context(
    db: AsyncSession, principal: Principal, tenant_id: UUID
) -> TenantContext:
    """Check the caller's current membership, then bind the session to that tenant.

    Raises:
        PermissionDeniedError: the caller has no active membership in the tenant.
    """
    role_names = await db.scalar(
        select(Membership.roles).where(
            Membership.tenant_id == tenant_id,
            Membership.user_sub == principal.sub,
            Membership.is_active.is_(True),
        )
    )
    if role_names is None:
        raise PermissionDeniedError("You are not an active member of this tenant.")

    await bind_tenant(db, tenant_id)
    study_ids = await db.scalars(
        select(MembershipStudy.study_id).where(
            MembershipStudy.tenant_id == tenant_id, MembershipStudy.user_sub == principal.sub
        )
    )
    roles = parse_roles(role_names)
    return TenantContext(
        tenant_id=tenant_id,
        principal=principal,
        roles=roles,
        capabilities=capabilities_for(roles),
        study_ids=frozenset(study_ids),
    )


async def display_names(
    db: AsyncSession, ctx: TenantContext, subs: Collection[str]
) -> dict[str, str]:
    # Display names of these users in the caller's tenant; users without one are left out.
    if not subs:
        return {}
    rows = await db.execute(
        select(Membership.user_sub, Membership.display_name).where(
            Membership.tenant_id == ctx.tenant_id,
            Membership.user_sub.in_(subs),
            Membership.display_name.is_not(None),
        )
    )
    return {sub: name for sub, name in rows if name is not None}


async def get_context(db: AsyncSession, ctx: TenantContext) -> ContextRead:
    tenant_name = (
        await db.execute(select(Tenant.name).where(Tenant.id == ctx.tenant_id))
    ).scalar_one()
    rows = await db.execute(
        select(StudySnapshot, Study.title, Study.external_ref)
        .join(
            Study,
            and_(Study.tenant_id == StudySnapshot.tenant_id, Study.id == StudySnapshot.study_id),
        )
        .where(StudySnapshot.tenant_id == ctx.tenant_id, StudySnapshot.study_id.in_(ctx.study_ids))
        .ext(distinct_on(StudySnapshot.study_id))
        .order_by(StudySnapshot.study_id, StudySnapshot.version.desc())
    )
    studies = sorted(
        (_snapshot_read(snapshot, title, ref) for snapshot, title, ref in rows),
        key=lambda study: study.study_title,
    )
    return ContextRead(
        tenant_id=ctx.tenant_id,
        tenant_name=tenant_name,
        roles=sorted(ctx.roles),
        capabilities=sorted(ctx.capabilities),
        studies=studies,
        languages=list(Language),
        post_formats=[
            PostFormatOption(
                channel=spec.channel,
                post_format=post_format,
                aspect_ratio=spec.aspect_ratio,
                width_px=spec.width_px,
                height_px=spec.height_px,
            )
            for post_format, spec in FORMAT_SPECS.items()
        ],
        active_policy=await policies.get_active_policy_summary(db, ctx),
    )


def _snapshot_read(snapshot: StudySnapshot, title: str, ref: str | None) -> StudySnapshotRead:
    return StudySnapshotRead(
        study_id=snapshot.study_id,
        study_title=title,
        study_external_ref=ref,
        snapshot_id=snapshot.id,
        version=snapshot.version,
        recruitment_status=snapshot.recruitment_status,
        language=snapshot.language,
        facts=StudyFacts.model_validate(snapshot.facts),
        source_ref=snapshot.source_ref,
        approved_by=snapshot.approved_by,
        approved_at=snapshot.approved_at,
        is_synthetic=snapshot.is_synthetic,
    )
