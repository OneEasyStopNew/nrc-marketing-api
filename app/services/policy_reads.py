# Reads behind the policy admin routes (NRC-4): sources, bundles and their graph, checks and audit
# trail. The writes live in policy_sources and policy_bundles; this module only reads, apart from
# committing a source registration.

from collections.abc import Sequence
from datetime import datetime
from typing import Final
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import Capability, TenantContext
from app.core.errors import NotFoundError, PermissionDeniedError
from app.models import (
    PolicyAuditEvent,
    PolicyBundle,
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicyRuleWording,
    PolicySource,
)
from app.models.enums import BundleLifecycleState, PolicyLanguage, ReviewState
from app.schemas.policy_admin import (
    PolicyAuditPage,
    PolicyAuditRead,
    PolicyBundleChecks,
    PolicyBundleDetail,
    PolicyBundlePage,
    PolicyEdgeList,
    PolicyEdgeRead,
    PolicyIssueRead,
    PolicyNodeList,
    PolicyNodeRead,
    PolicyRuleList,
    PolicyRuleRead,
    PolicySourceCreate,
    PolicySourcePage,
    PolicySourceRead,
    PolicyWordingRead,
)
from app.services import policy_sources
from app.services.paging import newest_first_page, page_after
from app.services.policy_checks import load_snapshot, run_checks


async def register_source(
    db: AsyncSession, ctx: TenantContext, body: PolicySourceCreate
) -> tuple[PolicySource, bool]:
    source, created = await policy_sources.register_source(
        db,
        ctx,
        upload_id=body.upload_id,
        title=body.title,
        source_ref=body.source_ref,
        language=body.language,
        is_synthetic=body.is_synthetic,
    )
    await db.commit()
    # Server-side defaults (created_at) are loaded here, not lazily while the response is built.
    await db.refresh(source)
    return source, created


async def list_sources(
    db: AsyncSession, ctx: TenantContext, *, before: datetime | None, limit: int
) -> PolicySourcePage:
    page, next_cursor = await newest_first_page(
        db,
        select(PolicySource).where(PolicySource.tenant_id == ctx.tenant_id),
        created_at=PolicySource.created_at,
        row_id=PolicySource.id,
        sort_key=lambda row: (row[0].created_at, row[0].id),
        before=before,
        limit=limit,
    )
    return PolicySourcePage(
        items=[PolicySourceRead.model_validate(row[0]) for row in page], next_cursor=next_cursor
    )


async def get_bundle(db: AsyncSession, ctx: TenantContext, bundle_id: UUID) -> PolicyBundle:
    bundle = await db.scalar(
        select(PolicyBundle).where(
            PolicyBundle.tenant_id == ctx.tenant_id, PolicyBundle.id == bundle_id
        )
    )
    if bundle is None:
        raise NotFoundError("Policy bundle not found.")
    return bundle


def _detail(bundle: PolicyBundle, rule_count: int) -> PolicyBundleDetail:
    fields = PolicyBundleDetail.model_fields.keys() - {"rule_count"}
    return PolicyBundleDetail(
        rule_count=rule_count, **{name: getattr(bundle, name) for name in fields}
    )


async def _rule_counts(
    db: AsyncSession, ctx: TenantContext, bundle_ids: Sequence[UUID]
) -> dict[UUID, int]:
    rows = await db.execute(
        select(PolicyRuleRecord.policy_bundle_id, func.count())
        .where(
            PolicyRuleRecord.tenant_id == ctx.tenant_id,
            PolicyRuleRecord.policy_bundle_id.in_(bundle_ids),
        )
        .group_by(PolicyRuleRecord.policy_bundle_id)
    )
    return {bundle_id: count for bundle_id, count in rows}


async def bundle_detail(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID
) -> PolicyBundleDetail:
    bundle = await get_bundle(db, ctx, bundle_id)
    return _detail(bundle, (await _rule_counts(db, ctx, [bundle.id])).get(bundle.id, 0))


async def list_bundles(
    db: AsyncSession,
    ctx: TenantContext,
    *,
    state: BundleLifecycleState | None,
    before_version: int | None,
    limit: int,
) -> PolicyBundlePage:
    query = select(PolicyBundle).where(PolicyBundle.tenant_id == ctx.tenant_id)
    if state is not None:
        query = query.where(PolicyBundle.lifecycle_state == state)
    if before_version is not None:
        query = query.where(PolicyBundle.version < before_version)
    # One extra row tells whether an older page exists.
    rows = (await db.scalars(query.order_by(PolicyBundle.version.desc()).limit(limit + 1))).all()
    page = rows[:limit]
    counts = await _rule_counts(db, ctx, [bundle.id for bundle in page])
    return PolicyBundlePage(
        items=[_detail(bundle, counts.get(bundle.id, 0)) for bundle in page],
        next_cursor=page[-1].version if len(rows) > limit else None,
    )


_REVIEWS_PROPOSALS: Final = frozenset({Capability.POLICY_MANAGE, Capability.POLICY_APPROVE})


async def list_rules(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    *,
    include_proposals: bool,
    after: UUID | None = None,
    limit: int = 1,
    rule_record_id: UUID | None = None,
) -> PolicyRuleList:
    """Rules with their node and wordings. PROPOSED ones are for those who draft or review them.

    D13 kept proposals from creative readers; the approver reviews proposed wordings and
    exception rules (D-16), so policy:approve reads them too.
    """
    if include_proposals and not ctx.capabilities & _REVIEWS_PROPOSALS:
        raise PermissionDeniedError("Only policy managers and approvers can see proposed rules.")
    bundle = await get_bundle(db, ctx, bundle_id)
    rules_query = (
        select(PolicyRuleRecord, PolicyNode)
        .join(
            PolicyNode,
            and_(
                PolicyNode.tenant_id == PolicyRuleRecord.tenant_id,
                PolicyNode.policy_bundle_id == PolicyRuleRecord.policy_bundle_id,
                PolicyNode.id == PolicyRuleRecord.node_id,
            ),
        )
        .where(
            PolicyRuleRecord.tenant_id == ctx.tenant_id,
            PolicyRuleRecord.policy_bundle_id == bundle.id,
        )
    )
    if rule_record_id is not None:
        rules_query = rules_query.where(PolicyRuleRecord.id == rule_record_id)
    if not include_proposals:
        rules_query = rules_query.where(PolicyNode.review_state != ReviewState.PROPOSED)
    page, next_cursor = await page_after(
        db,
        rules_query,
        # rule_id is unique in a bundle (it is the rule node's key); id breaks no ties, it keeps
        # the order total for the cursor.
        order=(PolicyRuleRecord.rule_id, PolicyRuleRecord.id),
        row_id=PolicyRuleRecord.id,
        id_of=lambda row: row[0].id,
        after=after,
        limit=limit,
    )
    # Only the wordings of this page's rules, in one query.
    wordings_query = (
        select(PolicyRuleWording)
        .where(
            PolicyRuleWording.tenant_id == ctx.tenant_id,
            PolicyRuleWording.policy_bundle_id == bundle.id,
            PolicyRuleWording.policy_rule_id.in_([row[0].id for row in page]),
        )
        .order_by(PolicyRuleWording.created_at, PolicyRuleWording.id)
    )
    if not include_proposals:
        wordings_query = wordings_query.where(
            PolicyRuleWording.review_state != ReviewState.PROPOSED
        )
    wordings: dict[UUID, list[PolicyWordingRead]] = {}
    for wording in await db.scalars(wordings_query):
        wordings.setdefault(wording.policy_rule_id, []).append(
            PolicyWordingRead.model_validate(wording)
        )
    items = [
        PolicyRuleRead(
            id=rule.id,
            rule_id=rule.rule_id,
            rule_version=rule.rule_version,
            title=rule.title,
            requirement=rule.requirement,
            gate=rule.gate,
            severity=rule.severity,
            mandatory=rule.mandatory,
            check_type=rule.check_type,
            node=PolicyNodeRead.model_validate(node),
            wordings=wordings.get(rule.id, []),
        )
        for rule, node in page
    ]
    return PolicyRuleList(items=items, next_cursor=next_cursor)


async def get_rule(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, rule_record_id: UUID
) -> PolicyRuleRead:
    # The caller just wrote this rule, so it is a manager and proposals are visible.
    rules = await list_rules(
        db, ctx, bundle_id, include_proposals=True, rule_record_id=rule_record_id
    )
    if not rules.items:
        raise NotFoundError("Rule not found.")
    return rules.items[0]


async def list_nodes(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, *, after: UUID | None, limit: int
) -> PolicyNodeList:
    bundle = await get_bundle(db, ctx, bundle_id)
    page, next_cursor = await page_after(
        db,
        select(PolicyNode).where(
            PolicyNode.tenant_id == ctx.tenant_id, PolicyNode.policy_bundle_id == bundle.id
        ),
        order=(PolicyNode.node_key, PolicyNode.id),
        row_id=PolicyNode.id,
        id_of=lambda row: row[0].id,
        after=after,
        limit=limit,
    )
    return PolicyNodeList(
        items=[PolicyNodeRead.model_validate(row[0]) for row in page], next_cursor=next_cursor
    )


async def list_edges(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, *, after: UUID | None, limit: int
) -> PolicyEdgeList:
    bundle = await get_bundle(db, ctx, bundle_id)
    page, next_cursor = await page_after(
        db,
        select(PolicyEdge).where(
            PolicyEdge.tenant_id == ctx.tenant_id, PolicyEdge.policy_bundle_id == bundle.id
        ),
        order=(PolicyEdge.edge_type, PolicyEdge.from_node_id, PolicyEdge.to_node_id, PolicyEdge.id),
        row_id=PolicyEdge.id,
        id_of=lambda row: row[0].id,
        after=after,
        limit=limit,
    )
    return PolicyEdgeList(
        items=[PolicyEdgeRead.model_validate(row[0]) for row in page], next_cursor=next_cursor
    )


async def bundle_checks(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    supported_languages: Sequence[PolicyLanguage],
) -> PolicyBundleChecks:
    bundle = await get_bundle(db, ctx, bundle_id)
    snapshot = await load_snapshot(db, ctx.tenant_id, bundle.id)
    return PolicyBundleChecks(
        issues=[
            PolicyIssueRead(code=issue.code, subject=issue.subject, message=issue.message)
            for issue in run_checks(snapshot, supported_languages)
        ]
    )


async def list_audit(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, *, before: datetime | None, limit: int
) -> PolicyAuditPage:
    bundle = await get_bundle(db, ctx, bundle_id)
    page, next_cursor = await newest_first_page(
        db,
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.tenant_id == ctx.tenant_id,
            PolicyAuditEvent.policy_bundle_id == bundle.id,
        ),
        created_at=PolicyAuditEvent.created_at,
        row_id=PolicyAuditEvent.id,
        sort_key=lambda row: (row[0].created_at, row[0].id),
        before=before,
        limit=limit,
    )
    return PolicyAuditPage(
        items=[PolicyAuditRead.model_validate(row[0]) for row in page], next_cursor=next_cursor
    )
