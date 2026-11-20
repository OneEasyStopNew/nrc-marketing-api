"""Policy evidence reads (NRC-5): a bundle's rules with their sources, and a run's stored context.

Both are read-only and run under the caller's tenant (RLS), so another tenant's bundle or run is
simply not found. Only approved nodes and edges are shown, as in retrieval.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.errors import ContextNotAvailableError, NotFoundError, PolicyContextUnavailableError
from app.models import (
    ContextManifestRecord,
    Creative,
    PolicyBundle,
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicySource,
    PolicySourceClause,
    ValidationRun,
)
from app.models.enums import PolicyEdgeType, PolicyNodeType
from app.schemas.policy_evidence import (
    BundleRuleRead,
    ContextEvidenceRead,
    ContextExceptionRead,
    ContextRuleRead,
    ContextStatementRead,
    MissingContextRead,
    PolicyBundleRulesRead,
    RuleApplicabilityRead,
    RuleExceptionRead,
    RuleSourceRead,
    RuleTextRead,
    SupersededRuleRead,
    ValidationContextRead,
)
from app.services.creatives import visible_to
from app.services.policy_context.manifest import EXCERPT_MAX_CHARS, ContextManifest
from app.services.policy_context.provider import USABLE_BUNDLE_STATES

# The edges a rule's detail follows, one hop from the rule.
_DETAIL_EDGES = (
    PolicyEdgeType.SUPPORTED_BY,
    PolicyEdgeType.REQUIRES_DISCLOSURE,
    PolicyEdgeType.EXCEPT_WHEN,
)


async def get_bundle_rules(
    db: AsyncSession, ctx: TenantContext, policy_bundle_id: UUID
) -> PolicyBundleRulesRead:
    """Return a bundle's approved rules with their sources, disclosures and exceptions.

    Only ACTIVE and RETIRED bundles (the ones runs can pin); drafts belong to policy review.

    Raises:
        NotFoundError: no such bundle for the tenant, or it is not active or retired.
        PolicyContextUnavailableError: the bundle has no approved rules in its graph (an older
            bundle without one); details.reason is GRAPH_MISSING.
    """
    bundle = (
        await db.execute(
            select(
                PolicyBundle.id,
                PolicyBundle.version,
                PolicyBundle.name,
                PolicyBundle.lifecycle_state,
                PolicyBundle.is_synthetic,
            ).where(PolicyBundle.tenant_id == ctx.tenant_id, PolicyBundle.id == policy_bundle_id)
        )
    ).one_or_none()
    if bundle is None or bundle.lifecycle_state not in USABLE_BUNDLE_STATES:
        raise NotFoundError("Policy bundle not found.")
    rules = (
        await db.execute(
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
                PolicyNode.approved.is_(True),
            )
            .order_by(PolicyRuleRecord.rule_id)
        )
    ).all()
    if not rules:
        raise PolicyContextUnavailableError(
            "This policy bundle has no rules to show.", details={"reason": "GRAPH_MISSING"}
        )
    linked = await _linked_nodes(db, ctx.tenant_id, bundle.id, [node.id for _, node in rules])
    # Sources are the clauses quoted by rule nodes and by the nodes rules are supported by.
    quoting = [node for _, node in rules] + [
        node for edge, node in _flatten(linked) if edge is PolicyEdgeType.SUPPORTED_BY
    ]
    clauses = await _clauses(
        db, ctx.tenant_id, {n.source_clause_id for n in quoting if n.source_clause_id is not None}
    )
    return PolicyBundleRulesRead(
        policy_bundle_id=bundle.id,
        version=bundle.version,
        name=bundle.name,
        lifecycle_state=bundle.lifecycle_state,
        is_synthetic=bundle.is_synthetic,
        rules=[_rule(rule, node, linked.get(node.id, []), clauses) for rule, node in rules],
    )


async def get_validation_context(
    db: AsyncSession, ctx: TenantContext, validation_run_id: UUID
) -> ValidationContextRead:
    """Return the policy context stored for a run.

    Raises:
        NotFoundError: no such run visible to the caller.
        ContextNotAvailableError: the run stored no context (an older run, or retrieval was
            unavailable and the run recorded a coverage gap).
    """
    row = (
        await db.execute(
            select(ValidationRun.id, ValidationRun.context_manifest_id)
            .join(
                Creative,
                and_(
                    Creative.tenant_id == ValidationRun.tenant_id,
                    Creative.id == ValidationRun.creative_id,
                ),
            )
            .where(*visible_to(ctx), ValidationRun.id == validation_run_id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Validation run not found.")
    if row.context_manifest_id is None:
        raise ContextNotAvailableError()
    stored = await db.scalar(
        select(ContextManifestRecord.manifest).where(
            ContextManifestRecord.tenant_id == ctx.tenant_id,
            ContextManifestRecord.id == row.context_manifest_id,
        )
    )
    if stored is None:
        # The foreign key makes this impossible; never answer with someone else's context.
        raise ContextNotAvailableError()
    return context_read(row.id, ContextManifest.model_validate(stored))


def context_read(validation_run_id: UUID, manifest: ContextManifest) -> ValidationContextRead:
    # The stored manifest without statement hashes; field names match, so submodels map directly.
    scope = manifest.scope
    return ValidationContextRead(
        validation_run_id=validation_run_id,
        policy_bundle_id=manifest.policy_bundle.id,
        policy_bundle_version=manifest.policy_bundle.version,
        is_synthetic=manifest.policy_bundle.is_synthetic,
        study_snapshot_id=manifest.study_snapshot.id,
        study_snapshot_version=manifest.study_snapshot.version,
        input_kind=scope.input_kind,
        channel=scope.channel,
        post_format=scope.post_format,
        language=scope.language,
        audience=scope.audience,
        jurisdiction=scope.jurisdiction,
        evaluated_at=scope.evaluated_at,
        retrieval_mode=manifest.retrieval_mode,
        embedding_model_id=manifest.embedding_model_id,
        statements=[ContextStatementRead(span=ref.span) for ref in manifest.statements],
        required_rules=[ContextRuleRead.model_validate(r) for r in manifest.required_rules],
        optional_rules=[ContextRuleRead.model_validate(r) for r in manifest.optional_rules],
        evidence=[ContextEvidenceRead.model_validate(e) for e in manifest.evidence],
        exceptions=[ContextExceptionRead.model_validate(e) for e in manifest.exceptions],
        superseded=[SupersededRuleRead.model_validate(s) for s in manifest.superseded],
        missing_context=[MissingContextRead.model_validate(m) for m in manifest.missing_context],
    )


_Linked = Mapping[UUID, Sequence[tuple[PolicyEdgeType, PolicyNode]]]


def _flatten(linked: _Linked) -> list[tuple[PolicyEdgeType, PolicyNode]]:
    return [pair for pairs in linked.values() for pair in pairs]


async def _linked_nodes(
    db: AsyncSession, tenant_id: UUID, bundle_id: UUID, rule_node_ids: Sequence[UUID]
) -> _Linked:
    # Every approved edge between a rule and an approved node, keyed by the rule's node.
    rows = (
        await db.execute(
            select(PolicyEdge.from_node_id, PolicyEdge.edge_type, PolicyNode)
            .join(
                PolicyNode,
                and_(
                    PolicyNode.tenant_id == PolicyEdge.tenant_id,
                    PolicyNode.policy_bundle_id == PolicyEdge.policy_bundle_id,
                    PolicyNode.id == PolicyEdge.to_node_id,
                ),
            )
            .where(
                PolicyEdge.tenant_id == tenant_id,
                PolicyEdge.policy_bundle_id == bundle_id,
                PolicyEdge.approved.is_(True),
                PolicyEdge.edge_type.in_(_DETAIL_EDGES),
                PolicyEdge.from_node_id.in_(rule_node_ids),
                PolicyNode.approved.is_(True),
            )
            .order_by(PolicyNode.node_key)
        )
    ).all()
    # Interpretations point at the rule they interpret (applies_to): followed backwards.
    applied = (
        await db.execute(
            select(PolicyEdge.to_node_id, PolicyEdge.edge_type, PolicyNode)
            .join(
                PolicyNode,
                and_(
                    PolicyNode.tenant_id == PolicyEdge.tenant_id,
                    PolicyNode.policy_bundle_id == PolicyEdge.policy_bundle_id,
                    PolicyNode.id == PolicyEdge.from_node_id,
                ),
            )
            .where(
                PolicyEdge.tenant_id == tenant_id,
                PolicyEdge.policy_bundle_id == bundle_id,
                PolicyEdge.approved.is_(True),
                PolicyEdge.edge_type == PolicyEdgeType.APPLIES_TO,
                PolicyEdge.to_node_id.in_(rule_node_ids),
                PolicyNode.approved.is_(True),
            )
            .order_by(PolicyNode.node_key)
        )
    ).all()
    linked: defaultdict[UUID, list[tuple[PolicyEdgeType, PolicyNode]]] = defaultdict(list)
    for rule_node_id, edge_type, node in [*rows, *applied]:
        linked[rule_node_id].append((edge_type, node))
    return linked


async def _clauses(
    db: AsyncSession, tenant_id: UUID, clause_ids: set[UUID]
) -> Mapping[UUID, tuple[PolicySourceClause, str]]:
    # Each clause with its source's title.
    if not clause_ids:
        return {}
    rows = (
        await db.execute(
            select(PolicySourceClause, PolicySource.title)
            .join(
                PolicySource,
                and_(
                    PolicySource.tenant_id == PolicySourceClause.tenant_id,
                    PolicySource.id == PolicySourceClause.source_id,
                ),
            )
            .where(PolicySourceClause.tenant_id == tenant_id, PolicySourceClause.id.in_(clause_ids))
        )
    ).all()
    return {clause.id: (clause, title) for clause, title in rows}


def _rule(
    rule: PolicyRuleRecord,
    node: PolicyNode,
    linked: Sequence[tuple[PolicyEdgeType, PolicyNode]],
    clauses: Mapping[UUID, tuple[PolicySourceClause, str]],
) -> BundleRuleRead:
    # Sources: the clause the rule node quotes, then those of the nodes it is supported by.
    supporting = [n for edge, n in linked if edge is PolicyEdgeType.SUPPORTED_BY]
    clause_ids = dict.fromkeys(
        n.source_clause_id for n in [node, *supporting] if n.source_clause_id is not None
    )
    # Interpretations support a rule or point at it (applies_to); one entry per node either way.
    notes = {
        n.node_key: n
        for edge, n in linked
        if edge in (PolicyEdgeType.SUPPORTED_BY, PolicyEdgeType.APPLIES_TO)
    }
    return BundleRuleRead(
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        title=rule.title,
        requirement=rule.requirement,
        gate=rule.gate,
        severity=rule.severity,
        mandatory=rule.mandatory,
        check_type=rule.check_type,
        exact_wording=rule.exact_wording,
        prohibited_phrases=list(rule.prohibited_phrases),
        applicability=_applicability(node),
        sources=[_source(*clauses[clause_id]) for clause_id in clause_ids if clause_id in clauses],
        disclosures=[_text(n) for edge, n in linked if edge is PolicyEdgeType.REQUIRES_DISCLOSURE],
        interpretations=[
            _text(n) for n in notes.values() if n.node_type is PolicyNodeType.INTERPRETATION
        ],
        examples=[_text(n) for n in notes.values() if n.node_type is PolicyNodeType.EXAMPLE],
        exceptions=[
            RuleExceptionRead(
                node_key=n.node_key,
                language=n.language,
                text=n.text,
                requires_study_fact=n.required_study_fact,
                applicability=_applicability(n),
            )
            for edge, n in linked
            if edge is PolicyEdgeType.EXCEPT_WHEN and n.node_type is PolicyNodeType.EXCEPTION
        ],
    )


def _applicability(node: PolicyNode) -> RuleApplicabilityRead:
    return RuleApplicabilityRead(
        input_kinds=list(node.input_kinds),
        channels=list(node.channels),
        post_formats=list(node.post_formats),
        languages=list(node.languages),
        audiences=list(node.audiences),
        jurisdictions=list(node.jurisdictions),
        study_specific=bool(node.study_ids),
        effective_from=node.effective_from,
        effective_to=node.effective_to,
    )


def _text(node: PolicyNode) -> RuleTextRead:
    return RuleTextRead(node_key=node.node_key, language=node.language, text=node.text)


def _source(clause: PolicySourceClause, source_title: str) -> RuleSourceRead:
    return RuleSourceRead(
        source_title=source_title,
        clause_ref=clause.clause_ref,
        section=clause.section,
        page=clause.page,
        language=clause.language,
        excerpt=clause.text[:EXCERPT_MAX_CHARS],
    )
