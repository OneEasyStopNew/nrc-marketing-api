"""Policy bundle lifecycle (NRC-4): drafts, graph edits, item review, submit, approve, reject.

Every change locks the bundle row, checks the caller's capability, the If-Match revision and the
lifecycle state, applies the change, bumps the revision and writes an audit row, all in one
transaction. The graph can change only while the bundle is DRAFT (edits) or REVIEW (review
decisions); a database trigger refuses anything else as a backstop. Approval runs the bundle
checks and writes the JSONB rules that validation reads today. Building, activation and
retirement extend TRANSITIONS in later steps.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import class_mapper

from app.auth.access import Capability, TenantContext
from app.core.errors import (
    NotFoundError,
    PermissionDeniedError,
    PolicyBundleHasIssuesError,
    PolicyBundleStateError,
    PolicyItemInvalidError,
    VersionConflictError,
)
from app.core.request_context import request_id_var
from app.models import (
    PolicyAuditEvent,
    PolicyBundle,
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicyRuleWording,
    PolicySource,
    PolicySourceClause,
)
from app.models.enums import (
    BundleLifecycleState,
    PolicyAuditAction,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ReviewState,
    ValidationInputKind,
    WordingKind,
)
from app.schemas.policies import MAX_RULES_PER_BUNDLE, PolicyRule
from app.schemas.policy_admin import (
    BundleCreate,
    EdgeCreate,
    NodeCreate,
    NodeUpdate,
    ReviewDecision,
    RuleCreate,
    RuleUpdate,
    TransitionRequest,
    WordingCreate,
)
from app.schemas.validations import SourceReference
from app.services.policy_checks import load_snapshot, run_checks

State = BundleLifecycleState
EDITABLE: Final = frozenset({State.DRAFT})
REVIEWABLE: Final = frozenset({State.DRAFT, State.REVIEW})


class Command(StrEnum):
    SUBMIT = "submit"
    RETURN_TO_DRAFT = "return_to_draft"
    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class Transition:
    from_states: frozenset[BundleLifecycleState]
    to_state: BundleLifecycleState
    # Any one of these lets the caller run the command.
    capabilities: frozenset[Capability]
    action: PolicyAuditAction
    needs_reason: bool = False


TRANSITIONS: Final[Mapping[Command, Transition]] = {
    Command.SUBMIT: Transition(
        frozenset({State.DRAFT}),
        State.REVIEW,
        frozenset({Capability.POLICY_MANAGE}),
        PolicyAuditAction.SUBMIT,
    ),
    # Editors take a bundle back to fix it; approvers send it back with their reason.
    Command.RETURN_TO_DRAFT: Transition(
        frozenset({State.REVIEW}),
        State.DRAFT,
        frozenset({Capability.POLICY_MANAGE, Capability.POLICY_APPROVE}),
        PolicyAuditAction.RETURN_TO_DRAFT,
    ),
    Command.APPROVE: Transition(
        frozenset({State.REVIEW}),
        State.APPROVED,
        frozenset({Capability.POLICY_APPROVE}),
        PolicyAuditAction.APPROVE,
    ),
    Command.REJECT: Transition(
        frozenset({State.REVIEW}),
        State.REJECTED,
        frozenset({Capability.POLICY_APPROVE}),
        PolicyAuditAction.REJECT,
        needs_reason=True,
    ),
}


class ReviewItemKind(StrEnum):
    NODE = "node"
    EDGE = "edge"
    WORDING = "wording"


_REVIEW_MODELS: Final[
    Mapping[ReviewItemKind, type[PolicyNode | PolicyEdge | PolicyRuleWording]]
] = {
    ReviewItemKind.NODE: PolicyNode,
    ReviewItemKind.EDGE: PolicyEdge,
    ReviewItemKind.WORDING: PolicyRuleWording,
}
# Interpretations stand in for a source clause; their reference names the node instead.
INTERPRETATION_SOURCE: Final = "interpretation"


async def lock_tenant_bundles(db: AsyncSession, tenant_id: UUID) -> None:
    # Serialises version numbering (and, later, activation) per tenant until the transaction ends.
    await db.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(f"policy_bundles:{tenant_id}", 0)))
    )


def _require(ctx: TenantContext, capabilities: frozenset[Capability]) -> None:
    if not capabilities & ctx.capabilities:
        raise PermissionDeniedError("Your role cannot do this to a policy bundle.")


async def _open(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    expected_revision: int,
    states: frozenset[BundleLifecycleState],
) -> PolicyBundle:
    # The row lock holds until commit: concurrent changes to the same bundle run one at a time,
    # and the graph trigger (FOR SHARE on this row) waits for us too.
    bundle = await db.scalar(
        select(PolicyBundle)
        .where(PolicyBundle.tenant_id == ctx.tenant_id, PolicyBundle.id == bundle_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if bundle is None:
        raise NotFoundError("Policy bundle not found.")
    if bundle.revision != expected_revision:
        raise VersionConflictError(
            "The policy bundle changed since you loaded it. Reload and reapply your change.",
            details={"current_version": bundle.revision},
        )
    if bundle.lifecycle_state not in states:
        raise PolicyBundleStateError(
            details={
                "state": bundle.lifecycle_state.value,
                "allowed": sorted(state.value for state in states),
            }
        )
    return bundle


async def _finish(
    db: AsyncSession,
    ctx: TenantContext,
    bundle: PolicyBundle,
    action: PolicyAuditAction,
    *,
    from_state: BundleLifecycleState | None = None,
    reason: str | None = None,
    details: Mapping[str, object] | None = None,
    item: PolicyNode | PolicyEdge | PolicyRuleRecord | PolicyRuleWording | None = None,
) -> None:
    bundle.revision += 1
    bundle.updated_at = func.now()
    db.add(
        PolicyAuditEvent(
            tenant_id=bundle.tenant_id,
            policy_bundle_id=bundle.id,
            action=action,
            from_state=from_state,
            to_state=bundle.lifecycle_state if from_state is not None else None,
            actor=ctx.principal.sub,
            reason=reason,
            details={"revision": bundle.revision, **(details or {})},
            request_id=request_id_var.get(),
        )
    )
    await db.commit()
    # Load server-side values (timestamps, defaults) for the caller's response.
    await db.refresh(bundle)
    if item is not None:
        await db.refresh(item)


async def create_bundle(db: AsyncSession, ctx: TenantContext, body: BundleCreate) -> PolicyBundle:
    """A new DRAFT with the tenant's next version, empty or copied from based_on_bundle_id."""
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    await lock_tenant_bundles(db, ctx.tenant_id)
    base = None
    if body.based_on_bundle_id is not None:
        base = await db.scalar(
            select(PolicyBundle).where(
                PolicyBundle.tenant_id == ctx.tenant_id,
                PolicyBundle.id == body.based_on_bundle_id,
            )
        )
        if base is None:
            raise NotFoundError("Policy bundle to copy not found.")
    latest = await db.scalar(
        select(func.max(PolicyBundle.version)).where(PolicyBundle.tenant_id == ctx.tenant_id)
    )
    bundle = PolicyBundle(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        version=(latest or 0) + 1,
        name=body.name,
        source_ref=body.source_ref,
        rules=[],
        is_synthetic=body.is_synthetic,
        lifecycle_state=State.DRAFT,
        based_on_bundle_id=body.based_on_bundle_id,
    )
    db.add(bundle)
    await db.flush()
    copied = await _copy_graph(db, ctx.tenant_id, base.id, bundle.id) if base else {}
    db.add(
        PolicyAuditEvent(
            tenant_id=ctx.tenant_id,
            policy_bundle_id=bundle.id,
            action=PolicyAuditAction.CREATE,
            from_state=None,
            to_state=State.DRAFT,
            actor=ctx.principal.sub,
            details={
                "revision": 1,
                "based_on_bundle_id": str(base.id) if base else None,
                **copied,
            },
            request_id=request_id_var.get(),
        )
    )
    await db.commit()
    await db.refresh(bundle)
    return bundle


_GraphModel = type[PolicyNode | PolicyEdge | PolicyRuleRecord | PolicyRuleWording]


def _copy_columns(model: _GraphModel) -> list[str]:
    # Every stored column except identity and generated ones; review fields are kept, so a copy
    # carries who approved each row and why (provenance survives a re-import, AC3).
    return [
        column.name
        for column in class_mapper(model).columns
        if column.name not in ("id", "policy_bundle_id") and column.computed is None
    ]


async def _copy_graph(
    db: AsyncSession, tenant_id: UUID, source_bundle: UUID, target_bundle: UUID
) -> dict[str, int]:
    ids: dict[UUID, UUID] = {}

    async def rows(model: _GraphModel) -> list[dict[str, object]]:
        columns = _copy_columns(model)
        result = await db.execute(
            select(model.id, *(getattr(model, name) for name in columns)).where(
                model.tenant_id == tenant_id, model.policy_bundle_id == source_bundle
            )
        )
        copies = []
        for row in result.mappings():
            new_id = uuid4()
            ids[row["id"]] = new_id
            copies.append({**{name: row[name] for name in columns}, "id": new_id})
        return copies

    nodes = await rows(PolicyNode)
    rules = await rows(PolicyRuleRecord)
    edges = await rows(PolicyEdge)
    wordings = await rows(PolicyRuleWording)
    remap = {
        "node_id",
        "from_node_id",
        "to_node_id",
        "policy_rule_id",
        "translation_of_id",
    }
    for model, copies in (
        (PolicyNode, nodes),
        (PolicyRuleRecord, rules),
        (PolicyEdge, edges),
        (PolicyRuleWording, wordings),
    ):
        for copy in copies:
            copy["policy_bundle_id"] = target_bundle
            for name in remap & copy.keys():
                old = copy[name]
                if isinstance(old, UUID):
                    copy[name] = ids[old]
        if copies:
            # One statement per table: a wording's translation_of points into the same batch, and
            # foreign keys are checked at the end of the statement.
            await db.execute(insert(model), copies)
    return {
        "nodes": len(nodes),
        "rules": len(rules),
        "edges": len(edges),
        "wordings": len(wordings),
    }


async def _require_clause(db: AsyncSession, tenant_id: UUID, clause_id: UUID | None) -> None:
    if clause_id is None:
        return
    found = await db.scalar(
        select(PolicySourceClause.id).where(
            PolicySourceClause.tenant_id == tenant_id, PolicySourceClause.id == clause_id
        )
    )
    if found is None:
        raise NotFoundError("Source clause not found.")


async def _in_bundle[M: PolicyNode | PolicyEdge | PolicyRuleRecord | PolicyRuleWording](
    db: AsyncSession, model: type[M], bundle: PolicyBundle, item_id: UUID, what: str
) -> M:
    item = await db.scalar(
        select(model).where(
            model.tenant_id == bundle.tenant_id,
            model.policy_bundle_id == bundle.id,
            model.id == item_id,
        )
    )
    if item is None:
        raise NotFoundError(f"{what} not found in this policy bundle.")
    return item


async def _node_key_free(db: AsyncSession, bundle: PolicyBundle, node_key: str) -> None:
    taken = await db.scalar(
        select(PolicyNode.id).where(
            PolicyNode.tenant_id == bundle.tenant_id,
            PolicyNode.policy_bundle_id == bundle.id,
            PolicyNode.node_key == node_key,
        )
    )
    if taken is not None:
        raise PolicyItemInvalidError(
            "Another node in this bundle already uses this key.", details={"node_key": node_key}
        )


def _check_window(values: Mapping[str, object]) -> None:
    start, end = values.get("effective_from"), values.get("effective_to")
    if isinstance(start, datetime) and isinstance(end, datetime) and start > end:
        raise PolicyItemInvalidError("effective_from must not be after effective_to.")


_NULLABLE_NODE_FIELDS: Final = frozenset(
    {"source_clause_id", "effective_from", "effective_to", "required_study_fact"}
)


def _changes(body: NodeUpdate | RuleUpdate, *, nullable: frozenset[str]) -> dict[str, object]:
    # Fields the caller sent; null only clears a field that may be empty.
    changes = body.model_dump(exclude_unset=True)
    cleared = sorted(
        name for name, value in changes.items() if value is None and name not in nullable
    )
    if cleared:
        raise PolicyItemInvalidError("These fields cannot be empty.", details={"fields": cleared})
    return changes


def _back_to_proposed(item: PolicyNode | PolicyEdge | PolicyRuleWording) -> None:
    # A changed item has to be reviewed again before it can count.
    item.review_state = ReviewState.PROPOSED
    item.reviewed_by = None
    item.reviewed_at = None
    item.review_note = None
    if not isinstance(item, PolicyRuleWording):
        item.approved = False


async def add_node(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, revision: int, body: NodeCreate
) -> PolicyNode:
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    await _node_key_free(db, bundle, body.node_key)
    await _require_clause(db, ctx.tenant_id, body.source_clause_id)
    values = body.model_dump()
    _check_window(values)
    node = PolicyNode(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        policy_bundle_id=bundle.id,
        review_state=ReviewState.PROPOSED,
        approved=False,
        **values,
    )
    db.add(node)
    await _finish(
        db, ctx, bundle, PolicyAuditAction.EDIT, details={"added_node_id": str(node.id)}, item=node
    )
    return node


async def update_node(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    node_id: UUID,
    body: NodeUpdate,
) -> PolicyNode:
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    node = await _in_bundle(db, PolicyNode, bundle, node_id, "Node")
    changes = _changes(body, nullable=_NULLABLE_NODE_FIELDS)
    await _require_clause(db, ctx.tenant_id, body.source_clause_id)
    _check_window(
        {
            "effective_from": changes.get("effective_from", node.effective_from),
            "effective_to": changes.get("effective_to", node.effective_to),
        }
    )
    for name, value in changes.items():
        setattr(node, name, value)
    _back_to_proposed(node)
    await _finish(
        db,
        ctx,
        bundle,
        PolicyAuditAction.EDIT,
        details={"changed_node_id": str(node.id)},
        item=node,
    )
    return node


async def add_rule(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, revision: int, body: RuleCreate
) -> PolicyRuleRecord:
    """A rule and its RULE node (node_key = rule_id), both PROPOSED."""
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    await _node_key_free(db, bundle, body.rule_id)
    rule_fields = {"rule_id", "rule_version", "title", "requirement", "gate", "severity"}
    rule_fields |= {"mandatory", "check_type"}
    node_values = body.model_dump(exclude=rule_fields)
    _check_window(node_values)
    node = PolicyNode(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        policy_bundle_id=bundle.id,
        node_key=body.rule_id,
        node_type=PolicyNodeType.RULE,
        review_state=ReviewState.PROPOSED,
        approved=False,
        **node_values,
    )
    rule = PolicyRuleRecord(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        policy_bundle_id=bundle.id,
        node_id=node.id,
        prohibited_phrases=[],
        **body.model_dump(include=rule_fields),
    )
    db.add(node)
    await db.flush()
    db.add(rule)
    await _finish(
        db, ctx, bundle, PolicyAuditAction.EDIT, details={"added_rule_id": str(rule.id)}, item=rule
    )
    return rule


async def update_rule(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    rule_record_id: UUID,
    body: RuleUpdate,
) -> PolicyRuleRecord:
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    rule = await _in_bundle(db, PolicyRuleRecord, bundle, rule_record_id, "Rule")
    node = await _in_bundle(db, PolicyNode, bundle, rule.node_id, "Rule node")
    for name, value in _changes(body, nullable=frozenset()).items():
        setattr(rule, name, value)
    _back_to_proposed(node)
    await _finish(
        db,
        ctx,
        bundle,
        PolicyAuditAction.EDIT,
        details={"changed_rule_id": str(rule.id)},
        item=rule,
    )
    return rule


async def add_edge(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, revision: int, body: EdgeCreate
) -> PolicyEdge:
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    if body.from_node_id == body.to_node_id:
        raise PolicyItemInvalidError("An edge must join two different nodes.")
    await _in_bundle(db, PolicyNode, bundle, body.from_node_id, "Node")
    await _in_bundle(db, PolicyNode, bundle, body.to_node_id, "Node")
    await _require_clause(db, ctx.tenant_id, body.provenance_clause_id)
    duplicate = await db.scalar(
        select(PolicyEdge.id).where(
            PolicyEdge.tenant_id == bundle.tenant_id,
            PolicyEdge.policy_bundle_id == bundle.id,
            PolicyEdge.edge_type == body.edge_type,
            PolicyEdge.from_node_id == body.from_node_id,
            PolicyEdge.to_node_id == body.to_node_id,
        )
    )
    if duplicate is not None:
        raise PolicyItemInvalidError("This edge already exists in the bundle.")
    edge = PolicyEdge(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        policy_bundle_id=bundle.id,
        review_state=ReviewState.PROPOSED,
        approved=False,
        **body.model_dump(),
    )
    db.add(edge)
    await _finish(
        db, ctx, bundle, PolicyAuditAction.EDIT, details={"added_edge_id": str(edge.id)}, item=edge
    )
    return edge


async def add_wording(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    rule_record_id: UUID,
    body: WordingCreate,
) -> PolicyRuleWording:
    """Required wording or a prohibited phrase for a rule. Kept exactly as given (AC7)."""
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    await _in_bundle(db, PolicyRuleRecord, bundle, rule_record_id, "Rule")
    await _require_clause(db, ctx.tenant_id, body.source_clause_id)
    if body.translation_of_id is not None:
        await _in_bundle(db, PolicyRuleWording, bundle, body.translation_of_id, "Wording")
    live = (
        await db.scalars(
            select(PolicyRuleWording).where(
                PolicyRuleWording.tenant_id == bundle.tenant_id,
                PolicyRuleWording.policy_rule_id == rule_record_id,
                PolicyRuleWording.language == body.language,
                PolicyRuleWording.kind == body.kind,
            )
        )
    ).all()
    if any(existing.text == body.text for existing in live):
        raise PolicyItemInvalidError("The rule already has this wording.")
    if body.kind is WordingKind.EXACT_WORDING and any(
        existing.review_state is not ReviewState.REJECTED for existing in live
    ):
        raise PolicyItemInvalidError(
            "The rule already has required wording in this language; reject it first."
        )
    wording = PolicyRuleWording(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        policy_bundle_id=bundle.id,
        policy_rule_id=rule_record_id,
        review_state=ReviewState.PROPOSED,
        **body.model_dump(),
    )
    db.add(wording)
    await _finish(
        db,
        ctx,
        bundle,
        PolicyAuditAction.EDIT,
        details={"added_wording_id": str(wording.id)},
        item=wording,
    )
    return wording


async def review_item(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    kind: ReviewItemKind,
    item_id: UUID,
    body: ReviewDecision,
) -> None:
    """Approve or reject one node, edge or wording while the bundle is a draft or in review."""
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, REVIEWABLE)
    item = await _in_bundle(db, _REVIEW_MODELS[kind], bundle, item_id, kind.value.capitalize())
    decision = ReviewState(body.decision)
    if (
        decision is ReviewState.APPROVED
        and isinstance(item, PolicyEdge)
        and item.provenance_clause_id is None
        and not (item.provenance_note or "").strip()
    ):
        # AC1: an approved edge cites a source clause or states its interpretation.
        raise PolicyItemInvalidError("An edge needs a source clause or a note before approval.")
    item.review_state = decision
    item.reviewed_by = ctx.principal.sub
    item.reviewed_at = func.now()
    item.review_note = body.note
    if not isinstance(item, PolicyRuleWording):
        item.approved = decision is ReviewState.APPROVED
    await _finish(
        db,
        ctx,
        bundle,
        PolicyAuditAction.REVIEW,
        details={"item_kind": kind.value, "item_id": str(item_id), "decision": decision.value},
    )


async def transition(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    command: Command,
    body: TransitionRequest,
    supported_languages: Sequence[PolicyLanguage],
) -> PolicyBundle:
    """Move the bundle along TRANSITIONS. Approval needs a bundle with no open issues (AC2)."""
    step = TRANSITIONS[command]
    _require(ctx, step.capabilities)
    if step.needs_reason and body.reason is None:
        raise PolicyItemInvalidError("Give a reason.", details={"field": "reason"})
    bundle = await _open(db, ctx, bundle_id, revision, step.from_states)
    from_state = bundle.lifecycle_state
    details: dict[str, object] = {}
    if command is Command.APPROVE:
        details = await _approve(db, ctx, bundle, supported_languages)
    bundle.lifecycle_state = step.to_state
    await _finish(
        db,
        ctx,
        bundle,
        step.action,
        from_state=from_state,
        reason=body.reason,
        details=details,
    )
    return bundle


async def _approve(
    db: AsyncSession,
    ctx: TenantContext,
    bundle: PolicyBundle,
    supported_languages: Sequence[PolicyLanguage],
) -> dict[str, object]:
    # Runs while the bundle is still REVIEW and locked, so the graph cannot change underneath.
    snapshot = await load_snapshot(db, bundle.tenant_id, bundle.id)
    issues = run_checks(snapshot, supported_languages)
    if issues:
        raise PolicyBundleHasIssuesError(
            details={
                "issues": [
                    {"code": i.code.value, "subject": i.subject, "message": i.message}
                    for i in issues
                ]
            }
        )
    projection = await _project_rules(db, bundle)
    # Write the rules' wording columns now, while the bundle is still REVIEW: once it is
    # APPROVED the graph trigger refuses them.
    await db.flush()
    if len(projection) > MAX_RULES_PER_BUNDLE:
        raise PolicyItemInvalidError(
            f"A bundle may hold at most {MAX_RULES_PER_BUNDLE} rules.",
            details={"rules": len(projection)},
        )
    bundle.rules = [rule.model_dump(mode="json") for rule in projection]
    bundle.approved_by = ctx.principal.sub
    bundle.approved_at = func.now()
    return {"rules": len(projection)}


async def _project_rules(db: AsyncSession, bundle: PolicyBundle) -> list[PolicyRule]:
    # The JSONB rules validation reads (and the rule's English wording columns NRC-5 reads), from
    # the approved graph only. Writes to policy_rules happen while the bundle is still REVIEW.
    approved_nodes = {
        node.id: node
        for node in (
            await db.scalars(
                select(PolicyNode).where(
                    PolicyNode.tenant_id == bundle.tenant_id,
                    PolicyNode.policy_bundle_id == bundle.id,
                    PolicyNode.review_state == ReviewState.APPROVED,
                )
            )
        ).all()
    }
    support: dict[UUID, list[PolicyNode]] = {}
    for edge in (
        await db.scalars(
            select(PolicyEdge).where(
                PolicyEdge.tenant_id == bundle.tenant_id,
                PolicyEdge.policy_bundle_id == bundle.id,
                PolicyEdge.edge_type == PolicyEdgeType.SUPPORTED_BY,
                PolicyEdge.review_state == ReviewState.APPROVED,
            )
        )
    ).all():
        target = approved_nodes.get(edge.to_node_id)
        if target is not None:
            support.setdefault(edge.from_node_id, []).append(target)
    clause_refs = {
        clause_id: (source_ref, clause_ref)
        for clause_id, source_ref, clause_ref in (
            await db.execute(
                select(
                    PolicySourceClause.id, PolicySource.source_ref, PolicySourceClause.clause_ref
                )
                .join(
                    PolicySource,
                    (PolicySource.tenant_id == PolicySourceClause.tenant_id)
                    & (PolicySource.id == PolicySourceClause.source_id),
                )
                .where(
                    PolicySourceClause.tenant_id == bundle.tenant_id,
                    PolicySourceClause.id.in_(
                        [n.source_clause_id for n in approved_nodes.values() if n.source_clause_id]
                    ),
                )
            )
        ).all()
    }
    wordings = (
        await db.scalars(
            select(PolicyRuleWording).where(
                PolicyRuleWording.tenant_id == bundle.tenant_id,
                PolicyRuleWording.policy_bundle_id == bundle.id,
                PolicyRuleWording.review_state == ReviewState.APPROVED,
                PolicyRuleWording.language == PolicyLanguage.EN,
            )
        )
    ).all()
    rules = (
        await db.scalars(
            select(PolicyRuleRecord)
            .where(
                PolicyRuleRecord.tenant_id == bundle.tenant_id,
                PolicyRuleRecord.policy_bundle_id == bundle.id,
            )
            .order_by(PolicyRuleRecord.rule_id)
        )
    ).all()
    projected = []
    for rule in rules:
        node = approved_nodes.get(rule.node_id)
        if node is None:
            continue
        mine = [w for w in wordings if w.policy_rule_id == rule.id]
        rule.exact_wording = next(
            (w.text for w in mine if w.kind is WordingKind.EXACT_WORDING), None
        )
        rule.prohibited_phrases = sorted(
            w.text for w in mine if w.kind is WordingKind.PROHIBITED_PHRASE
        )
        projected.append(
            PolicyRule(
                rule_id=rule.rule_id,
                rule_version=rule.rule_version,
                title=rule.title,
                requirement=rule.requirement,
                gate=rule.gate,
                severity=rule.severity,
                mandatory=rule.mandatory,
                applies_to=[ValidationInputKind(kind) for kind in node.input_kinds]
                or list(ValidationInputKind),
                source=_source_of(support.get(node.id, []), clause_refs),
            )
        )
    return projected


def _source_of(
    supporting: Sequence[PolicyNode], clause_refs: Mapping[UUID, tuple[str, str]]
) -> SourceReference:
    # The first supporting source clause by key, else the first interpretation (the checks
    # guarantee one of them exists for every approved rule).
    for node in sorted(supporting, key=lambda n: n.node_key):
        if (
            node.node_type is PolicyNodeType.CLAUSE
            and node.source_clause_id is not None
            and node.source_clause_id in clause_refs
        ):
            source_ref, clause_ref = clause_refs[node.source_clause_id]
            return SourceReference(source_id=source_ref, clause=clause_ref)
    for node in sorted(supporting, key=lambda n: n.node_key):
        if node.node_type is PolicyNodeType.INTERPRETATION:
            return SourceReference(source_id=INTERPRETATION_SOURCE, clause=node.node_key)
    raise PolicyItemInvalidError("A rule has no source clause or interpretation.")
