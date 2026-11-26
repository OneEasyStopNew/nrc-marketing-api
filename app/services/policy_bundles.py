"""Policy bundle lifecycle (NRC-4): drafts, graph edits, item review, submit, approve, reject.

Every change locks the bundle row, checks the caller's capability, the If-Match revision and the
lifecycle state, applies the change, bumps the revision and writes an audit row, all in one
transaction. The graph can change only while the bundle is DRAFT (edits) or REVIEW (review
decisions); a database trigger refuses anything else as a backstop. Approval runs the bundle
checks and writes the JSONB rules that validation reads today, and queues the index build the
policy worker runs (app.workers.policy); BUILD queues it again after a failed one. Activation and
retirement extend TRANSITIONS in later steps.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
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
    PolicyLimitError,
    VersionConflictError,
)
from app.core.request_context import request_id_var
from app.models import (
    OutboxEvent,
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
    REASON_REQUIRED_COMMANDS,
    BundleLifecycleState,
    CheckCode,
    Command,
    PolicyAuditAction,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ReviewItemKind,
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
from app.services.policy_checks import Issue, load_snapshot, run_checks
from app.services.policy_index import stale_ready_builds

State = BundleLifecycleState
EDITABLE: Final = frozenset({State.DRAFT})
# Bundles still being written: what POLICY_MAX_OPEN_BUNDLES_PER_TENANT counts.
OPEN: Final = frozenset({State.DRAFT, State.REVIEW})
REVIEWABLE: Final = frozenset({State.DRAFT, State.REVIEW})


@dataclass(frozen=True, slots=True)
class Transition:
    from_states: frozenset[BundleLifecycleState]
    to_state: BundleLifecycleState
    # Any one of these lets the caller run the command.
    capabilities: frozenset[Capability]
    action: PolicyAuditAction


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
    ),
    # Queues the worker's index build again after a failed one (the bundle stays APPROVED).
    Command.BUILD: Transition(
        frozenset({State.APPROVED}),
        State.APPROVED,
        frozenset({Capability.POLICY_ACTIVATE}),
        PolicyAuditAction.BUILD_STARTED,
    ),
    # AC6: activation, rollback and retirement always name a reason.
    Command.ACTIVATE: Transition(
        frozenset({State.READY}),
        State.ACTIVE,
        frozenset({Capability.POLICY_ACTIVATE}),
        PolicyAuditAction.ACTIVATE,
    ),
    # Back to a bundle that was active before (activated_at set); the graph never changed.
    Command.ROLLBACK: Transition(
        frozenset({State.RETIRED}),
        State.ACTIVE,
        frozenset({Capability.POLICY_ACTIVATE}),
        PolicyAuditAction.ROLLBACK,
    ),
    # Leaves the tenant with no active policy: validation then answers NO_ACTIVE_POLICY.
    Command.RETIRE: Transition(
        frozenset({State.ACTIVE}),
        State.RETIRED,
        frozenset({Capability.POLICY_ACTIVATE}),
        PolicyAuditAction.RETIRE,
    ),
}
# Commands that change which bundle is active: serialised per tenant, and announced.
_ACTIVATING: Final = frozenset({Command.ACTIVATE, Command.ROLLBACK})
_CHANGES_ACTIVE: Final = _ACTIVATING | {Command.RETIRE}
# Consumers (approval and dispatch invalidation, NRC-12/13) read ids only and load the rest.
POLICY_BUNDLE_EVENT: Final = "policy_bundle.activated"
POLICY_BUNDLE_EVENT_VERSION: Final = 1


_REVIEW_MODELS: Final[
    Mapping[ReviewItemKind, type[PolicyNode | PolicyEdge | PolicyRuleWording]]
] = {
    ReviewItemKind.NODE: PolicyNode,
    ReviewItemKind.EDGE: PolicyEdge,
    ReviewItemKind.WORDING: PolicyRuleWording,
}
# D-16: the policy owner approves exceptions and disclosures (policy:approve), including the
# except_when edges that attach an exception to a rule: approving the link is what turns the
# exception on. Every other item is reviewed by the policy administrator who drafts the bundle
# (policy:manage). Creating and editing any item stays policy:manage.
OWNER_REVIEWED_NODES: Final = frozenset({PolicyNodeType.EXCEPTION, PolicyNodeType.DISCLOSURE})
OWNER_REVIEWED_EDGES: Final = frozenset({PolicyEdgeType.EXCEPT_WHEN})


def review_capability(item: PolicyNode | PolicyEdge | PolicyRuleWording) -> Capability:
    match item:
        case PolicyNode():
            owner_reviews = item.node_type in OWNER_REVIEWED_NODES
        case PolicyEdge():
            owner_reviews = item.edge_type in OWNER_REVIEWED_EDGES
        case PolicyRuleWording():
            owner_reviews = item.kind is WordingKind.STUDY_APPROVED_WORDING
    return Capability.POLICY_APPROVE if owner_reviews else Capability.POLICY_MANAGE


# Interpretations stand in for a source clause; their reference names the node instead.
INTERPRETATION_SOURCE: Final = "interpretation"


def _queue_build(bundle: PolicyBundle) -> None:
    # What lease_policy_builds() looks for: due now, with a fresh attempt budget.
    bundle.available_at = func.now()
    bundle.attempts = 0
    bundle.failed_attempts = 0
    bundle.last_error_code = None


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
    event: Mapping[str, object] | None = None,
) -> None:
    bundle.revision += 1
    bundle.updated_at = func.now()
    audit_id = uuid4()
    if event is not None:
        # Same transaction as the change: consumers hear of it exactly when it commits.
        db.add(
            OutboxEvent(
                tenant_id=bundle.tenant_id,
                job_id=None,
                event_type=POLICY_BUNDLE_EVENT,
                payload={
                    "event_version": POLICY_BUNDLE_EVENT_VERSION,
                    "tenant_id": str(bundle.tenant_id),
                    "policy_bundle_id": str(bundle.id),
                    "version": bundle.version,
                    "action": action.value,
                    "audit_event_id": str(audit_id),
                    **event,
                },
                attempts=0,
            )
        )
    db.add(
        PolicyAuditEvent(
            id=audit_id,
            tenant_id=bundle.tenant_id,
            policy_bundle_id=bundle.id,
            action=action,
            from_state=from_state,
            to_state=bundle.lifecycle_state if from_state is not None else None,
            actor=ctx.principal.sub,
            reason=reason,
            # D-16: each approval and activation records who acted and in which roles.
            details={
                "revision": bundle.revision,
                "actor_roles": sorted(role.value for role in ctx.roles),
                **(details or {}),
            },
            request_id=request_id_var.get(),
        )
    )
    await db.commit()
    # Load server-side values (timestamps, defaults) for the caller's response.
    await db.refresh(bundle)
    if item is not None:
        await db.refresh(item)


async def create_bundle(
    db: AsyncSession,
    ctx: TenantContext,
    body: BundleCreate,
    max_open_bundles: int,
    max_items: int,
) -> PolicyBundle:
    """A new DRAFT with the tenant's next version, empty or copied from based_on_bundle_id.

    Raises PolicyLimitError when the tenant already has max_open_bundles in DRAFT or REVIEW, or
    the bundle to copy holds more than max_items live nodes, edges and wordings.
    """
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    # The tenant lock also serialises this count, so parallel creates can't both take the last one.
    await lock_tenant_bundles(db, ctx.tenant_id)
    open_bundles = await db.scalar(
        select(func.count()).where(
            PolicyBundle.tenant_id == ctx.tenant_id, PolicyBundle.lifecycle_state.in_(OPEN)
        )
    )
    if (open_bundles or 0) >= max_open_bundles:
        raise PolicyLimitError(
            "The tenant has the most draft and in-review policy bundles it may have. A place "
            "frees when a bundle in review is approved or rejected.",
            details={"limit": max_open_bundles},
        )
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
        # A copy over the cap could take no edit at all (e.g. an owner-imported bundle, or a cap
        # lowered since): refuse it up front.
        if await _live_items(db, ctx.tenant_id, base.id) > max_items:
            raise _limit_reached(max_items)
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


async def _live_items(db: AsyncSession, tenant_id: UUID, bundle_id: UUID) -> int:
    # Nodes, edges and wordings that are not REJECTED: rejected rows stay for history (no DELETE)
    # and never take effect, so they don't use up POLICY_MAX_ITEMS_PER_BUNDLE.
    counts = [
        select(func.count())
        .select_from(model)
        .where(
            model.tenant_id == tenant_id,
            model.policy_bundle_id == bundle_id,
            model.review_state != ReviewState.REJECTED,
        )
        .scalar_subquery()
        for model in (PolicyNode, PolicyEdge, PolicyRuleWording)
    ]
    return int(await db.scalar(select(counts[0] + counts[1] + counts[2])) or 0)


def _limit_reached(max_items: int) -> PolicyLimitError:
    return PolicyLimitError(
        "This policy bundle holds the most nodes, edges and wordings it may have. Rejected "
        "items don't count: reject ones you no longer need, or split the policy.",
        details={"limit": max_items},
    )


async def _room_for_items(db: AsyncSession, bundle: PolicyBundle, max_items: int) -> None:
    # Under the bundle row lock (_open), so parallel adds can't both take the last place. One
    # request adds at most a rule and its node over the line.
    if await _live_items(db, bundle.tenant_id, bundle.id) >= max_items:
        raise _limit_reached(max_items)


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
    # Request models refuse a window they hold whole (422). This catches one sent date against
    # the other one stored: a conflict with the node as it is.
    start, end = values.get("effective_from"), values.get("effective_to")
    if isinstance(start, datetime) and isinstance(end, datetime) and start > end:
        raise PolicyItemInvalidError("effective_from must not be after effective_to.")


def _changes(body: NodeUpdate | RuleUpdate) -> dict[str, object]:
    # Fields the caller sent. The request model already refused nulls for required fields.
    return body.model_dump(exclude_unset=True)


async def _open_for_add(
    db: AsyncSession, ctx: TenantContext, bundle_id: UUID, revision: int, max_items: int
) -> PolicyBundle:
    # Every add: policy:manage, the draft locked at the expected revision, and room under the cap
    # (checked under that lock).
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    await _room_for_items(db, bundle, max_items)
    return bundle


async def _room_to_revive(
    db: AsyncSession,
    bundle: PolicyBundle,
    item: PolicyNode | PolicyEdge | PolicyRuleWording,
    max_items: int,
) -> None:
    # A REJECTED item doesn't count toward POLICY_MAX_ITEMS_PER_BUNDLE; one leaving REJECTED
    # (a review decision, or an edit that proposes it again) takes a place like an add does.
    if item.review_state is ReviewState.REJECTED:
        await _room_for_items(db, bundle, max_items)


def _back_to_proposed(item: PolicyNode | PolicyEdge | PolicyRuleWording) -> None:
    # A changed item has to be reviewed again before it can count.
    item.review_state = ReviewState.PROPOSED
    item.reviewed_by = None
    item.reviewed_at = None
    item.review_note = None
    if not isinstance(item, PolicyRuleWording):
        item.approved = False


async def add_node(
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    body: NodeCreate,
    max_items: int,
) -> PolicyNode:
    bundle = await _open_for_add(db, ctx, bundle_id, revision, max_items)
    await _node_key_free(db, bundle, body.node_key)
    await _require_clause(db, ctx.tenant_id, body.source_clause_id)
    values = body.model_dump()
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
    max_items: int,
) -> PolicyNode:
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    node = await _in_bundle(db, PolicyNode, bundle, node_id, "Node")
    await _room_to_revive(db, bundle, node, max_items)
    changes = _changes(body)
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
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    body: RuleCreate,
    max_items: int,
) -> PolicyRuleRecord:
    """A rule and its RULE node (node_key = rule_id), both PROPOSED."""
    bundle = await _open_for_add(db, ctx, bundle_id, revision, max_items)
    await _node_key_free(db, bundle, body.rule_id)
    rule_fields = {"rule_id", "rule_version", "title", "requirement", "gate", "severity"}
    rule_fields |= {"mandatory", "check_type"}
    node_values = body.model_dump(exclude=rule_fields)
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
    max_items: int,
) -> PolicyRuleRecord:
    _require(ctx, frozenset({Capability.POLICY_MANAGE}))
    bundle = await _open(db, ctx, bundle_id, revision, EDITABLE)
    rule = await _in_bundle(db, PolicyRuleRecord, bundle, rule_record_id, "Rule")
    node = await _in_bundle(db, PolicyNode, bundle, rule.node_id, "Rule node")
    await _room_to_revive(db, bundle, node, max_items)
    for name, value in _changes(body).items():
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
    db: AsyncSession,
    ctx: TenantContext,
    bundle_id: UUID,
    revision: int,
    body: EdgeCreate,
    max_items: int,
) -> PolicyEdge:
    bundle = await _open_for_add(db, ctx, bundle_id, revision, max_items)
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
    max_items: int,
) -> PolicyRuleWording:
    """Required wording or a prohibited phrase for a rule. Kept exactly as given (AC7)."""
    bundle = await _open_for_add(db, ctx, bundle_id, revision, max_items)
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
    max_items: int,
) -> None:
    """Approve or reject one node, edge or wording while the bundle is a draft or in review."""
    # Outsiders are refused before any lookup; the exact capability depends on the item.
    _require(ctx, frozenset({Capability.POLICY_MANAGE, Capability.POLICY_APPROVE}))
    bundle = await _open(db, ctx, bundle_id, revision, REVIEWABLE)
    item = await _in_bundle(db, _REVIEW_MODELS[kind], bundle, item_id, kind.value.capitalize())
    _require(ctx, frozenset({review_capability(item)}))
    decision = ReviewState(body.decision)
    if decision is not ReviewState.REJECTED:
        await _room_to_revive(db, bundle, item, max_items)
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
    # The API refuses a missing reason with a 422 (TransitionCommandRequest); this guards every
    # other caller.
    if command in REASON_REQUIRED_COMMANDS and body.reason is None:
        raise PolicyItemInvalidError("Give a reason.", details={"field": "reason"})
    if command in _CHANGES_ACTIVE:
        # Taken before the row lock, in the same order everywhere, so two activations in one
        # tenant run one after the other instead of deadlocking.
        await lock_tenant_bundles(db, ctx.tenant_id)
    bundle = await _open(db, ctx, bundle_id, revision, step.from_states)
    from_state = bundle.lifecycle_state
    details: dict[str, object] = {}
    event: dict[str, object] | None = None
    if command is Command.APPROVE:
        details = await _approve(db, ctx, bundle, supported_languages)
    elif command is Command.BUILD:
        _queue_build(bundle)
    elif command in _ACTIVATING:
        if command is Command.ROLLBACK and bundle.activated_at is None:
            raise PolicyBundleStateError(
                "Only a bundle that was active before can be rolled back to.",
                details={
                    "state": bundle.lifecycle_state.value,
                    "allowed": [BundleLifecycleState.RETIRED.value],
                },
            )
        previous = await _activate(db, ctx, bundle, supported_languages, body.reason)
        details = event = {"previous_bundle_id": str(previous) if previous else None}
    elif command is Command.RETIRE:
        event = {"previous_bundle_id": None}
    bundle.lifecycle_state = step.to_state
    await _finish(
        db,
        ctx,
        bundle,
        step.action,
        from_state=from_state,
        reason=body.reason,
        details=details,
        event=event,
    )
    return bundle


async def _activate(
    db: AsyncSession,
    ctx: TenantContext,
    bundle: PolicyBundle,
    supported_languages: Sequence[PolicyLanguage],
    reason: str | None,
) -> UUID | None:
    # Under the tenant lock: re-check the bundle and its index, then retire the active bundle
    # first, because a tenant may have only one ACTIVE bundle (unique index). Returns its id.
    snapshot = await load_snapshot(db, bundle.tenant_id, bundle.id)
    issues = run_checks(snapshot, supported_languages)
    issues += [
        Issue(
            CheckCode.INDEX_STALE,
            f"build:{build_id}",
            "The index no longer matches the bundle's approved nodes.",
        )
        for build_id in await stale_ready_builds(db, bundle.tenant_id, bundle.id)
    ]
    if issues:
        raise _has_issues(issues, "The policy bundle has open issues and cannot be activated.")
    previous = await db.scalar(
        select(PolicyBundle)
        .where(
            PolicyBundle.tenant_id == bundle.tenant_id,
            PolicyBundle.lifecycle_state == State.ACTIVE,
            PolicyBundle.id != bundle.id,
        )
        .with_for_update()
    )
    if previous is not None:
        previous.lifecycle_state = State.RETIRED
        previous.revision += 1
        previous.updated_at = func.now()
        db.add(
            PolicyAuditEvent(
                tenant_id=previous.tenant_id,
                policy_bundle_id=previous.id,
                action=PolicyAuditAction.RETIRE,
                from_state=State.ACTIVE,
                to_state=State.RETIRED,
                actor=ctx.principal.sub,
                reason=reason,
                details={
                    "revision": previous.revision,
                    "replaced_by_bundle_id": str(bundle.id),
                },
                request_id=request_id_var.get(),
            )
        )
        await db.flush()
    bundle.activated_at = func.now()
    bundle.activated_by = ctx.principal.sub
    return previous.id if previous else None


def _has_issues(issues: Sequence[Issue], message: str | None = None) -> PolicyBundleHasIssuesError:
    # The one place the issues' response shape (details.issues) is written.
    return PolicyBundleHasIssuesError(
        message,
        details={
            "issues": [
                {"code": i.code.value, "subject": i.subject, "message": i.message} for i in issues
            ]
        },
    )


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
        raise _has_issues(issues)
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
    _queue_build(bundle)
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
