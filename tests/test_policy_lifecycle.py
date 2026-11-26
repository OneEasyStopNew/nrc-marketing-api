"""Policy bundle lifecycle: drafts, edits, item review, submit, approve, reject (NRC-4)."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.auth.access import Capability, Role, TenantContext, capabilities_for
from app.auth.tokens import Principal
from app.core.errors import (
    NotFoundError,
    PermissionDeniedError,
    PolicyBundleHasIssuesError,
    PolicyBundleStateError,
    PolicyItemInvalidError,
    VersionConflictError,
)
from app.db.session import TenantSession, bind_tenant
from app.models import (
    OutboxEvent,
    PolicyAuditEvent,
    PolicyBundle,
    PolicyEdge,
    PolicyIndexBuild,
    PolicyNode,
    PolicyRuleRecord,
    PolicyRuleWording,
    PolicySource,
    PolicySourceClause,
)
from app.models.enums import (
    BundleLifecycleState,
    CheckType,
    Command,
    Gate,
    IndexBuildState,
    PolicyAuditAction,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ReviewItemKind,
    ReviewState,
    Severity,
    WordingKind,
)
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
from app.services import policy_bundles as bundles
from app.services.policies import get_active_policy
from tests.conftest import (
    IntegrationEnv,
    SeededTenant,
    World,
    add_policy_bundle,
    app_session,
    as_tenant_owner,
)

EN = (PolicyLanguage.EN,)
ADMIN, APPROVER = "policy-admin-a", "policy-approver-a"
CLAUSE_TEXT = "[SYNTHETIC] Every post must say: This is a research study. Taking part is voluntary."
SOURCE_REF = "synthetic:nrc-4-lifecycle"
NO_REASON = TransitionRequest()
# Bundle and item caps are tested in test_policy_admin_api; these tests stay well under them.
TEST_MAX_OPEN_BUNDLES = 100
TEST_MAX_ITEMS = 20_000


def ctx_for(tenant_id: UUID, sub: str, role: Role) -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        principal=Principal(sub=sub, username=sub, scopes=frozenset()),
        roles=frozenset({role}),
        capabilities=capabilities_for({role}),
        study_ids=frozenset(),
    )


@dataclass
class Draft:
    # A bundle built through the service, with the ids a test needs.
    id: UUID
    revision: int
    clause_node: UUID
    rule: UUID
    rule_node: UUID
    edge: UUID
    wording: UUID


class Env:
    def __init__(self, db: AsyncSession, tenant_id: UUID, clause_id: UUID) -> None:
        self.db = db
        self.tenant_id = tenant_id
        self.clause_id = clause_id
        self.admin = ctx_for(tenant_id, ADMIN, Role.POLICY_ADMIN)
        self.approver = ctx_for(tenant_id, APPROVER, Role.POLICY_APPROVER)

    async def revision(self, bundle_id: UUID) -> int:
        await bind_tenant(self.db, self.tenant_id)
        value = await self.db.scalar(
            select(PolicyBundle.revision).where(PolicyBundle.id == bundle_id)
        )
        assert value is not None
        return value

    async def create(self, **changes: object) -> PolicyBundle:
        body = BundleCreate.model_validate(
            {"name": "[SYNTHETIC] Policy", "source_ref": SOURCE_REF, "is_synthetic": True} | changes
        )
        return await bundles.create_bundle(
            self.db, self.admin, body, TEST_MAX_OPEN_BUNDLES, TEST_MAX_ITEMS
        )

    async def draft(self) -> Draft:
        # One rule, supported by a clause node that cites the source clause, with exact wording
        # taken verbatim from that clause. Nothing reviewed yet.
        bundle = await self.create()
        rev = bundle.revision
        clause = await bundles.add_node(
            self.db,
            self.admin,
            bundle.id,
            rev,
            NodeCreate(
                node_key="CL-1",
                node_type=PolicyNodeType.CLAUSE,
                text=CLAUSE_TEXT,
                source_clause_id=self.clause_id,
            ),
            TEST_MAX_ITEMS,
        )
        rule = await bundles.add_rule(
            self.db,
            self.admin,
            bundle.id,
            rev + 1,
            RuleCreate(
                rule_id="REC-07",
                title="Research and voluntary disclosure",
                requirement="Say that this is a research study and that taking part is voluntary.",
                gate=Gate.COMPLETENESS,
                severity=Severity.CRITICAL,
                mandatory=True,
                check_type=CheckType.DETERMINISTIC,
                text="[SYNTHETIC] Say that this is a research study.",
            ),
            TEST_MAX_ITEMS,
        )
        edge = await bundles.add_edge(
            self.db,
            self.admin,
            bundle.id,
            rev + 2,
            EdgeCreate(
                edge_type=PolicyEdgeType.SUPPORTED_BY,
                from_node_id=rule.node_id,
                to_node_id=clause.id,
                provenance_clause_id=self.clause_id,
            ),
            TEST_MAX_ITEMS,
        )
        wording = await bundles.add_wording(
            self.db,
            self.admin,
            bundle.id,
            rev + 3,
            rule.id,
            WordingCreate(
                kind=WordingKind.EXACT_WORDING,
                text="This is a research study. Taking part is voluntary.",
            ),
            TEST_MAX_ITEMS,
        )
        return Draft(bundle.id, rev + 4, clause.id, rule.id, rule.node_id, edge.id, wording.id)

    async def review_all(self, draft: Draft, decision: str = "APPROVED") -> None:
        items = [
            (ReviewItemKind.NODE, draft.clause_node),
            (ReviewItemKind.NODE, draft.rule_node),
            (ReviewItemKind.EDGE, draft.edge),
            (ReviewItemKind.WORDING, draft.wording),
        ]
        for kind, item_id in items:
            await bundles.review_item(
                self.db,
                self.admin,
                draft.id,
                draft.revision,
                kind,
                item_id,
                ReviewDecision(decision=decision),
                TEST_MAX_ITEMS,
            )
            draft.revision += 1

    async def move(
        self,
        draft: Draft,
        command: Command,
        ctx: TenantContext | None = None,
        body: TransitionRequest = NO_REASON,
    ) -> PolicyBundle:
        bundle = await bundles.transition(
            self.db, ctx or self.admin, draft.id, draft.revision, command, body, EN
        )
        draft.revision = bundle.revision
        return bundle

    async def audit(self, bundle_id: UUID) -> list[PolicyAuditAction]:
        await bind_tenant(self.db, self.tenant_id)
        return list(
            (
                await self.db.scalars(
                    select(PolicyAuditEvent.action)
                    .where(PolicyAuditEvent.policy_bundle_id == bundle_id)
                    # Revisions only grow; tests share one transaction, so created_at ties.
                    .order_by(PolicyAuditEvent.details["revision"].as_integer())
                )
            ).all()
        )


async def add_source_clause(connection: AsyncConnection, tenant_id: UUID) -> UUID:
    # An owner-imported synthetic source with one clause (EXTRACTED, so no upload needed).
    source_id, clause_id = uuid4(), uuid4()
    async with as_tenant_owner(connection, tenant_id):
        await connection.execute(
            insert(PolicySource).values(
                id=source_id,
                tenant_id=tenant_id,
                title="[SYNTHETIC] Lifecycle source",
                source_ref=SOURCE_REF,
                checksum_sha256=uuid4().hex * 2,
                language=PolicyLanguage.EN,
                is_synthetic=True,
                created_by="synthetic-fixture",
            )
        )
        await connection.execute(
            insert(PolicySourceClause).values(
                id=clause_id,
                tenant_id=tenant_id,
                source_id=source_id,
                clause_ref="DISC-1",
                section="Disclosures",
                page=1,
                text=CLAUSE_TEXT,
                checksum_sha256=uuid4().hex * 2,
                language=PolicyLanguage.EN,
            )
        )
    return clause_id


@pytest.fixture
async def env(connection: AsyncConnection, world: World) -> AsyncIterator[Env]:
    clause_id = await add_source_clause(connection, world.a.tenant_id)
    async with app_session(connection) as db:
        await bind_tenant(db, world.a.tenant_id)
        yield Env(db, world.a.tenant_id, clause_id)


# ---- Create


async def test_a_new_bundle_is_a_draft_with_the_next_version(env: Env, world: World) -> None:
    bundle = await env.create()

    assert bundle.lifecycle_state is BundleLifecycleState.DRAFT
    assert bundle.revision == 1
    assert bundle.rules == []
    assert bundle.approved_by is None
    # The world's tenant already has version 1 (its demo bundle).
    assert bundle.version == 2
    assert (await env.create()).version == 3
    assert await env.audit(bundle.id) == [PolicyAuditAction.CREATE]


async def test_only_policy_admins_create_bundles(env: Env) -> None:
    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await bundles.create_bundle(
            env.db,
            env.approver,
            BundleCreate(name="[SYNTHETIC] x", source_ref="synthetic:x", is_synthetic=True),
            TEST_MAX_OPEN_BUNDLES,
            TEST_MAX_ITEMS,
        )


async def test_a_copy_keeps_the_graph_and_its_review(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)

    copy = await env.create(based_on_bundle_id=draft.id)

    assert copy.based_on_bundle_id == draft.id
    nodes = (
        await env.db.scalars(select(PolicyNode).where(PolicyNode.policy_bundle_id == copy.id))
    ).all()
    assert sorted(n.node_key for n in nodes) == ["CL-1", "REC-07"]
    assert {n.review_state for n in nodes} == {ReviewState.APPROVED}
    assert {n.reviewed_by for n in nodes} == {ADMIN}
    rule = await env.db.scalar(
        select(PolicyRuleRecord).where(PolicyRuleRecord.policy_bundle_id == copy.id)
    )
    assert rule is not None
    assert rule.node_id in {n.id for n in nodes}
    edge = await env.db.scalar(select(PolicyEdge).where(PolicyEdge.policy_bundle_id == copy.id))
    assert edge is not None
    assert {edge.from_node_id, edge.to_node_id} == {n.id for n in nodes}
    wording = await env.db.scalar(
        select(PolicyRuleWording).where(PolicyRuleWording.policy_bundle_id == copy.id)
    )
    assert wording is not None
    assert wording.policy_rule_id == rule.id
    # The original is untouched.
    assert await env.revision(draft.id) == draft.revision


async def other_tenants_bundle(connection: AsyncConnection, world: World, env: Env) -> UUID:
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_id = await add_policy_bundle(
            connection, world.b.tenant_id, version=1, state=BundleLifecycleState.DRAFT
        )
    await bind_tenant(env.db, env.tenant_id)
    return bundle_id


async def test_copying_another_tenants_bundle_is_not_found(
    connection: AsyncConnection, env: Env, world: World
) -> None:
    other = await other_tenants_bundle(connection, world, env)

    with pytest.raises(NotFoundError, match="to copy not found"):
        await env.create(based_on_bundle_id=other)


# ---- Edits


async def test_every_edit_bumps_the_revision_and_is_audited(env: Env) -> None:
    draft = await env.draft()

    assert draft.revision == 5
    assert await env.audit(draft.id) == [PolicyAuditAction.CREATE] + [PolicyAuditAction.EDIT] * 4
    node = await env.db.get(PolicyNode, draft.clause_node)
    assert node is not None
    assert node.review_state is ReviewState.PROPOSED
    assert node.approved is False


async def test_a_stale_revision_is_refused(env: Env) -> None:
    draft = await env.draft()

    with pytest.raises(VersionConflictError, match="changed since you loaded it") as error:
        await bundles.update_node(
            env.db,
            env.admin,
            draft.id,
            draft.revision - 1,
            draft.clause_node,
            NodeUpdate(text="[SYNTHETIC] Changed."),
            TEST_MAX_ITEMS,
        )
    assert error.value.details == {"current_version": draft.revision}


async def test_editing_a_node_sends_it_back_to_review(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)

    node = await bundles.update_node(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        draft.clause_node,
        NodeUpdate(jurisdictions=["US"]),
        TEST_MAX_ITEMS,
    )

    assert node.jurisdictions == ["US"]
    assert node.review_state is ReviewState.PROPOSED
    assert node.reviewed_by is None
    assert node.approved is False


async def test_editing_a_rule_sends_its_node_back_to_review(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)

    rule = await bundles.update_rule(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        draft.rule,
        RuleUpdate(severity=Severity.MAJOR),
        TEST_MAX_ITEMS,
    )

    assert rule.severity is Severity.MAJOR
    node = await env.db.get(PolicyNode, draft.rule_node)
    assert node is not None
    assert node.review_state is ReviewState.PROPOSED


def test_a_required_field_cannot_be_cleared() -> None:
    # Refused by the request model (a 422), so no update ever sees it.
    with pytest.raises(ValidationError, match="These fields cannot be empty: text"):
        NodeUpdate.model_validate({"text": None})
    assert NodeUpdate.model_validate({"effective_to": None}).effective_to is None


async def test_a_node_key_is_used_once_per_bundle(env: Env) -> None:
    draft = await env.draft()

    with pytest.raises(PolicyItemInvalidError, match="already uses this key"):
        await bundles.add_node(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            NodeCreate(node_key="REC-07", node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC] x"),
            TEST_MAX_ITEMS,
        )


async def test_a_rule_has_one_live_exact_wording_per_language(env: Env) -> None:
    draft = await env.draft()

    with pytest.raises(PolicyItemInvalidError, match="reject it first"):
        await bundles.add_wording(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            draft.rule,
            WordingCreate(kind=WordingKind.EXACT_WORDING, text="This is a research study."),
            TEST_MAX_ITEMS,
        )


async def test_an_edge_must_join_nodes_of_the_same_bundle(env: Env, world: World) -> None:
    draft = await env.draft()
    other = await env.draft()

    with pytest.raises(NotFoundError, match="Node not found in this policy bundle"):
        await bundles.add_edge(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            EdgeCreate(
                edge_type=PolicyEdgeType.SUPPORTED_BY,
                from_node_id=draft.rule_node,
                to_node_id=other.clause_node,
                provenance_note="[SYNTHETIC] note",
            ),
            TEST_MAX_ITEMS,
        )


async def test_an_unknown_source_clause_is_not_found(env: Env) -> None:
    bundle = await env.create()

    with pytest.raises(NotFoundError, match="Source clause not found"):
        await bundles.add_node(
            env.db,
            env.admin,
            bundle.id,
            bundle.revision,
            NodeCreate(
                node_key="CL-X",
                node_type=PolicyNodeType.CLAUSE,
                text="[SYNTHETIC] x",
                source_clause_id=uuid4(),
            ),
            TEST_MAX_ITEMS,
        )


async def test_approvers_cannot_edit(env: Env) -> None:
    bundle = await env.create()

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await bundles.add_node(
            env.db,
            env.approver,
            bundle.id,
            bundle.revision,
            NodeCreate(node_key="EX-1", node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC] x"),
            TEST_MAX_ITEMS,
        )


async def test_another_tenants_bundle_is_not_found(
    connection: AsyncConnection, env: Env, world: World
) -> None:
    other = await other_tenants_bundle(connection, world, env)

    with pytest.raises(NotFoundError, match="Policy bundle not found"):
        await bundles.add_node(
            env.db,
            env.admin,
            other,
            1,
            NodeCreate(node_key="EX-1", node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC] x"),
            TEST_MAX_ITEMS,
        )


# ---- Review and transitions


async def test_the_whole_way_to_approval(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)
    await env.move(draft, Command.SUBMIT)

    bundle = await env.move(draft, Command.APPROVE, env.approver)

    assert bundle.lifecycle_state is BundleLifecycleState.APPROVED
    assert bundle.approved_by == APPROVER
    assert bundle.approved_at is not None
    # Approval queues the worker's index build.
    assert bundle.available_at is not None
    assert (bundle.attempts, bundle.failed_attempts, bundle.last_error_code) == (0, 0, None)
    assert bundle.rules == [
        {
            "rule_id": "REC-07",
            "rule_version": 1,
            "title": "Research and voluntary disclosure",
            "requirement": "Say that this is a research study and that taking part is voluntary.",
            "gate": "completeness",
            "severity": "critical",
            "mandatory": True,
            "applies_to": ["brief", "text", "image"],
            "source": {"source_id": SOURCE_REF, "clause": "DISC-1", "excerpt": None},
        }
    ]
    rule = await env.db.get(PolicyRuleRecord, draft.rule, populate_existing=True)
    assert rule is not None
    assert rule.exact_wording == "This is a research study. Taking part is voluntary."
    assert await env.audit(draft.id) == [
        PolicyAuditAction.CREATE,
        *[PolicyAuditAction.EDIT] * 4,
        *[PolicyAuditAction.REVIEW] * 4,
        PolicyAuditAction.SUBMIT,
        PolicyAuditAction.APPROVE,
    ]
    event = await env.db.scalar(
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.policy_bundle_id == draft.id,
            PolicyAuditEvent.action == PolicyAuditAction.APPROVE,
        )
    )
    assert event is not None
    assert (event.actor, event.from_state, event.to_state) == (
        APPROVER,
        BundleLifecycleState.REVIEW,
        BundleLifecycleState.APPROVED,
    )
    assert event.details == {
        "revision": draft.revision,
        "actor_roles": ["policy_approver"],
        "rules": 1,
    }


async def test_items_can_be_reviewed_while_in_review(env: Env) -> None:
    draft = await env.draft()
    await env.move(draft, Command.SUBMIT)

    await env.review_all(draft)

    assert await env.revision(draft.id) == draft.revision


async def test_approval_with_open_issues_changes_nothing(env: Env) -> None:
    draft = await env.draft()
    await env.move(draft, Command.SUBMIT)

    with pytest.raises(PolicyBundleHasIssuesError, match="open issues") as error:
        await env.move(draft, Command.APPROVE, env.approver)

    assert error.value.details is not None
    issues = error.value.details["issues"]
    assert isinstance(issues, list)
    assert {issue["code"] for issue in issues} == {"UNREVIEWED_ITEM", "EMPTY_BUNDLE"}
    await env.db.rollback()
    bundle = await env.db.get(PolicyBundle, draft.id, populate_existing=True)
    assert bundle is not None
    assert bundle.lifecycle_state is BundleLifecycleState.REVIEW
    assert bundle.revision == draft.revision
    assert bundle.rules == []


async def test_policy_admins_cannot_approve_or_reject(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)
    await env.move(draft, Command.SUBMIT)

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, Command.APPROVE, env.admin)
    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, Command.REJECT, env.admin, TransitionRequest(reason="[SYNTHETIC]"))


async def test_approvers_cannot_submit_or_review_a_clause(env: Env) -> None:
    # Clauses, rules, edges and quoted wording are the administrator's review (D-16).
    draft = await env.draft()

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, Command.SUBMIT, env.approver)
    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await bundles.review_item(
            env.db,
            env.approver,
            draft.id,
            draft.revision,
            ReviewItemKind.NODE,
            draft.clause_node,
            ReviewDecision(decision="APPROVED"),
            TEST_MAX_ITEMS,
        )


async def add_exception(env: Env, draft: Draft) -> UUID:
    node = await bundles.add_node(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        NodeCreate(
            node_key="EXC-1",
            node_type=PolicyNodeType.EXCEPTION,
            text="[SYNTHETIC] Unless the post only names the study.",
        ),
        TEST_MAX_ITEMS,
    )
    draft.revision += 1
    return node.id


async def review(env: Env, draft: Draft, ctx: TenantContext, node_id: UUID) -> None:
    await bundles.review_item(
        env.db,
        ctx,
        draft.id,
        draft.revision,
        ReviewItemKind.NODE,
        node_id,
        ReviewDecision(decision="APPROVED"),
        TEST_MAX_ITEMS,
    )
    draft.revision += 1


# D-16: the policy owner approves exceptions and disclosures; the administrator drafts them.
async def test_an_approver_reviews_an_exception_and_the_audit_names_the_role(env: Env) -> None:
    draft = await env.draft()
    exception_id = await add_exception(env, draft)

    await review(env, draft, env.approver, exception_id)

    node = await env.db.get(PolicyNode, exception_id, populate_existing=True)
    assert node is not None
    assert (node.review_state, node.reviewed_by) == (ReviewState.APPROVED, APPROVER)
    event = await env.db.scalar(
        select(PolicyAuditEvent)
        .where(
            PolicyAuditEvent.policy_bundle_id == draft.id,
            PolicyAuditEvent.action == PolicyAuditAction.REVIEW,
        )
        .order_by(PolicyAuditEvent.created_at.desc(), PolicyAuditEvent.id)
        .limit(1)
    )
    assert event is not None
    assert (event.actor, event.details["actor_roles"], event.details["item_id"]) == (
        APPROVER,
        ["policy_approver"],
        str(exception_id),
    )


@pytest.mark.parametrize("role", [Role.POLICY_ADMIN, Role.CREATOR, Role.REVIEWER, Role.ADMIN])
async def test_only_an_approver_reviews_an_exception(env: Env, role: Role) -> None:
    draft = await env.draft()
    exception_id = await add_exception(env, draft)

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await review(env, draft, ctx_for(env.tenant_id, "caller-a", role), exception_id)


async def add_except_when(env: Env, draft: Draft, exception_id: UUID) -> UUID:
    edge = await bundles.add_edge(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        EdgeCreate(
            edge_type=PolicyEdgeType.EXCEPT_WHEN,
            from_node_id=draft.rule_node,
            to_node_id=exception_id,
            provenance_note="[SYNTHETIC] The clause names this exception.",
        ),
        TEST_MAX_ITEMS,
    )
    draft.revision += 1
    return edge.id


async def test_only_an_approver_reviews_the_link_that_turns_an_exception_on(env: Env) -> None:
    # Approving the except_when edge is what makes the exception apply to the rule (D-16).
    draft = await env.draft()
    edge_id = await add_except_when(env, draft, await add_exception(env, draft))
    decide = ReviewDecision(decision="APPROVED")

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await bundles.review_item(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            ReviewItemKind.EDGE,
            edge_id,
            decide,
            TEST_MAX_ITEMS,
        )
    await bundles.review_item(
        env.db,
        env.approver,
        draft.id,
        draft.revision,
        ReviewItemKind.EDGE,
        edge_id,
        decide,
        TEST_MAX_ITEMS,
    )

    edge = await env.db.get(PolicyEdge, edge_id, populate_existing=True)
    assert edge is not None
    assert (edge.review_state, edge.reviewed_by) == (ReviewState.APPROVED, APPROVER)


# Who reviews each kind of item (D-16). Every kind is listed, so a new one fails here first.
REVIEWERS: list[tuple[PolicyNode | PolicyEdge | PolicyRuleWording, Capability]] = [
    (PolicyNode(node_type=PolicyNodeType.CLAUSE), Capability.POLICY_MANAGE),
    (PolicyNode(node_type=PolicyNodeType.RULE), Capability.POLICY_MANAGE),
    (PolicyNode(node_type=PolicyNodeType.INTERPRETATION), Capability.POLICY_MANAGE),
    (PolicyNode(node_type=PolicyNodeType.EXAMPLE), Capability.POLICY_MANAGE),
    (PolicyNode(node_type=PolicyNodeType.EXCEPTION), Capability.POLICY_APPROVE),
    (PolicyNode(node_type=PolicyNodeType.DISCLOSURE), Capability.POLICY_APPROVE),
    *(
        (
            PolicyEdge(edge_type=edge_type),
            Capability.POLICY_APPROVE
            if edge_type is PolicyEdgeType.EXCEPT_WHEN
            else Capability.POLICY_MANAGE,
        )
        for edge_type in PolicyEdgeType
    ),
    (PolicyRuleWording(kind=WordingKind.EXACT_WORDING), Capability.POLICY_MANAGE),
    (PolicyRuleWording(kind=WordingKind.PROHIBITED_PHRASE), Capability.POLICY_MANAGE),
    (PolicyRuleWording(kind=WordingKind.STUDY_APPROVED_WORDING), Capability.POLICY_APPROVE),
]


def test_every_item_kind_has_its_reviewer() -> None:
    assert {type(item).__name__ + ":" + str(_kind(item)) for item, _ in REVIEWERS} == {
        *(f"PolicyNode:{t}" for t in PolicyNodeType),
        *(f"PolicyEdge:{t}" for t in PolicyEdgeType),
        *(f"PolicyRuleWording:{k}" for k in WordingKind),
    }
    for item, capability in REVIEWERS:
        assert bundles.review_capability(item) is capability, item


def _kind(item: PolicyNode | PolicyEdge | PolicyRuleWording) -> object:
    match item:
        case PolicyNode():
            return item.node_type
        case PolicyEdge():
            return item.edge_type
        case PolicyRuleWording():
            return item.kind


async def test_an_approver_can_send_a_bundle_back_to_draft(env: Env) -> None:
    draft = await env.draft()
    await env.move(draft, Command.SUBMIT)

    bundle = await env.move(
        draft, Command.RETURN_TO_DRAFT, env.approver, TransitionRequest(reason="[SYNTHETIC] fix")
    )

    assert bundle.lifecycle_state is BundleLifecycleState.DRAFT


async def test_rejection_needs_a_reason_and_is_final(env: Env) -> None:
    draft = await env.draft()
    await env.move(draft, Command.SUBMIT)

    with pytest.raises(PolicyItemInvalidError, match="Give a reason"):
        await env.move(draft, Command.REJECT, env.approver)
    bundle = await env.move(
        draft, Command.REJECT, env.approver, TransitionRequest(reason="[SYNTHETIC] wrong source")
    )

    assert bundle.lifecycle_state is BundleLifecycleState.REJECTED
    with pytest.raises(PolicyBundleStateError, match="not in a state") as error:
        await env.move(draft, Command.SUBMIT)
    assert error.value.details == {"state": "REJECTED", "allowed": ["DRAFT"]}
    event = await env.db.scalar(
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.policy_bundle_id == draft.id,
            PolicyAuditEvent.action == PolicyAuditAction.REJECT,
        )
    )
    assert event is not None
    assert event.reason == "[SYNTHETIC] wrong source"


@pytest.mark.parametrize(
    ("command", "state"),
    [
        (Command.APPROVE, "DRAFT"),
        (Command.REJECT, "DRAFT"),
        (Command.RETURN_TO_DRAFT, "DRAFT"),
    ],
)
async def test_commands_refuse_the_wrong_state(env: Env, command: Command, state: str) -> None:
    draft = await env.draft()
    ctx = env.admin if command is Command.RETURN_TO_DRAFT else env.approver

    with pytest.raises(PolicyBundleStateError, match="not in a state") as error:
        await env.move(draft, command, ctx, TransitionRequest(reason="[SYNTHETIC]"))
    assert error.value.details is not None
    assert error.value.details["state"] == state


async def approved_bundle(env: Env) -> Draft:
    draft = await env.draft()
    await env.review_all(draft)
    await env.move(draft, Command.SUBMIT)
    await env.move(draft, Command.APPROVE, env.approver)
    return draft


async def test_build_queues_an_approved_bundle_again(env: Env) -> None:
    draft = await approved_bundle(env)
    # As a failed build leaves it: not queued, with the failure's code.
    await env.db.execute(
        update(PolicyBundle)
        .where(PolicyBundle.id == draft.id)
        .values(available_at=None, attempts=3, failed_attempts=1, last_error_code="FAILED_CODE")
    )
    await env.db.commit()

    bundle = await env.move(draft, Command.BUILD, env.admin)

    assert bundle.lifecycle_state is BundleLifecycleState.APPROVED
    assert bundle.available_at is not None
    assert (bundle.attempts, bundle.failed_attempts, bundle.last_error_code) == (0, 0, None)
    event = await env.db.scalar(
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.policy_bundle_id == draft.id,
            PolicyAuditEvent.action == PolicyAuditAction.BUILD_STARTED,
        )
    )
    assert event is not None
    assert (event.actor, event.from_state, event.to_state) == (
        ADMIN,
        BundleLifecycleState.APPROVED,
        BundleLifecycleState.APPROVED,
    )


# D-16: approval and activation are separate permissions. The capability is checked before the
# bundle is loaded, so one approved bundle serves every command.
@pytest.mark.parametrize(
    "command", [Command.BUILD, Command.ACTIVATE, Command.ROLLBACK, Command.RETIRE]
)
async def test_approvers_cannot_activate_build_roll_back_or_retire(
    env: Env, command: Command
) -> None:
    draft = await approved_bundle(env)

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, command, env.approver, because())


@pytest.mark.parametrize("command", list(Command))
@pytest.mark.parametrize("role", [Role.CREATOR, Role.REVIEWER, Role.PUBLISHER, Role.ADMIN])
async def test_roles_without_policy_capabilities_cannot_move_a_bundle(
    env: Env, command: Command, role: Role
) -> None:
    draft = await approved_bundle(env)
    outsider = ctx_for(env.tenant_id, "outsider-a", role)

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, command, outsider, because())


async def test_approval_and_activation_audit_the_actor_and_role(
    connection: AsyncConnection, env: Env
) -> None:
    draft = await ready(connection, env)
    await env.move(draft, Command.ACTIVATE, env.admin, because())

    events = (
        await env.db.execute(
            select(PolicyAuditEvent.action, PolicyAuditEvent.actor, PolicyAuditEvent.details).where(
                PolicyAuditEvent.policy_bundle_id == draft.id,
                PolicyAuditEvent.action.in_(
                    [PolicyAuditAction.APPROVE, PolicyAuditAction.ACTIVATE]
                ),
            )
        )
    ).all()
    assert sorted((action, actor, details["actor_roles"]) for action, actor, details in events) == [
        (PolicyAuditAction.ACTIVATE, ADMIN, ["policy_admin"]),
        (PolicyAuditAction.APPROVE, APPROVER, ["policy_approver"]),
    ]


@pytest.mark.parametrize("state", [BundleLifecycleState.DRAFT, BundleLifecycleState.READY])
async def test_build_refuses_a_bundle_that_is_not_approved(
    env: Env, state: BundleLifecycleState
) -> None:
    draft = await env.draft()
    if state is BundleLifecycleState.READY:
        await env.review_all(draft)
        await env.move(draft, Command.SUBMIT)
        await env.move(draft, Command.APPROVE, env.approver)
        # What the worker's READY change does.
        await env.db.execute(
            update(PolicyBundle).where(PolicyBundle.id == draft.id).values(lifecycle_state=state)
        )
        await env.db.commit()

    with pytest.raises(PolicyBundleStateError, match="not in a state") as error:
        await env.move(draft, Command.BUILD, env.admin)
    assert error.value.details == {"state": state.value, "allowed": ["APPROVED"]}


async def test_a_bundle_in_review_cannot_be_edited(env: Env) -> None:
    draft = await env.draft()
    await env.move(draft, Command.SUBMIT)

    with pytest.raises(PolicyBundleStateError, match="not in a state"):
        await bundles.update_node(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            draft.clause_node,
            NodeUpdate(text="[SYNTHETIC] Changed."),
            TEST_MAX_ITEMS,
        )


async def test_an_approved_bundle_cannot_be_edited_or_reviewed(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)
    await env.move(draft, Command.SUBMIT)
    await env.move(draft, Command.APPROVE, env.approver)

    with pytest.raises(PolicyBundleStateError, match="not in a state"):
        await bundles.review_item(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            ReviewItemKind.NODE,
            draft.clause_node,
            ReviewDecision(decision="REJECTED"),
            TEST_MAX_ITEMS,
        )


async def test_an_edge_without_provenance_cannot_be_approved(env: Env) -> None:
    draft = await env.draft()
    edge = await bundles.add_edge(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        EdgeCreate(
            edge_type=PolicyEdgeType.APPLIES_TO,
            from_node_id=draft.rule_node,
            to_node_id=draft.clause_node,
        ),
        TEST_MAX_ITEMS,
    )

    with pytest.raises(PolicyItemInvalidError, match="needs a source clause or a note"):
        await bundles.review_item(
            env.db,
            env.admin,
            draft.id,
            draft.revision + 1,
            ReviewItemKind.EDGE,
            edge.id,
            ReviewDecision(decision="APPROVED"),
            TEST_MAX_ITEMS,
        )


async def test_rejected_items_do_not_block_approval(env: Env) -> None:
    draft = await env.draft()
    example = await bundles.add_node(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        NodeCreate(node_key="EX-1", node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC] x"),
        TEST_MAX_ITEMS,
    )
    draft.revision += 1
    await env.review_all(draft)
    await bundles.review_item(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        ReviewItemKind.NODE,
        example.id,
        ReviewDecision(decision="REJECTED", note="[SYNTHETIC] not needed"),
        TEST_MAX_ITEMS,
    )
    draft.revision += 1
    await env.move(draft, Command.SUBMIT)

    bundle = await env.move(draft, Command.APPROVE, env.approver)

    assert bundle.lifecycle_state is BundleLifecycleState.APPROVED


# ---- Concurrency, on committed data


async def test_two_editors_with_the_same_revision_one_wins(
    committed_tenant: SeededTenant, test_env: IntegrationEnv
) -> None:
    admin = ctx_for(committed_tenant.tenant_id, ADMIN, Role.POLICY_ADMIN)
    api = create_async_engine(test_env.test_database_url.get_secret_value(), poolclass=NullPool)

    async def session() -> AsyncSession:
        db = AsyncSession(api, sync_session_class=TenantSession, expire_on_commit=False)
        await bind_tenant(db, admin.tenant_id)
        return db

    async def add(key: str) -> str:
        async with await session() as db:
            try:
                await bundles.add_node(
                    db,
                    admin,
                    bundle_id,
                    1,
                    NodeCreate(node_key=key, node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC]"),
                    TEST_MAX_ITEMS,
                )
            except VersionConflictError:
                return "stale"
            return "saved"

    try:
        async with await session() as db:
            bundle_id = (
                await bundles.create_bundle(
                    db,
                    admin,
                    BundleCreate(
                        name="[SYNTHETIC] race", source_ref="synthetic:race", is_synthetic=True
                    ),
                    TEST_MAX_OPEN_BUNDLES,
                    TEST_MAX_ITEMS,
                )
            ).id
        results = await asyncio.gather(add("EX-1"), add("EX-2"))
        async with await session() as db:
            revision = await db.scalar(
                select(PolicyBundle.revision).where(PolicyBundle.id == bundle_id)
            )
    finally:
        await api.dispose()

    assert sorted(results) == ["saved", "stale"]
    assert revision == 2


# ---- Activation, rollback and retirement


async def approved(env: Env) -> Draft:
    draft = await env.draft()
    await env.review_all(draft)
    await env.move(draft, Command.SUBMIT)
    await env.move(draft, Command.APPROVE, env.approver)
    return draft


async def mark_ready(conn: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> None:
    # What the build worker does (tested in test_policy_build_worker.py), as the owner.
    async with as_tenant_owner(conn, tenant_id):
        await conn.execute(
            update(PolicyBundle)
            .where(PolicyBundle.id == bundle_id)
            .values(lifecycle_state=BundleLifecycleState.READY, available_at=None)
        )


async def ready(connection: AsyncConnection, env: Env) -> Draft:
    draft = await approved(env)
    await mark_ready(connection, env.tenant_id, draft.id)
    await bind_tenant(env.db, env.tenant_id)
    return draft


def because(reason: str = "[SYNTHETIC] reviewed and signed off") -> TransitionRequest:
    return TransitionRequest(reason=reason)


async def active_bundles(env: Env) -> list[UUID]:
    await bind_tenant(env.db, env.tenant_id)
    return list(
        (
            await env.db.scalars(
                select(PolicyBundle.id).where(
                    PolicyBundle.lifecycle_state == BundleLifecycleState.ACTIVE
                )
            )
        ).all()
    )


async def events(env: Env) -> list[dict[str, object]]:
    await bind_tenant(env.db, env.tenant_id)
    rows = (
        await env.db.scalars(
            select(OutboxEvent).where(OutboxEvent.event_type == bundles.POLICY_BUNDLE_EVENT)
        )
    ).all()
    return [row.payload for row in rows]


async def test_activation_replaces_the_active_bundle_and_tells_consumers(
    connection: AsyncConnection, env: Env, world: World
) -> None:
    old = world.a.policy_bundle_id
    draft = await ready(connection, env)

    bundle = await env.move(draft, Command.ACTIVATE, env.admin, because())

    assert bundle.lifecycle_state is BundleLifecycleState.ACTIVE
    assert bundle.activated_by == ADMIN
    assert bundle.activated_at is not None
    assert await active_bundles(env) == [draft.id]
    retired = await env.db.get(PolicyBundle, old, populate_existing=True)
    assert retired is not None
    assert retired.lifecycle_state is BundleLifecycleState.RETIRED
    activation = await env.db.scalar(
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.policy_bundle_id == draft.id,
            PolicyAuditEvent.action == PolicyAuditAction.ACTIVATE,
        )
    )
    assert activation is not None
    assert (activation.actor, activation.reason) == (
        ADMIN,
        "[SYNTHETIC] reviewed and signed off",
    )
    assert activation.details["previous_bundle_id"] == str(old)
    replaced = await env.db.scalar(
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.policy_bundle_id == old,
            PolicyAuditEvent.action == PolicyAuditAction.RETIRE,
        )
    )
    assert replaced is not None
    assert replaced.details["replaced_by_bundle_id"] == str(draft.id)
    assert await events(env) == [
        {
            "event_version": 1,
            "tenant_id": str(env.tenant_id),
            "policy_bundle_id": str(draft.id),
            "version": bundle.version,
            "action": "ACTIVATE",
            "audit_event_id": str(activation.id),
            "previous_bundle_id": str(old),
        }
    ]
    outbox = await env.db.scalar(
        select(OutboxEvent).where(OutboxEvent.event_type == bundles.POLICY_BUNDLE_EVENT)
    )
    assert outbox is not None
    assert outbox.job_id is None
    # Validation now reads the new bundle's rules.
    policy = await get_active_policy(env.db, env.admin)
    assert (policy.policy_bundle_id, [r.rule_id for r in policy.rules]) == (draft.id, ["REC-07"])


async def test_activation_needs_a_reason_and_the_activate_capability(
    connection: AsyncConnection, env: Env
) -> None:
    draft = await ready(connection, env)

    with pytest.raises(PolicyItemInvalidError, match="Give a reason"):
        await env.move(draft, Command.ACTIVATE, env.admin)
    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, Command.ACTIVATE, env.approver, because())


async def test_a_bundle_that_was_not_built_cannot_be_activated(env: Env) -> None:
    draft = await approved(env)

    with pytest.raises(PolicyBundleStateError, match="not in a state") as error:
        await env.move(draft, Command.ACTIVATE, env.admin, because())
    assert error.value.details == {"state": "APPROVED", "allowed": ["READY"]}


async def test_a_stale_index_blocks_activation_and_changes_nothing(
    connection: AsyncConnection, env: Env, world: World
) -> None:
    draft = await ready(connection, env)
    async with as_tenant_owner(connection, env.tenant_id):
        await connection.execute(
            insert(PolicyIndexBuild).values(
                tenant_id=env.tenant_id,
                policy_bundle_id=draft.id,
                model_id="synthetic-stale",
                dimensions=1024,
                state=IndexBuildState.READY,
                node_count=1,
                content_checksum="0" * 64,
                completed_at=datetime(2026, 11, 23, tzinfo=UTC),
            )
        )
    await bind_tenant(env.db, env.tenant_id)

    with pytest.raises(PolicyBundleHasIssuesError, match="cannot be activated") as error:
        await env.move(draft, Command.ACTIVATE, env.admin, because())

    assert error.value.details is not None
    issues = error.value.details["issues"]
    assert isinstance(issues, list)
    assert [issue["code"] for issue in issues] == ["INDEX_STALE"]
    await env.db.rollback()
    assert await active_bundles(env) == [world.a.policy_bundle_id]
    assert await events(env) == []


async def test_retiring_the_active_bundle_leaves_no_active_policy(
    connection: AsyncConnection, env: Env
) -> None:
    draft = await ready(connection, env)
    await env.move(draft, Command.ACTIVATE, env.admin, because())

    bundle = await env.move(draft, Command.RETIRE, env.admin, because("[SYNTHETIC] withdrawn"))

    assert bundle.lifecycle_state is BundleLifecycleState.RETIRED
    assert await active_bundles(env) == []
    with pytest.raises(NotFoundError, match="no active policy"):
        await get_active_policy(env.db, env.admin)
    retire = [e for e in await events(env) if e["action"] == "RETIRE"]
    assert len(retire) == 1
    assert (retire[0]["policy_bundle_id"], retire[0]["previous_bundle_id"]) == (str(draft.id), None)


async def test_rollback_returns_to_the_previously_active_bundle(
    connection: AsyncConnection, env: Env
) -> None:
    first = await ready(connection, env)
    await env.move(first, Command.ACTIVATE, env.admin, because())
    second = await ready(connection, env)
    await env.move(second, Command.ACTIVATE, env.admin, because())
    first.revision = await env.revision(first.id)

    bundle = await env.move(first, Command.ROLLBACK, env.admin, because("[SYNTHETIC] v3 wrong"))

    assert bundle.lifecycle_state is BundleLifecycleState.ACTIVE
    assert await active_bundles(env) == [first.id]
    rollback = await env.db.scalar(
        select(PolicyAuditEvent).where(
            PolicyAuditEvent.policy_bundle_id == first.id,
            PolicyAuditEvent.action == PolicyAuditAction.ROLLBACK,
        )
    )
    assert rollback is not None
    assert rollback.reason == "[SYNTHETIC] v3 wrong"
    assert rollback.details["previous_bundle_id"] == str(second.id)
    assert [e for e in await events(env) if e["action"] == "ROLLBACK"] == [
        {
            "event_version": 1,
            "tenant_id": str(env.tenant_id),
            "policy_bundle_id": str(first.id),
            "version": bundle.version,
            "action": "ROLLBACK",
            "audit_event_id": str(rollback.id),
            "previous_bundle_id": str(second.id),
        }
    ]


async def test_only_a_bundle_that_was_active_can_be_rolled_back_to(
    connection: AsyncConnection, env: Env
) -> None:
    draft = await approved(env)
    async with as_tenant_owner(connection, env.tenant_id):
        await connection.execute(
            update(PolicyBundle)
            .where(PolicyBundle.id == draft.id)
            .values(lifecycle_state=BundleLifecycleState.RETIRED)
        )
    await bind_tenant(env.db, env.tenant_id)

    with pytest.raises(PolicyBundleStateError, match="was active before"):
        await env.move(draft, Command.ROLLBACK, env.admin, because())


async def test_parallel_activations_leave_exactly_one_active_bundle(
    committed_tenant: SeededTenant, test_env: IntegrationEnv
) -> None:
    tenant_id = committed_tenant.tenant_id
    owner = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    api = create_async_engine(test_env.test_database_url.get_secret_value(), poolclass=NullPool)

    async def session() -> AsyncSession:
        db = AsyncSession(api, sync_session_class=TenantSession, expire_on_commit=False)
        await bind_tenant(db, tenant_id)
        return db

    async def activate(draft: Draft) -> None:
        async with await session() as db:
            activator = ctx_for(tenant_id, ADMIN, Role.POLICY_ADMIN)
            await bundles.transition(
                db, activator, draft.id, draft.revision, Command.ACTIVATE, because(), EN
            )

    try:
        async with owner.connect() as conn:
            clause_id = await add_source_clause(conn, tenant_id)
            await conn.commit()
        drafts = []
        for _ in range(2):
            async with await session() as db:
                drafts.append(await approved(Env(db, tenant_id, clause_id)))
        async with owner.connect() as conn:
            for draft in drafts:
                await mark_ready(conn, tenant_id, draft.id)
            await conn.commit()

        await asyncio.gather(*(activate(draft) for draft in drafts))

        async with await session() as db:
            states = (
                await db.scalars(
                    select(PolicyBundle.lifecycle_state).where(
                        PolicyBundle.id.in_([d.id for d in drafts])
                    )
                )
            ).all()
            announced = await db.scalar(
                select(func.count())
                .select_from(OutboxEvent)
                .where(OutboxEvent.event_type == bundles.POLICY_BUNDLE_EVENT)
            )
    finally:
        await api.dispose()
        await owner.dispose()

    assert sorted(states) == [BundleLifecycleState.ACTIVE, BundleLifecycleState.RETIRED]
    assert announced == 2
