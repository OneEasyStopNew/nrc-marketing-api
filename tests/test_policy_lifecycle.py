"""Policy bundle lifecycle: drafts, edits, item review, submit, approve, reject (NRC-4)."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.auth.access import Role, TenantContext, capabilities_for
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
    CheckType,
    Gate,
    PolicyAuditAction,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
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
from app.services.policy_bundles import Command, ReviewItemKind
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
        return await bundles.create_bundle(self.db, self.admin, body)

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
    )

    assert node.jurisdictions == ["US"]
    assert node.review_state is ReviewState.PROPOSED
    assert node.reviewed_by is None
    assert node.approved is False


async def test_editing_a_rule_sends_its_node_back_to_review(env: Env) -> None:
    draft = await env.draft()
    await env.review_all(draft)

    rule = await bundles.update_rule(
        env.db, env.admin, draft.id, draft.revision, draft.rule, RuleUpdate(severity=Severity.MAJOR)
    )

    assert rule.severity is Severity.MAJOR
    node = await env.db.get(PolicyNode, draft.rule_node)
    assert node is not None
    assert node.review_state is ReviewState.PROPOSED


async def test_a_required_field_cannot_be_cleared(env: Env) -> None:
    draft = await env.draft()

    with pytest.raises(PolicyItemInvalidError, match="cannot be empty") as error:
        await bundles.update_node(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            draft.clause_node,
            NodeUpdate.model_validate({"text": None}),
        )
    assert error.value.details == {"fields": ["text"]}


async def test_a_node_key_is_used_once_per_bundle(env: Env) -> None:
    draft = await env.draft()

    with pytest.raises(PolicyItemInvalidError, match="already uses this key"):
        await bundles.add_node(
            env.db,
            env.admin,
            draft.id,
            draft.revision,
            NodeCreate(node_key="REC-07", node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC] x"),
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
    assert event.details == {"revision": draft.revision, "rules": 1}


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


async def test_approvers_cannot_submit_or_review_items(env: Env) -> None:
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
        )


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

    bundle = await env.move(draft, Command.BUILD, env.approver)

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
        APPROVER,
        BundleLifecycleState.APPROVED,
        BundleLifecycleState.APPROVED,
    )


async def test_policy_admins_cannot_run_build(env: Env) -> None:
    draft = await approved_bundle(env)

    with pytest.raises(PermissionDeniedError, match="cannot do this"):
        await env.move(draft, Command.BUILD, env.admin)


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
        await env.move(draft, Command.BUILD, env.approver)
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
        )


async def test_rejected_items_do_not_block_approval(env: Env) -> None:
    draft = await env.draft()
    example = await bundles.add_node(
        env.db,
        env.admin,
        draft.id,
        draft.revision,
        NodeCreate(node_key="EX-1", node_type=PolicyNodeType.EXAMPLE, text="[SYNTHETIC] x"),
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
