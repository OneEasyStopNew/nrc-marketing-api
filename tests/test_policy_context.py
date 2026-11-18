"""Policy context retrieval (NRC-5): rule selection, the graph walk, exceptions, the cache."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.core.errors import PolicyContextUnavailableError
from app.db.session import bind_tenant
from app.models import ContextManifestRecord, StudySnapshot
from app.models.enums import (
    BundleLifecycleState,
    Channel,
    Language,
    PolicyLanguage,
    PolicyNodeType,
    RecruitmentStatus,
    ValidationInputKind,
)
from app.services.policy_context import (
    Applicability,
    AuroraPolicyContextProvider,
    ContextManifest,
    ContextRequest,
    ContextScope,
    MissingContextCode,
    StoredManifest,
)
from scripts.import_policy_graph import (
    ClauseImport,
    EdgeImport,
    NodeImport,
    PolicyGraphImport,
    RuleImport,
    SourceImport,
    import_policy_graph,
    load_synthetic_graph,
)
from scripts.seed_demo import SYNTHETIC_FACTS
from tests.conftest import (
    APP_ROLE,
    World,
    add_policy_bundle,
    app_session,
    as_role,
    as_tenant_owner,
    bind_test_tenant,
    count_rows,
    owner_session,
)

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
TEXT_RULES = ("REC-01", "REC-02", "REC-03", "REC-04", "REC-05", "REC-07", "REC-08")


def _scope(
    kind: ValidationInputKind = ValidationInputKind.TEXT,
    *,
    audience: str | None = None,
    at: datetime = NOW,
) -> ContextScope:
    return ContextScope(
        input_kind=kind,
        channel=Channel.FACEBOOK,
        language=PolicyLanguage.EN,
        audience=audience,
        evaluated_at=at,
    )


def _provider(**limits: int) -> AuroraPolicyContextProvider:
    settings = get_settings().model_copy(
        update={f"policy_retrieval_{name}": value for name, value in limits.items()}
    )
    return AuroraPolicyContextProvider(settings)


async def _build(
    connection: AsyncConnection,
    tenant_id: UUID,
    bundle_id: UUID,
    snapshot_id: UUID,
    scope: ContextScope | None = None,
    provider: AuroraPolicyContextProvider | None = None,
) -> StoredManifest:
    # As the API role, like the validation request that will call it.
    request = ContextRequest(
        policy_bundle_id=bundle_id, study_snapshot_id=snapshot_id, scope=scope or _scope()
    )
    async with app_session(connection) as db:
        await bind_tenant(db, tenant_id)
        stored = await (provider or _provider()).build(db, tenant_id, request, actor="tester")
        await db.commit()
    return stored


async def _synthetic(connection: AsyncConnection, world: World) -> tuple[UUID, UUID]:
    # Tenant A's seeded bundle with the synthetic rulebook imported.
    assert world.a.policy_bundle_id is not None
    async with owner_session(connection, world.a.tenant_id) as db:
        await import_policy_graph(
            db, world.a.tenant_id, world.a.policy_bundle_id, load_synthetic_graph()
        )
    return world.a.policy_bundle_id, world.a.snapshot_id


async def _custom(
    connection: AsyncConnection,
    world: World,
    graph: PolicyGraphImport,
    version: int = 2,
    state: BundleLifecycleState = BundleLifecycleState.ACTIVE,
) -> UUID:
    # A second bundle (ACTIVE unless said otherwise, no JSONB rules) holding a small test graph.
    async with as_tenant_owner(connection, world.a.tenant_id):
        bundle_id = await add_policy_bundle(connection, world.a.tenant_id, version, state)
    async with owner_session(connection, world.a.tenant_id) as db:
        await import_policy_graph(db, world.a.tenant_id, bundle_id, graph)
    return bundle_id


def _graph(
    nodes: Sequence[NodeImport], rules: Sequence[RuleImport], edges: Sequence[EdgeImport]
) -> PolicyGraphImport:
    clause = ClauseImport(clause_ref="T-1", text="[SYNTHETIC] test clause")
    return PolicyGraphImport(
        source=SourceImport(
            title="[SYNTHETIC] test source",
            source_ref="synthetic:test",
            is_synthetic=True,
            clauses=[clause],
        ),
        nodes=[
            NodeImport(
                node_key="CL-T",
                node_type=PolicyNodeType.CLAUSE,
                text=clause.text,
                clause_ref="T-1",
                approved=True,
            ),
            *nodes,
        ],
        rules=list(rules),
        edges=list(edges),
    )


def _rule_node(key: str, **conditions: Any) -> NodeImport:
    return NodeImport(
        node_key=key,
        node_type=PolicyNodeType.RULE,
        text=f"[SYNTHETIC] rule {key}",
        clause_ref="T-1",
        approved=True,
        **conditions,
    )


def _exception_node(key: str, **conditions: Any) -> NodeImport:
    return NodeImport(
        node_key=key,
        node_type=PolicyNodeType.EXCEPTION,
        text=f"[SYNTHETIC] exception {key}",
        approved=True,
        **conditions,
    )


def _rule(key: str, *, mandatory: bool = True) -> RuleImport:
    return RuleImport(
        node_key=key,
        rule_id=key,
        rule_version=1,
        title=f"[SYNTHETIC] {key}",
        requirement="[SYNTHETIC] requirement",
        gate="policy",
        severity="major",
        mandatory=mandatory,
        check_type="semantic",
    )


def _edge(kind: str, source: str, target: str, *, approved: bool = True) -> EdgeImport:
    return EdgeImport(
        edge_type=kind,
        from_node_key=source,
        to_node_key=target,
        approved=approved,
        provenance_note="[SYNTHETIC] reviewed" if approved else None,
    )


def _ids(rules: Sequence[Any]) -> list[str]:
    return [rule.rule_id for rule in rules]


def _codes(manifest: ContextManifest) -> list[tuple[MissingContextCode, str | None]]:
    return [(item.code, item.rule_id) for item in manifest.missing_context]


# Rule selection (synthetic rulebook)


async def test_text_check_selects_rules_by_their_conditions(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)

    manifest = (await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)).manifest

    assert _ids(manifest.required_rules) == list(TEXT_RULES)
    assert _ids(manifest.optional_rules) == ["REC-06"]
    assert "REC-09" not in _ids(manifest.required_rules)  # image only
    assert manifest.missing_context == ()
    assert manifest.policy_bundle.is_synthetic


async def test_brief_check_leaves_out_caption_only_rules(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)

    manifest = (
        await _build(
            connection, world.a.tenant_id, bundle_id, snapshot_id, _scope(ValidationInputKind.BRIEF)
        )
    ).manifest

    assert "REC-07" not in _ids(manifest.required_rules)
    assert manifest.optional_rules == ()


async def test_required_disclosure_is_selected_whatever_the_post_says(
    connection: AsyncConnection, world: World
) -> None:
    # AC1: the disclosure comes from the rule's conditions and edges, not from searching the text.
    bundle_id, snapshot_id = await _synthetic(connection, world)

    manifest = (await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)).manifest

    assert "REC-07" in _ids(manifest.required_rules)
    paths = {item.node_key: item.paths for item in manifest.evidence}
    assert paths["DISC-01"] == (("REC-07", "requires_disclosure", "DISC-01"),)
    assert paths["DISC-01-ES"] == (("REC-07", "requires_disclosure", "DISC-01-ES"),)
    assert paths["CL-FDA-1-ES"] == (("REC-01", "supported_by", "CL-FDA-1-ES"),)


async def test_evidence_shared_by_two_rules_keeps_both_paths(
    connection: AsyncConnection, world: World
) -> None:
    # Image check: REC-04 and REC-09 both rest on the same clause.
    bundle_id, snapshot_id = await _synthetic(connection, world)

    manifest = (
        await _build(
            connection, world.a.tenant_id, bundle_id, snapshot_id, _scope(ValidationInputKind.IMAGE)
        )
    ).manifest

    shared = next(item for item in manifest.evidence if item.node_key == "CL-FDA-4")
    assert shared.paths == (
        ("REC-04", "supported_by", "CL-FDA-4"),
        ("REC-09", "supported_by", "CL-FDA-4"),
    )


# Exceptions (AC2)


async def test_exception_needing_a_missing_study_fact_does_not_apply(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)

    manifest = (await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)).manifest

    exceptions = {item.node_key: item for item in manifest.exceptions}
    assert exceptions["EXC-01"].applicable is Applicability.FALSE
    assert exceptions["EXC-01"].reasons == ("STUDY_FACT_MISSING:compensation_statement",)
    assert exceptions["EXC-02"].path == ("REC-06", "except_when", "EXC-01", "except_when", "EXC-02")
    assert {item.for_rule_id for item in manifest.exceptions} == {"REC-06"}
    # The rule stays selected either way.
    assert _ids(manifest.optional_rules) == ["REC-06"]


async def test_exception_applies_when_the_snapshot_has_the_fact(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, _ = await _synthetic(connection, world)
    snapshot_id = uuid4()
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            insert(StudySnapshot).values(
                id=snapshot_id,
                tenant_id=world.a.tenant_id,
                study_id=world.a.study_id,
                version=2,
                recruitment_status=RecruitmentStatus.RECRUITING,
                language=Language.EN,
                facts={**SYNTHETIC_FACTS, "compensation_statement": "[SYNTHETIC] $50 per visit"},
                source_ref="synthetic:test-fixture",
                approved_by="synthetic-fixture",
                approved_at=NOW,
                is_synthetic=True,
            )
        )

    manifest = (await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)).manifest

    exceptions = {item.node_key: item for item in manifest.exceptions}
    assert exceptions["EXC-01"].applicable is Applicability.TRUE
    assert _ids(manifest.optional_rules) == ["REC-06"]


async def test_exception_without_an_approved_path_or_matching_conditions_cannot_apply(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [
            _rule_node("R-1"),
            _exception_node("X-UNAPPROVED-EDGE"),
            _exception_node("X-OTHER-CHANNEL", channels=["instagram"]),
        ],
        [_rule("R-1")],
        [
            _edge("except_when", "R-1", "X-UNAPPROVED-EDGE", approved=False),
            _edge("except_when", "R-1", "X-OTHER-CHANNEL"),
        ],
    )
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert [item.node_key for item in manifest.exceptions] == ["X-OTHER-CHANNEL"]
    assert manifest.exceptions[0].applicable is Applicability.FALSE
    assert manifest.exceptions[0].reasons == ("CHANNEL_NOT_MATCHED",)
    assert _ids(manifest.required_rules) == ["R-1"]


async def test_exception_shared_by_two_rules_is_listed_for_both(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [_rule_node("R-1"), _rule_node("R-2"), _exception_node("X-1")],
        [_rule("R-1"), _rule("R-2")],
        [_edge("except_when", "R-1", "X-1"), _edge("except_when", "R-2", "X-1")],
    )
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert [(e.for_rule_id, e.node_key) for e in manifest.exceptions] == [
        ("R-1", "X-1"),
        ("R-2", "X-1"),
    ]


# Bounded, safe failure (AC3, AC4)


async def test_cyclic_exceptions_are_walked_once_each(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [_rule_node("R-1"), _exception_node("X-1"), _exception_node("X-2")],
        [_rule("R-1")],
        [
            _edge("except_when", "R-1", "X-1"),
            _edge("except_when", "X-1", "X-2"),
            _edge("except_when", "X-2", "X-1"),
        ],
    )
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert [item.node_key for item in manifest.exceptions] == ["X-1", "X-2"]
    assert manifest.missing_context == ()


async def test_hitting_the_hop_limit_is_recorded(connection: AsyncConnection, world: World) -> None:
    # REC-06 -> EXC-01 -> EXC-02 needs two hops; with one, EXC-02 is left out and that's said.
    bundle_id, snapshot_id = await _synthetic(connection, world)

    manifest = (
        await _build(
            connection, world.a.tenant_id, bundle_id, snapshot_id, provider=_provider(max_hops=1)
        )
    ).manifest

    assert (MissingContextCode.TRAVERSAL_TRUNCATED, None) in _codes(manifest)
    assert "EXC-02" not in {item.node_key for item in manifest.exceptions}
    assert _ids(manifest.required_rules) == list(TEXT_RULES)


async def test_hitting_the_node_limit_is_recorded(
    connection: AsyncConnection, world: World
) -> None:
    examples = [
        NodeImport(
            node_key=f"EX-{n:02}",
            node_type=PolicyNodeType.EXAMPLE,
            text=f"[SYNTHETIC] example {n}",
            approved=True,
        )
        for n in range(12)
    ]
    graph = _graph(
        [_rule_node("R-1"), *examples],
        [_rule("R-1")],
        [_edge("supported_by", "R-1", node.node_key) for node in examples],
    )
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(
            connection,
            world.a.tenant_id,
            bundle_id,
            world.a.snapshot_id,
            provider=_provider(max_nodes=10),
        )
    ).manifest

    assert len(manifest.evidence) == 10
    assert (MissingContextCode.TRAVERSAL_TRUNCATED, None) in _codes(manifest)
    assert _ids(manifest.required_rules) == ["R-1"]


async def test_rule_restricted_by_an_unknown_audience_is_kept_and_flagged(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [_rule_node("R-ADULTS", audiences=["adults"])],
        [_rule("R-ADULTS")],
        [],
    )
    bundle_id = await _custom(connection, world, graph)
    tenant, snapshot = world.a.tenant_id, world.a.snapshot_id

    unknown = (await _build(connection, tenant, bundle_id, snapshot)).manifest
    other = (
        await _build(connection, tenant, bundle_id, snapshot, _scope(audience="teens"))
    ).manifest
    known = (
        await _build(connection, tenant, bundle_id, snapshot, _scope(audience="adults"))
    ).manifest

    assert _ids(unknown.required_rules) == ["R-ADULTS"]
    assert unknown.required_rules[0].unresolved == ("audience",)
    assert _codes(unknown) == [(MissingContextCode.APPLICABILITY_UNRESOLVED, "R-ADULTS")]
    assert other.required_rules == ()
    assert _ids(known.required_rules) == ["R-ADULTS"]
    assert known.missing_context == ()


async def test_rules_outside_their_effective_dates_are_not_selected(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [
            _rule_node("R-EXPIRED", effective_to=datetime(2026, 1, 1, tzinfo=UTC)),
            _rule_node("R-FUTURE", effective_from=datetime(2027, 1, 1, tzinfo=UTC)),
            _rule_node("R-NOW"),
        ],
        [_rule("R-EXPIRED"), _rule("R-FUTURE"), _rule("R-NOW")],
        [],
    )
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest
    later = (
        await _build(
            connection,
            world.a.tenant_id,
            bundle_id,
            world.a.snapshot_id,
            _scope(at=datetime(2027, 6, 1, tzinfo=UTC)),
        )
    ).manifest

    assert _ids(manifest.required_rules) == ["R-NOW"]
    assert _ids(later.required_rules) == ["R-FUTURE", "R-NOW"]


async def test_rule_without_any_provenance_is_flagged(
    connection: AsyncConnection, world: World
) -> None:
    bare = NodeImport(
        node_key="R-BARE", node_type=PolicyNodeType.RULE, text="[SYNTHETIC] r", approved=True
    )
    bundle_id = await _custom(connection, world, _graph([bare], [_rule("R-BARE")], []))

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert _codes(manifest) == [(MissingContextCode.SOURCE_MISSING, "R-BARE")]


async def test_unapproved_rule_node_is_never_selected(
    connection: AsyncConnection, world: World
) -> None:
    proposed = _rule_node("R-PROPOSED").model_copy(update={"approved": False})
    graph = _graph([_rule_node("R-1"), proposed], [_rule("R-1"), _rule("R-PROPOSED")], [])
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert _ids(manifest.required_rules) == ["R-1"]


# Supersedes


async def test_newer_rule_supersedes_an_older_one(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [_rule_node("R-NEW"), _rule_node("R-OLD")],
        [_rule("R-NEW"), _rule("R-OLD")],
        [_edge("supersedes", "R-NEW", "R-OLD")],
    )
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert _ids(manifest.required_rules) == ["R-NEW"]
    assert [(s.rule_id, s.superseded_by) for s in manifest.superseded] == [("R-OLD", "R-NEW")]


@pytest.mark.parametrize(
    ("rules", "edges", "conflicted"),
    [
        # An optional rule never replaces a mandatory one.
        (
            [_rule("R-NEW", mandatory=False), _rule("R-OLD")],
            [_edge("supersedes", "R-NEW", "R-OLD")],
            "R-OLD",
        ),
        # Two rules superseding each other: keep both.
        (
            [_rule("R-NEW"), _rule("R-OLD")],
            [_edge("supersedes", "R-NEW", "R-OLD"), _edge("supersedes", "R-OLD", "R-NEW")],
            "R-NEW",
        ),
    ],
)
async def test_ambiguous_supersedes_is_a_conflict_and_keeps_the_rules(
    connection: AsyncConnection,
    world: World,
    rules: list[RuleImport],
    edges: list[EdgeImport],
    conflicted: str,
) -> None:
    graph = _graph([_rule_node("R-NEW"), _rule_node("R-OLD")], rules, edges)
    bundle_id = await _custom(connection, world, graph)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    kept = _ids(manifest.required_rules) + _ids(manifest.optional_rules)
    assert sorted(kept) == ["R-NEW", "R-OLD"]
    assert manifest.superseded == ()
    assert (MissingContextCode.CONFLICT, conflicted) in _codes(manifest)


# Unavailable context never becomes a result (AC4)


async def test_bundle_without_a_graph_is_unavailable(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    with pytest.raises(PolicyContextUnavailableError, match="could not be built") as raised:
        await _build(connection, world.a.tenant_id, world.a.policy_bundle_id, world.a.snapshot_id)
    assert raised.value.details == {"reason": "GRAPH_MISSING"}


async def test_retired_bundle_still_builds_for_reproduction(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph([_rule_node("R-1")], [_rule("R-1")], [])
    bundle_id = await _custom(connection, world, graph, state=BundleLifecycleState.RETIRED)

    manifest = (
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    ).manifest

    assert _ids(manifest.required_rules) == ["R-1"]


@pytest.mark.parametrize("state", [BundleLifecycleState.READY, BundleLifecycleState.APPROVED])
async def test_bundle_that_was_never_activated_is_not_usable(
    connection: AsyncConnection, world: World, state: BundleLifecycleState
) -> None:
    graph = _graph([_rule_node("R-1")], [_rule("R-1")], [])
    bundle_id = await _custom(connection, world, graph, state=state)

    with pytest.raises(PolicyContextUnavailableError, match="could not be built") as raised:
        await _build(connection, world.a.tenant_id, bundle_id, world.a.snapshot_id)
    assert raised.value.details == {"reason": "BUNDLE_NOT_USABLE", "lifecycle_state": state}


async def test_draft_bundle_is_not_usable(connection: AsyncConnection, world: World) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        draft = await add_policy_bundle(
            connection, world.a.tenant_id, 2, BundleLifecycleState.DRAFT
        )
    with pytest.raises(PolicyContextUnavailableError, match="could not be built") as raised:
        await _build(connection, world.a.tenant_id, draft, world.a.snapshot_id)
    assert raised.value.details == {"reason": "BUNDLE_NOT_USABLE", "lifecycle_state": "DRAFT"}


async def test_another_tenants_bundle_or_snapshot_is_not_found(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_b = await add_policy_bundle(connection, world.b.tenant_id, 1)

    with pytest.raises(PolicyContextUnavailableError, match="could not be built") as bundle:
        await _build(connection, world.a.tenant_id, bundle_b, snapshot_id)
    with pytest.raises(PolicyContextUnavailableError, match="could not be built") as snapshot:
        await _build(connection, world.a.tenant_id, bundle_id, world.b.snapshot_id)

    assert bundle.value.details == {"reason": "BUNDLE_NOT_FOUND"}
    assert snapshot.value.details == {"reason": "SNAPSHOT_NOT_FOUND"}


# Stored manifests and the cache


async def test_same_request_reuses_the_stored_manifest(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)

    first = await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)
    second = await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)

    assert (first.reused, second.reused) == (False, True)
    assert second.id == first.id
    assert second.manifest == first.manifest
    await bind_test_tenant(connection, world.a.tenant_id)
    assert await count_rows(connection, ContextManifestRecord) == 1


async def test_any_changed_dependency_builds_a_new_manifest(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)
    tenant = world.a.tenant_id
    base = await _build(connection, tenant, bundle_id, snapshot_id)

    other_kind = await _build(
        connection, tenant, bundle_id, snapshot_id, _scope(ValidationInputKind.BRIEF)
    )
    other_limits = await _build(
        connection, tenant, bundle_id, snapshot_id, provider=_provider(max_hops=2)
    )
    same_inputs_later = await _build(
        connection, tenant, bundle_id, snapshot_id, _scope(at=datetime(2026, 12, 1, tzinfo=UTC))
    )

    assert len({base.id, other_kind.id, other_limits.id}) == 3
    # No node in this bundle is dated, so a later moment decides nothing new: same manifest.
    assert same_inputs_later.id == base.id


async def test_changed_graph_content_builds_a_new_manifest(
    connection: AsyncConnection, world: World
) -> None:
    # The key hashes what was loaded, so even an edit to an "immutable" bundle isn't served stale.
    bundle_id, snapshot_id = await _synthetic(connection, world)
    before = await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            text(
                "UPDATE policy_rules SET severity = 'minor' WHERE rule_id = 'REC-03'"
                " AND policy_bundle_id = :b"
            ),
            {"b": bundle_id},
        )

    after = await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)

    assert after.id != before.id
    rec03 = next(rule for rule in after.manifest.required_rules if rule.rule_id == "REC-03")
    assert rec03.severity == "minor"


async def test_crossing_an_effective_date_builds_a_new_manifest(
    connection: AsyncConnection, world: World
) -> None:
    graph = _graph(
        [_rule_node("R-FROM-2027", effective_from=datetime(2027, 1, 1, tzinfo=UTC))],
        [_rule("R-FROM-2027")],
        [],
    )
    bundle_id = await _custom(connection, world, graph)
    tenant, snapshot = world.a.tenant_id, world.a.snapshot_id

    before = await _build(connection, tenant, bundle_id, snapshot)
    after = await _build(
        connection, tenant, bundle_id, snapshot, _scope(at=datetime(2027, 2, 1, tzinfo=UTC))
    )

    assert before.id != after.id
    assert before.manifest.required_rules == ()
    assert _ids(after.manifest.required_rules) == ["R-FROM-2027"]


async def test_manifests_are_tenant_scoped_and_immutable(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id, snapshot_id = await _synthetic(connection, world)
    stored = await _build(connection, world.a.tenant_id, bundle_id, snapshot_id)

    await as_role(connection, APP_ROLE)
    await bind_test_tenant(connection, world.b.tenant_id)
    assert await count_rows(connection, ContextManifestRecord) == 0

    await bind_test_tenant(connection, world.a.tenant_id)
    with pytest.raises(DBAPIError, match="permission denied"):
        async with connection.begin_nested():
            await connection.execute(
                text("UPDATE context_manifests SET created_by = 'x' WHERE id = :id"),
                {"id": stored.id},
            )
    row = await connection.scalar(
        select(ContextManifestRecord.manifest).where(ContextManifestRecord.id == stored.id)
    )
    assert ContextManifest.model_validate(row) == stored.manifest
