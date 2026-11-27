"""Policy graph import (NRC-5): validation, idempotency, source reuse, the synthetic rulebook."""

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.orm import aliased

from app.models import (
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicySource,
    PolicySourceClause,
)
from app.models.enums import PolicyEdgeType, PolicyLanguage, PolicyNodeType, PostFormat
from scripts.import_policy_graph import (
    EdgeImport,
    NodeImport,
    PolicyGraphImport,
    PolicyGraphImportError,
    RuleImport,
    import_policy_graph,
    load_synthetic_graph,
    validate_graph,
)
from scripts.seed_demo import demo_policy_rules
from tests.conftest import (
    World,
    add_policy_bundle,
    as_tenant_owner,
    bind_test_tenant,
    count_rows,
    owner_session,
)

GRAPH_MODELS = (PolicySource, PolicySourceClause, PolicyNode, PolicyRuleRecord, PolicyEdge)
# Sources, clauses, nodes, rules, edges in the synthetic rulebook.
EXPECTED_COUNTS = (1, 9, 22, 9, 16)


@pytest.fixture
def seed_graph() -> bool:
    # These tests write their own graph rows into tenant A's bundle, so it starts without one.
    return False


def _bundle_of(world: World) -> UUID:
    assert world.a.policy_bundle_id is not None
    return world.a.policy_bundle_id


async def _counts(connection: AsyncConnection) -> tuple[int, ...]:
    return tuple([await count_rows(connection, model) for model in GRAPH_MODELS])


async def _edge_triples(
    connection: AsyncConnection, bundle_id: UUID
) -> set[tuple[PolicyEdgeType, str, str]]:
    source, target = aliased(PolicyNode), aliased(PolicyNode)
    rows = await connection.execute(
        select(PolicyEdge.edge_type, source.node_key, target.node_key)
        .join(source, source.id == PolicyEdge.from_node_id)
        .join(target, target.id == PolicyEdge.to_node_id)
        .where(PolicyEdge.policy_bundle_id == bundle_id)
    )
    return {(row[0], row[1], row[2]) for row in rows}


def _with_rule(graph: PolicyGraphImport, **update: object) -> PolicyGraphImport:
    # First rule changed, the rest kept.
    return graph.model_copy(
        update={"rules": [graph.rules[0].model_copy(update=update), *graph.rules[1:]]}
    )


def _with_node(graph: PolicyGraphImport, index: int, **update: object) -> PolicyGraphImport:
    nodes = list(graph.nodes)
    nodes[index] = nodes[index].model_copy(update=update)
    return graph.model_copy(update={"nodes": nodes})


def _index(graph: PolicyGraphImport, node_key: str) -> int:
    return next(n for n, node in enumerate(graph.nodes) if node.node_key == node_key)


def _with_edge(graph: PolicyGraphImport, **fields: object) -> PolicyGraphImport:
    edge = EdgeImport.model_validate(fields)
    return graph.model_copy(update={"edges": [*graph.edges, edge]})


# The file itself


def test_synthetic_file_parses_and_validates() -> None:
    graph = load_synthetic_graph()
    validate_graph(graph)
    assert [rule.rule_id for rule in graph.rules] == [f"REC-0{n}" for n in range(1, 10)]
    assert graph.source.is_synthetic
    assert all(node.approved for node in graph.nodes)
    assert all(edge.approved for edge in graph.edges)


# Import


async def test_import_creates_the_synthetic_rulebook(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        result = await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        assert await _counts(connection) == EXPECTED_COUNTS
        assert result.skipped is False
        assert (result.sources_created, result.sources_reused) == (1, 0)
        assert (result.clauses_created, result.nodes, result.rules, result.edges) == (9, 22, 9, 16)

        assert (
            await db.scalar(
                select(func.count()).select_from(PolicyNode).where(PolicyNode.approved.is_(False))
            )
            == 0
        )
        source = (await db.scalars(select(PolicySource))).one()
        assert source.is_synthetic
        assert source.title.startswith("[SYNTHETIC]")
        texts = await db.scalars(
            select(PolicyNode.text).where(PolicyNode.node_type != PolicyNodeType.DISCLOSURE)
        )
        assert all(text.startswith("[SYNTHETIC]") for text in texts)
        assert (
            await db.scalar(
                select(func.count())
                .select_from(PolicyEdge)
                .where(PolicyEdge.approved.is_(True), PolicyEdge.tenant_id == tenant_id)
            )
            == 16
        )


async def test_graph_rule_ids_equal_the_bundles_jsonb_rule_ids(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        rule_ids = set(await db.scalars(select(PolicyRuleRecord.rule_id)))
    assert rule_ids == {str(rule["rule_id"]) for rule in demo_policy_rules()}


async def test_second_import_writes_nothing(connection: AsyncConnection, world: World) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        again = await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        assert again.skipped is True
        assert not any(value for key, value in asdict(again).items() if key != "skipped")
        assert await _counts(connection) == EXPECTED_COUNTS


async def test_second_bundle_of_the_tenant_reuses_the_source(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id = world.a.tenant_id
    async with as_tenant_owner(connection, tenant_id):
        second_bundle = await add_policy_bundle(connection, tenant_id, version=2)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, _bundle_of(world), load_synthetic_graph())
        result = await import_policy_graph(db, tenant_id, second_bundle, load_synthetic_graph())
        assert (result.sources_created, result.sources_reused, result.clauses_created) == (0, 1, 0)
        sources, clauses, nodes, rules, edges = await _counts(connection)
        assert (sources, clauses) == (1, 9)
        assert (nodes, rules, edges) == (44, 18, 32)


async def test_other_tenant_sees_none_of_the_imported_rows(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
    await bind_test_tenant(connection, world.b.tenant_id)
    assert await _counts(connection) == (0, 0, 0, 0, 0)
    await bind_test_tenant(connection, tenant_id)
    assert await _counts(connection) == EXPECTED_COUNTS


async def test_edges_carry_the_disclosure_and_exception_chain(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
    edges = await _edge_triples(connection, bundle_id)
    assert (PolicyEdgeType.REQUIRES_DISCLOSURE, "REC-07", "DISC-01") in edges
    assert (PolicyEdgeType.REQUIRES_DISCLOSURE, "REC-07", "DISC-01-ES") in edges
    assert (PolicyEdgeType.EXCEPT_WHEN, "REC-06", "EXC-01") in edges
    assert (PolicyEdgeType.EXCEPT_WHEN, "EXC-01", "EXC-02") in edges
    assert not any(kind is PolicyEdgeType.SUPERSEDES for kind, _, _ in edges)


async def test_spanish_clause_is_found_with_the_spanish_config(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
    # "cura" stems to "cur" in Spanish; the English config would not match the es clause.
    found = await connection.scalars(
        select(PolicyNode.node_key).where(
            PolicyNode.search_vector.op("@@")(func.plainto_tsquery("spanish", "curan"))
        )
    )
    assert list(found) == ["CL-FDA-1-ES"]


async def test_prohibited_phrases_round_trip(connection: AsyncConnection, world: World) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        phrases = await db.scalar(
            select(PolicyRuleRecord.prohibited_phrases).where(PolicyRuleRecord.rule_id == "REC-04")
        )
    assert phrases == ["guarantee", "guaranteed", "guarantees"]


# Failures write nothing

_FAILURES: dict[str, tuple[Callable[[PolicyGraphImport], PolicyGraphImport], str]] = {
    "duplicate_node_key": (
        lambda g: g.model_copy(update={"nodes": [*g.nodes, g.nodes[0]]}),
        "Duplicate node_key",
    ),
    "edge_endpoint_missing": (
        lambda g: _with_edge(
            g,
            edge_type="supported_by",
            from_node_key="REC-01",
            to_node_key="CL-NOPE",
            approved=True,
            provenance_note="n",
        ),
        "endpoint that is not a node",
    ),
    "node_language_differs_from_its_clause": (
        lambda g: _with_node(g, _index(g, "CL-FDA-1-ES"), language=PolicyLanguage.EN),
        "is en but cites a es clause",
    ),
    "clause_node_text_differs_from_its_clause": (
        lambda g: _with_node(g, _index(g, "CL-FDA-1"), text="[SYNTHETIC] edited wording"),
        "text differs from clause FDA-1",
    ),
    "node_clause_ref_missing": (
        lambda g: _with_node(g, 0, clause_ref="NOPE"),
        "cites clause_ref NOPE",
    ),
    "edge_clause_ref_missing": (
        lambda g: _with_edge(
            g,
            edge_type="supported_by",
            from_node_key="REC-01",
            to_node_key="CL-FDA-2",
            approved=True,
            provenance_clause_ref="NOPE",
        ),
        "cites clause_ref NOPE",
    ),
    "rule_on_non_rule_node": (
        lambda g: g.model_copy(
            update={
                "rules": [
                    *g.rules,
                    g.rules[0].model_copy(update={"node_key": "DISC-01", "rule_id": "REC-90"}),
                ]
            }
        ),
        "which is a DISCLOSURE node",
    ),
    "rule_node_without_rule": (
        lambda g: g.model_copy(update={"rules": g.rules[1:]}),
        "RULE node without a rule row: REC-01",
    ),
    "unknown_study_fact": (
        lambda g: _with_node(g, 0, required_study_fact="not_a_fact"),
        "not a StudyFacts field",
    ),
    "edge_type_on_wrong_nodes": (
        lambda g: _with_edge(
            g,
            edge_type="requires_disclosure",
            from_node_key="REC-01",
            to_node_key="CL-FDA-1",
            approved=True,
            provenance_note="n",
        ),
        "does not allow",
    ),
    "approved_edge_without_provenance": (
        lambda g: _with_edge(
            g,
            edge_type="supported_by",
            from_node_key="REC-01",
            to_node_key="CL-FDA-2",
            approved=True,
            provenance_note="  ",
        ),
        "has no provenance",
    ),
    "blank_prohibited_phrase": (
        lambda g: _with_rule(g, prohibited_phrases=["proven", "   "]),
        "blank prohibited phrase",
    ),
    # policy_rules.prohibited_phrases holds varchar(200).
    "prohibited_phrase_over_its_column": (
        lambda g: _with_rule(g, prohibited_phrases=["x" * 201]),
        "prohibited phrase over 200 characters",
    ),
    "rule_ids_differ_from_bundle": (
        lambda g: _with_rule(g, rule_id="REC-90"),
        "differ from the bundle's",
    ),
}


@pytest.mark.parametrize("case", sorted(_FAILURES))
async def test_inconsistent_graph_is_rejected_and_nothing_is_written(
    connection: AsyncConnection, world: World, case: str
) -> None:
    mutate, message = _FAILURES[case]
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        with pytest.raises(PolicyGraphImportError, match=message):
            await import_policy_graph(db, tenant_id, bundle_id, mutate(load_synthetic_graph()))
        assert await _counts(connection) == (0, 0, 0, 0, 0)


async def test_import_into_an_unknown_bundle_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    async with owner_session(connection, world.a.tenant_id) as db:
        with pytest.raises(PolicyGraphImportError, match="not found for this tenant"):
            await import_policy_graph(db, world.a.tenant_id, uuid4(), load_synthetic_graph())
        assert await _counts(connection) == (0, 0, 0, 0, 0)


async def test_import_into_another_tenants_bundle_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    # Bound to tenant A, tenant B's bundle is invisible (row-level security), so it is not found.
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_b = await add_policy_bundle(connection, world.b.tenant_id, version=1)
    async with owner_session(connection, world.a.tenant_id) as db:
        with pytest.raises(PolicyGraphImportError, match="not found for this tenant"):
            await import_policy_graph(db, world.a.tenant_id, bundle_b, load_synthetic_graph())
        assert await _counts(connection) == (0, 0, 0, 0, 0)


async def test_different_graph_for_a_bundle_that_has_one_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    changed = _with_rule(load_synthetic_graph(), prohibited_phrases=["cure"])
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        with pytest.raises(PolicyGraphImportError, match="already has a different graph"):
            await import_policy_graph(db, tenant_id, bundle_id, changed)
        assert await _counts(connection) == EXPECTED_COUNTS


def _change_edge(graph: PolicyGraphImport) -> PolicyGraphImport:
    first, *rest = graph.edges
    return graph.model_copy(
        update={"edges": [first.model_copy(update={"approved": not first.approved}), *rest]}
    )


def _change_node(graph: PolicyGraphImport) -> PolicyGraphImport:
    first, *rest = graph.nodes
    return graph.model_copy(
        update={
            "nodes": [first.model_copy(update={"post_formats": [PostFormat.FEED_IMAGE]}), *rest]
        }
    )


def _change_rule(graph: PolicyGraphImport) -> PolicyGraphImport:
    first, *rest = graph.rules
    title = first.title + " (changed)"
    return graph.model_copy(update={"rules": [first.model_copy(update={"title": title}), *rest]})


def _change_effective_date(graph: PolicyGraphImport) -> PolicyGraphImport:
    first, *rest = graph.nodes
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return graph.model_copy(
        update={"nodes": [first.model_copy(update={"effective_from": start}), *rest]}
    )


@pytest.mark.parametrize(
    "change", [_change_edge, _change_node, _change_rule, _change_effective_date]
)
async def test_a_file_that_changes_any_imported_field_is_rejected_not_skipped(
    connection: AsyncConnection,
    world: World,
    change: Callable[[PolicyGraphImport], PolicyGraphImport],
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        with pytest.raises(PolicyGraphImportError, match="already has a different graph"):
            await import_policy_graph(db, tenant_id, bundle_id, change(load_synthetic_graph()))


async def test_same_instant_in_another_offset_is_the_same_file(
    connection: AsyncConnection, world: World
) -> None:
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    utc = _change_effective_date(load_synthetic_graph())
    first, *rest = utc.nodes
    assert first.effective_from is not None
    shifted = first.effective_from.astimezone(timezone(timedelta(hours=5, minutes=30)))
    other_offset = utc.model_copy(
        update={"nodes": [first.model_copy(update={"effective_from": shifted}), *rest]}
    )
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, utc)
        assert (await import_policy_graph(db, tenant_id, bundle_id, other_offset)).skipped


@pytest.mark.parametrize("field", ["effective_from", "effective_to"])
def test_effective_date_without_a_timezone_is_rejected(field: str) -> None:
    node = {"node_key": "CL-1", "node_type": "CLAUSE", "text": "[SYNTHETIC] clause"}

    NodeImport.model_validate({**node, field: "2026-12-01T00:00:00Z"})
    with pytest.raises(ValueError, match="timezone info"):
        NodeImport.model_validate({**node, field: "2026-12-01T00:00:00"})


@pytest.mark.parametrize(
    ("field", "valid", "typo"),
    [
        ("channels", "facebook", "Facebook"),
        ("post_formats", "feed_image", "feed-image"),
        ("input_kinds", "brief", "briefs"),
        ("languages", "es", "spanish"),
    ],
)
def test_condition_value_outside_its_enum_is_rejected(field: str, valid: str, typo: str) -> None:
    # A typo must stop the import, not import a rule that never applies to any post.
    node = {"node_key": "R-1", "node_type": "RULE", "text": "[SYNTHETIC] rule"}

    assert getattr(NodeImport.model_validate({**node, field: [valid]}), field) == [valid]
    # The typo is the second item, so it is reported at index 1.
    with pytest.raises(ValueError, match=rf"{field}\.1\s+Input should be"):
        NodeImport.model_validate({**node, field: [valid, typo]})


async def test_stored_value_outside_the_enums_is_a_different_graph(
    connection: AsyncConnection, world: World
) -> None:
    # Rows imported before the enums were enforced may hold such values; a re-run must still get
    # the import's own refusal, not a validation error from reading the stored side.
    tenant_id, bundle_id = world.a.tenant_id, _bundle_of(world)
    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())
        await db.execute(
            update(PolicyNode)
            .where(PolicyNode.policy_bundle_id == bundle_id)
            .values(channels=["instagram"])
        )
        with pytest.raises(PolicyGraphImportError, match="already has a different graph"):
            await import_policy_graph(db, tenant_id, bundle_id, load_synthetic_graph())


def test_a_typo_in_a_graph_file_stops_the_whole_import() -> None:
    graph = load_synthetic_graph().model_dump(mode="json")
    graph["nodes"][0]["channels"] = ["Facebook"]

    with pytest.raises(ValueError, match=r"nodes\.0\.channels\.0\s+Input should be 'facebook'"):
        PolicyGraphImport.model_validate(graph)


def test_rule_import_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RuleImport.model_validate({"node_key": "REC-01", "surprise": 1})
