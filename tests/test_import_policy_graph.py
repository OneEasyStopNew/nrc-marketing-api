"""Policy graph import (NRC-5): validation, idempotency, source reuse, the synthetic rulebook."""

from collections.abc import Callable
from dataclasses import asdict
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.orm import aliased

from app.models import (
    PolicyEdge,
    PolicyNode,
    PolicyRuleRecord,
    PolicySource,
    PolicySourceClause,
)
from app.models.enums import PolicyEdgeType, PolicyLanguage, PolicyNodeType
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
EXPECTED_COUNTS = (1, 8, 21, 9, 15)


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
        assert (result.clauses_created, result.nodes, result.rules, result.edges) == (8, 21, 9, 15)

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
            == 15
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
        assert (sources, clauses) == (1, 8)
        assert (nodes, rules, edges) == (42, 18, 30)


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


@pytest.mark.parametrize("field", ["effective_from", "effective_to"])
def test_effective_date_without_a_timezone_is_rejected(field: str) -> None:
    node = {"node_key": "CL-1", "node_type": "CLAUSE", "text": "[SYNTHETIC] clause"}

    NodeImport.model_validate({**node, field: "2026-12-01T00:00:00Z"})
    with pytest.raises(ValueError, match="timezone info"):
        NodeImport.model_validate({**node, field: "2026-12-01T00:00:00"})


def test_rule_import_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RuleImport.model_validate({"node_key": "REC-01", "surprise": 1})
