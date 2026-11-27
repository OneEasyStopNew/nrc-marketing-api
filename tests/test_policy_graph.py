"""Policy graph tables (NRC-5): isolation, grants, cross-bundle links, search, embeddings."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models import (
    PolicyBundle,
    PolicyEdge,
    PolicyIndexBuild,
    PolicyNode,
    PolicyNodeEmbedding,
    PolicyRuleRecord,
    PolicySource,
    PolicySourceClause,
)
from app.models.enums import (
    BundleLifecycleState,
    CheckType,
    Gate,
    IndexBuildState,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    ReviewState,
    Severity,
)
from app.models.policy_graph import EMBEDDING_DIMENSIONS
from tests.conftest import (
    APP_ROLE,
    WORKER_ROLE,
    World,
    add_policy_bundle,
    as_role,
    as_tenant_owner,
    bind_test_tenant,
    count_rows,
)

GRAPH_MODELS = (
    PolicySource,
    PolicySourceClause,
    PolicyNode,
    PolicyRuleRecord,
    PolicyEdge,
    PolicyNodeEmbedding,
    PolicyIndexBuild,
)
MODEL_ID = "synthetic-embedding-model"
SYNTHETIC_CHECKSUM = "a" * 64


@pytest.fixture
def seed_graph() -> bool:
    # These tests write their own graph rows into tenant A's bundle, so it starts without one.
    return False


@asynccontextmanager
async def rejected(connection: AsyncConnection, match: str) -> AsyncIterator[None]:
    # The savepoint keeps the test's transaction usable after the expected database error.
    with pytest.raises(DBAPIError, match=match):
        async with connection.begin_nested():
            yield


async def _add_source(connection: AsyncConnection, tenant_id: UUID, checksum: str) -> UUID:
    source_id = uuid4()
    await connection.execute(
        insert(PolicySource).values(
            id=source_id,
            tenant_id=tenant_id,
            title="[SYNTHETIC] source",
            source_ref="synthetic:test-fixture",
            checksum_sha256=checksum,
            language=PolicyLanguage.EN,
            is_synthetic=True,
            created_by="synthetic-fixture",
        )
    )
    return source_id


async def _add_clause(
    connection: AsyncConnection, tenant_id: UUID, source_id: UUID, ref: str = "§1.1"
) -> UUID:
    clause_id = uuid4()
    await connection.execute(
        insert(PolicySourceClause).values(
            id=clause_id,
            tenant_id=tenant_id,
            source_id=source_id,
            clause_ref=ref,
            text="[SYNTHETIC] clause wording",
            checksum_sha256=SYNTHETIC_CHECKSUM,
            language=PolicyLanguage.EN,
        )
    )
    return clause_id


async def _add_node(
    connection: AsyncConnection,
    tenant_id: UUID,
    bundle_id: UUID,
    key: str,
    *,
    body: str = "[SYNTHETIC] node wording",
    node_type: PolicyNodeType = PolicyNodeType.RULE,
    language: PolicyLanguage = PolicyLanguage.EN,
    clause_id: UUID | None = None,
) -> UUID:
    node_id = uuid4()
    await connection.execute(
        insert(PolicyNode).values(
            id=node_id,
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            node_key=key,
            node_type=node_type,
            text=body,
            language=language,
            source_clause_id=clause_id,
        )
    )
    return node_id


async def _add_rule(
    connection: AsyncConnection,
    tenant_id: UUID,
    bundle_id: UUID,
    node_id: UUID,
    rule_id: str = "TST-01",
    *,
    version: int = 1,
    phrases: list[str] | None = None,
) -> None:
    await connection.execute(
        insert(PolicyRuleRecord).values(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            node_id=node_id,
            rule_id=rule_id,
            rule_version=version,
            prohibited_phrases=phrases or [],
            title="[SYNTHETIC] rule",
            requirement="[SYNTHETIC] requirement",
            gate=Gate.POLICY,
            severity=Severity.MAJOR,
            mandatory=True,
            check_type=CheckType.SEMANTIC,
        )
    )


def _review(approved: bool) -> dict[str, object]:
    # The CHECKs tie approved to review_state, and a decided row names its reviewer.
    if not approved:
        return {"review_state": ReviewState.PROPOSED}
    return {
        "review_state": ReviewState.APPROVED,
        "reviewed_by": "synthetic-fixture",
        "reviewed_at": datetime(2026, 11, 17, tzinfo=UTC),
    }


async def _add_edge(
    connection: AsyncConnection,
    tenant_id: UUID,
    bundle_id: UUID,
    from_id: UUID,
    to_id: UUID,
    *,
    approved: bool = False,
    note: str | None = None,
    clause_id: UUID | None = None,
) -> None:
    await connection.execute(
        insert(PolicyEdge).values(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            edge_type=PolicyEdgeType.SUPPORTED_BY,
            from_node_id=from_id,
            to_node_id=to_id,
            approved=approved,
            **_review(approved),
            provenance_note=note,
            provenance_clause_id=clause_id,
        )
    )


async def _add_embedding(
    connection: AsyncConnection,
    tenant_id: UUID,
    bundle_id: UUID,
    node_id: UUID,
    vector: list[float],
) -> None:
    await connection.execute(
        insert(PolicyNodeEmbedding).values(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            node_id=node_id,
            model_id=MODEL_ID,
            embedding=vector,
            content_sha256=SYNTHETIC_CHECKSUM,
        )
    )


async def _add_build(
    connection: AsyncConnection,
    tenant_id: UUID,
    bundle_id: UUID,
    state: IndexBuildState,
    model_id: str = MODEL_ID,
    *,
    complete: bool | None = None,
) -> None:
    # A READY build is complete unless the test says otherwise.
    finished = state is IndexBuildState.READY if complete is None else complete
    await connection.execute(
        insert(PolicyIndexBuild).values(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            model_id=model_id,
            dimensions=EMBEDDING_DIMENSIONS,
            state=state,
            node_count=1 if finished else None,
            content_checksum=SYNTHETIC_CHECKSUM if finished else None,
            completed_at=datetime(2026, 11, 17, tzinfo=UTC) if finished else None,
        )
    )


@dataclass(frozen=True, slots=True)
class Graph:
    tenant_id: UUID
    bundle_id: UUID
    rule_node: UUID
    clause_node: UUID


async def _seed_graph(connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> Graph:
    # One row in every graph table, as the owner.
    async with as_tenant_owner(connection, tenant_id):
        source_id = await _add_source(connection, tenant_id, SYNTHETIC_CHECKSUM)
        clause_id = await _add_clause(connection, tenant_id, source_id)
        rule_node = await _add_node(connection, tenant_id, bundle_id, "REC-01", clause_id=clause_id)
        clause_node = await _add_node(
            connection, tenant_id, bundle_id, "CL-1", node_type=PolicyNodeType.CLAUSE
        )
        await _add_rule(connection, tenant_id, bundle_id, rule_node)
        await _add_edge(
            connection, tenant_id, bundle_id, rule_node, clause_node, approved=True, note="n"
        )
        await _add_embedding(
            connection, tenant_id, bundle_id, rule_node, [0.5] * EMBEDDING_DIMENSIONS
        )
        await _add_build(connection, tenant_id, bundle_id, IndexBuildState.READY)
    return Graph(tenant_id, bundle_id, rule_node, clause_node)


def _bundle_of(world: World) -> UUID:
    assert world.a.policy_bundle_id is not None
    return world.a.policy_bundle_id


# Tenancy and grants


async def test_other_tenant_sees_none_of_the_graph(
    connection: AsyncConnection, world: World
) -> None:
    await _seed_graph(connection, world.a.tenant_id, _bundle_of(world))

    await bind_test_tenant(connection, world.a.tenant_id)
    own = [await count_rows(connection, model) for model in GRAPH_MODELS]
    await bind_test_tenant(connection, world.b.tenant_id)
    other = [await count_rows(connection, model) for model in GRAPH_MODELS]

    assert own == [1, 1, 2, 1, 1, 1, 1]
    assert other == [0] * len(GRAPH_MODELS)


# NRC-4 lets the API write sources, nodes, rules and edges, and the worker the clauses,
# embeddings and builds (exact privileges: tests/test_worker_role.py). Writes that stay refused:
NOT_WRITABLE = [
    (APP_ROLE, PolicySourceClause),
    (APP_ROLE, PolicyNodeEmbedding),
    (APP_ROLE, PolicyIndexBuild),
    (WORKER_ROLE, PolicySource),
    (WORKER_ROLE, PolicyNode),
    (WORKER_ROLE, PolicyRuleRecord),
    (WORKER_ROLE, PolicyEdge),
]


@pytest.mark.parametrize(
    ("role", "model"), NOT_WRITABLE, ids=lambda v: v if isinstance(v, str) else v.__tablename__
)
async def test_roles_cannot_write_the_graph_tables_they_do_not_own(
    connection: AsyncConnection, world: World, role: str, model: type[Any]
) -> None:
    await as_role(connection, role)
    await bind_test_tenant(connection, world.a.tenant_id)

    # The privilege check comes before any constraint, so no valid row is needed.
    async with rejected(connection, "permission denied"):
        await connection.execute(text(f"INSERT INTO {model.__tablename__} DEFAULT VALUES"))


async def test_new_bundle_without_a_state_is_a_draft(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id = uuid4()
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            insert(PolicyBundle).values(
                id=bundle_id,
                tenant_id=world.a.tenant_id,
                version=2,
                name="[SYNTHETIC] graph test bundle",
                source_ref="synthetic:test-fixture",
                rules=[],
                is_synthetic=True,
                approved_by="synthetic-fixture",
                approved_at=datetime(2026, 11, 17, tzinfo=UTC),
            )
        )

    row = (
        await connection.execute(
            select(
                PolicyBundle.lifecycle_state, PolicyBundle.activated_at, PolicyBundle.activated_by
            ).where(PolicyBundle.id == bundle_id)
        )
    ).one()

    assert row.lifecycle_state == BundleLifecycleState.DRAFT
    assert row.activated_at is None
    assert row.activated_by is None


# Links cannot cross bundles or tenants


@dataclass(frozen=True, slots=True)
class Pair:
    tenant_id: UUID
    bundle_id: UUID
    node_a: UUID
    node_b: UUID


async def _pair(connection: AsyncConnection, world: World) -> Pair:
    bundle_id = _bundle_of(world)
    async with as_tenant_owner(connection, world.a.tenant_id):
        node_a = await _add_node(connection, world.a.tenant_id, bundle_id, "REC-01")
        node_b = await _add_node(connection, world.a.tenant_id, bundle_id, "DISC-01")
    return Pair(world.a.tenant_id, bundle_id, node_a, node_b)


async def test_edge_to_a_node_of_another_bundle_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        other_bundle = await add_policy_bundle(connection, pair.tenant_id, version=2)
        foreign = await _add_node(connection, pair.tenant_id, other_bundle, "REC-01")

        async with rejected(connection, "fk_policy_edges_to_node_policy_nodes"):
            await _add_edge(connection, pair.tenant_id, pair.bundle_id, pair.node_a, foreign)
        async with rejected(connection, "fk_policy_edges_from_node_policy_nodes"):
            await _add_edge(connection, pair.tenant_id, pair.bundle_id, foreign, pair.node_a)


async def test_edge_to_a_node_of_another_tenant_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_b = await add_policy_bundle(connection, world.b.tenant_id, version=1)
        foreign = await _add_node(connection, world.b.tenant_id, bundle_b, "REC-01")
    async with as_tenant_owner(connection, pair.tenant_id):
        async with rejected(connection, "fk_policy_edges_to_node_policy_nodes"):
            await _add_edge(connection, pair.tenant_id, pair.bundle_id, pair.node_a, foreign)
        # Pointing the row at the other tenant's bundle does not help: the bundle's own
        # nodes are all that its foreign keys accept (both endpoints fail; either is reported).
        async with rejected(connection, "fk_policy_edges_(from|to)_node_policy_nodes"):
            await _add_edge(connection, pair.tenant_id, bundle_b, foreign, pair.node_a)


async def test_node_cannot_cite_another_tenants_clause(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.b.tenant_id):
        source_b = await _add_source(connection, world.b.tenant_id, SYNTHETIC_CHECKSUM)
        clause_b = await _add_clause(connection, world.b.tenant_id, source_b)
    async with (
        as_tenant_owner(connection, world.a.tenant_id),
        rejected(connection, "fk_policy_nodes_tenant_id_policy_source_clauses"),
    ):
        await _add_node(
            connection, world.a.tenant_id, _bundle_of(world), "REC-01", clause_id=clause_b
        )


async def test_rule_cannot_attach_to_a_node_of_another_bundle(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        other_bundle = await add_policy_bundle(connection, pair.tenant_id, version=2)

        async with rejected(connection, "fk_policy_rules_tenant_id_policy_nodes"):
            await _add_rule(connection, pair.tenant_id, other_bundle, pair.node_a)


async def test_edge_to_itself_is_rejected(connection: AsyncConnection, world: World) -> None:
    pair = await _pair(connection, world)
    async with (
        as_tenant_owner(connection, pair.tenant_id),
        rejected(connection, "ck_policy_edges_no_self_edge"),
    ):
        await _add_edge(connection, pair.tenant_id, pair.bundle_id, pair.node_a, pair.node_a)


@pytest.mark.parametrize("note", [None, "", "   "])
async def test_approved_edge_needs_provenance(
    connection: AsyncConnection, world: World, note: str | None
) -> None:
    pair = await _pair(connection, world)
    async with (
        as_tenant_owner(connection, pair.tenant_id),
        rejected(connection, "ck_policy_edges_approved_has_provenance"),
    ):
        await _add_edge(
            connection,
            pair.tenant_id,
            pair.bundle_id,
            pair.node_a,
            pair.node_b,
            approved=True,
            note=note,
        )


async def test_edges_with_provenance_or_unapproved_are_stored(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        source_id = await _add_source(connection, pair.tenant_id, SYNTHETIC_CHECKSUM)
        clause_id = await _add_clause(connection, pair.tenant_id, source_id)
        await _add_edge(
            connection, pair.tenant_id, pair.bundle_id, pair.node_a, pair.node_b, note="reviewed"
        )
        await _add_edge(
            connection,
            pair.tenant_id,
            pair.bundle_id,
            pair.node_b,
            pair.node_a,
            approved=True,
            clause_id=clause_id,
        )

        assert await count_rows(connection, PolicyEdge) == 2


async def test_duplicate_edge_is_rejected(connection: AsyncConnection, world: World) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        await _add_edge(connection, pair.tenant_id, pair.bundle_id, pair.node_a, pair.node_b)

        async with rejected(connection, "uq_policy_edges_endpoints"):
            await _add_edge(connection, pair.tenant_id, pair.bundle_id, pair.node_a, pair.node_b)


async def test_node_key_is_unique_within_a_bundle_only(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        async with rejected(connection, "uq_policy_nodes_tenant_id_policy_bundle_id_node_key"):
            await _add_node(connection, pair.tenant_id, pair.bundle_id, "REC-01")

        other_bundle = await add_policy_bundle(connection, pair.tenant_id, version=2)
        await _add_node(connection, pair.tenant_id, other_bundle, "REC-01")

        assert await count_rows(connection, PolicyNode) == 3


async def test_source_checksum_and_clause_ref_are_unique(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        source_id = await _add_source(connection, world.a.tenant_id, SYNTHETIC_CHECKSUM)
        await _add_clause(connection, world.a.tenant_id, source_id)

        async with rejected(connection, "uq_policy_sources_tenant_id_checksum_sha256"):
            await _add_source(connection, world.a.tenant_id, SYNTHETIC_CHECKSUM)
        async with rejected(connection, "uq_policy_source_clauses_tenant_id_source_id_clause_ref"):
            await _add_clause(connection, world.a.tenant_id, source_id)


async def test_rule_id_must_match_the_policy_rule_pattern_and_be_unique(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        async with rejected(connection, "ck_policy_rules_rule_id_pattern"):
            await _add_rule(connection, pair.tenant_id, pair.bundle_id, pair.node_a, "rec-1")

        await _add_rule(connection, pair.tenant_id, pair.bundle_id, pair.node_a, "REC-01")
        async with rejected(connection, "uq_policy_rules_tenant_id_policy_bundle_id_rule_id"):
            await _add_rule(connection, pair.tenant_id, pair.bundle_id, pair.node_b, "REC-01")


async def test_node_language_must_be_known(connection: AsyncConnection, world: World) -> None:
    async with (
        as_tenant_owner(connection, world.a.tenant_id),
        rejected(connection, "ck_policy_nodes_language"),
    ):
        await connection.execute(
            text(
                "INSERT INTO policy_nodes (id, tenant_id, policy_bundle_id, node_key, node_type,"
                " text, language) VALUES (:id, :t, :b, 'REC-01', 'RULE', 'x', 'fr')"
            ),
            {"id": uuid4(), "t": world.a.tenant_id, "b": _bundle_of(world)},
        )


# Embeddings


async def test_embedding_round_trips_with_1024_dimensions(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    vector = [index / EMBEDDING_DIMENSIONS for index in range(EMBEDDING_DIMENSIONS)]
    async with as_tenant_owner(connection, pair.tenant_id):
        await _add_embedding(connection, pair.tenant_id, pair.bundle_id, pair.node_a, vector)

    stored = await connection.scalar(select(PolicyNodeEmbedding.embedding))

    assert stored is not None
    assert len(stored) == EMBEDDING_DIMENSIONS
    assert stored == pytest.approx(vector, rel=1e-6)


async def test_embedding_with_the_wrong_dimension_is_rejected(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with (
        as_tenant_owner(connection, pair.tenant_id),
        rejected(connection, f"expected {EMBEDDING_DIMENSIONS} dimensions, not 3"),
    ):
        await _add_embedding(
            connection, pair.tenant_id, pair.bundle_id, pair.node_a, [0.1, 0.2, 0.3]
        )


# Full-text search column


async def _matches(connection: AsyncConnection, config: str, query: str) -> list[str]:
    rows = await connection.execute(
        text(
            "SELECT node_key FROM policy_nodes"
            " WHERE search_vector @@ websearch_to_tsquery(CAST(:config AS regconfig), :query)"
            " ORDER BY node_key"
        ),
        {"config": config, "query": query},
    )
    return [row.node_key for row in rows]


async def test_search_vector_is_generated_per_node_language(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id = _bundle_of(world)
    async with as_tenant_owner(connection, world.a.tenant_id):
        await _add_node(
            connection, world.a.tenant_id, bundle_id, "EN-1", body="[SYNTHETIC] This product cures."
        )
        await _add_node(
            connection,
            world.a.tenant_id,
            bundle_id,
            "ES-1",
            body="[SYNTHETIC] Los medicamentos nuevos requieren aprobación.",
            language=PolicyLanguage.ES,
        )

    # A stemmed word matches in the node's own language only.
    assert await _matches(connection, "english", "cure") == ["EN-1"]
    assert await _matches(connection, "spanish", "medicamento") == ["ES-1"]
    assert await _matches(connection, "english", "medicamento") == []


async def test_search_vector_has_a_gin_index(connection: AsyncConnection) -> None:
    definition = await connection.scalar(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_policy_nodes_search_vector'")
    )

    assert "USING gin (search_vector)" in str(definition)


# Index builds


async def test_only_one_ready_build_per_bundle_and_model(
    connection: AsyncConnection, world: World
) -> None:
    bundle_id = _bundle_of(world)
    tenant_id = world.a.tenant_id
    async with as_tenant_owner(connection, tenant_id):
        await _add_build(connection, tenant_id, bundle_id, IndexBuildState.READY)

        async with rejected(connection, "uq_policy_index_builds_ready"):
            await _add_build(connection, tenant_id, bundle_id, IndexBuildState.READY)

        # Failed, running and other-model builds are history or parallel work, not a second READY.
        await _add_build(connection, tenant_id, bundle_id, IndexBuildState.FAILED)
        await _add_build(connection, tenant_id, bundle_id, IndexBuildState.BUILDING)
        await _add_build(connection, tenant_id, bundle_id, IndexBuildState.READY, "other-model")
        assert await count_rows(connection, PolicyIndexBuild) == 4


async def test_ready_build_must_record_what_it_indexed(
    connection: AsyncConnection, world: World
) -> None:
    # Retrieval compares node_count and content_checksum with the bundle before trusting it.
    async with (
        as_tenant_owner(connection, world.a.tenant_id),
        rejected(connection, "ck_policy_index_builds_ready_is_complete"),
    ):
        await _add_build(
            connection, world.a.tenant_id, _bundle_of(world), IndexBuildState.READY, complete=False
        )


# Rule and applicability checks


async def test_effective_window_cannot_be_inverted(
    connection: AsyncConnection, world: World
) -> None:
    async with (
        as_tenant_owner(connection, world.a.tenant_id),
        rejected(connection, "ck_policy_nodes_effective_window"),
    ):
        await connection.execute(
            insert(PolicyNode).values(
                tenant_id=world.a.tenant_id,
                policy_bundle_id=_bundle_of(world),
                node_key="REC-01",
                node_type=PolicyNodeType.RULE,
                text="[SYNTHETIC] rule",
                language=PolicyLanguage.EN,
                effective_from=datetime(2027, 1, 1, tzinfo=UTC),
                effective_to=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )


async def test_rule_version_starts_at_one(connection: AsyncConnection, world: World) -> None:
    pair = await _pair(connection, world)
    async with (
        as_tenant_owner(connection, pair.tenant_id),
        rejected(connection, "ck_policy_rules_rule_version_positive"),
    ):
        await _add_rule(connection, pair.tenant_id, pair.bundle_id, pair.node_a, version=0)


async def test_prohibited_phrases_cannot_contain_an_empty_phrase(
    connection: AsyncConnection, world: World
) -> None:
    pair = await _pair(connection, world)
    async with as_tenant_owner(connection, pair.tenant_id):
        async with rejected(connection, "ck_policy_rules_no_empty_phrase"):
            await _add_rule(
                connection, pair.tenant_id, pair.bundle_id, pair.node_a, phrases=["cure", ""]
            )

        await _add_rule(
            connection, pair.tenant_id, pair.bundle_id, pair.node_a, phrases=["cure", "proven"]
        )
        assert await count_rows(connection, PolicyRuleRecord) == 1
