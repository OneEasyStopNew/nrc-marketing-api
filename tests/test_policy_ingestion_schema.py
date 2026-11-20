"""NRC-4 schema: review state, single ACTIVE, frozen graph, wordings, audit, queues and grants.

Everything here is checked in the database, as the roles that matter (owner, API, worker). Service
and API behaviour is tested with the code that uses these tables.
"""

import hashlib
import importlib.util
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models.enums import BundleLifecycleState
from scripts.import_policy_graph import import_policy_graph, load_synthetic_graph
from tests.conftest import (
    APP_ROLE,
    WORKER_ROLE,
    SeededTenant,
    World,
    add_policy_bundle,
    as_role,
    as_tenant_owner,
    bind_test_tenant,
    owner_session,
)

ROOT = Path(__file__).resolve().parents[1]
STATES = tuple(BundleLifecycleState)
# Bundle states in which the API and worker roles may still write graph rows.
OPEN_STATES = (BundleLifecycleState.DRAFT, BundleLifecycleState.REVIEW)
FROZEN_STATES = tuple(state for state in STATES if state not in OPEN_STATES)
APPROVED_ON = datetime(2026, 11, 17, tzinfo=UTC)
LEASE_FUNCTIONS = {
    "lease_policy_sources": "lease_policy_sources(text, integer, integer)",
    "lease_policy_builds": "lease_policy_builds(text, integer, integer)",
}


def _migration() -> ModuleType:
    # The file name starts with a date, so it cannot be imported by name.
    path = next((ROOT / "migrations" / "versions").glob("*_8e41b6d0a7c3_policy_ingestion.py"))
    spec = importlib.util.spec_from_file_location("policy_ingestion_migration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@asynccontextmanager
async def rejected(connection: AsyncConnection, match: str) -> AsyncIterator[None]:
    # The savepoint keeps the test's transaction usable after the expected database error.
    with pytest.raises(DBAPIError, match=match):
        async with connection.begin_nested():
            yield


async def _insert(connection: AsyncConnection, table: str, **values: object) -> UUID:
    # Table and column names are this test's literals.
    values.setdefault("id", uuid4())
    names = ", ".join(values)
    params = ", ".join(f":{name}" for name in values)
    await connection.execute(text(f"INSERT INTO {table} ({names}) VALUES ({params})"), values)  # noqa: S608
    created = values["id"]
    assert isinstance(created, UUID)
    return created


async def _scalar(connection: AsyncConnection, sql: str, **params: object) -> Any:
    return await connection.scalar(text(sql), params)


@dataclass(frozen=True, slots=True)
class Graph:
    tenant_id: UUID
    bundle_id: UUID
    node_a: UUID  # RULE node with the rule below
    node_b: UUID  # CLAUSE node
    spare: UUID  # RULE node without a rule
    rule: UUID
    edge: UUID
    wording: UUID


async def _graph(connection: AsyncConnection, tenant_id: UUID, bundle_id: UUID) -> Graph:
    # One row in every guarded table. Written as the owner, who the frozen-graph trigger exempts,
    # so it works in a bundle of any state. Leaves the connection as the API role, tenant bound.
    async with as_tenant_owner(connection, tenant_id):
        scope = {"tenant_id": tenant_id, "policy_bundle_id": bundle_id}
        nodes = {
            key: await _insert(
                connection,
                "policy_nodes",
                **scope,
                node_key=key,
                node_type=kind,
                text=f"[SYNTHETIC] {key}",
                language="en",
            )
            for key, kind in (("R-A", "RULE"), ("C-B", "CLAUSE"), ("R-SPARE", "RULE"))
        }
        rule = await _rule(connection, scope, nodes["R-A"], "SYN-01")
        edge = await _insert(
            connection,
            "policy_edges",
            **scope,
            edge_type="supported_by",
            from_node_id=nodes["R-A"],
            to_node_id=nodes["C-B"],
        )
        wording = await _wording(connection, scope, rule, "EXACT_WORDING", "[SYNTHETIC] exact")
    return Graph(
        tenant_id, bundle_id, nodes["R-A"], nodes["C-B"], nodes["R-SPARE"], rule, edge, wording
    )


async def _rule(
    connection: AsyncConnection, scope: dict[str, UUID], node_id: UUID, rule_id: str
) -> UUID:
    return await _insert(
        connection,
        "policy_rules",
        **scope,
        node_id=node_id,
        rule_id=rule_id,
        rule_version=1,
        title="[SYNTHETIC] rule",
        requirement="[SYNTHETIC] requirement",
        gate="policy",
        severity="major",
        mandatory=True,
        check_type="semantic",
    )


async def _wording(
    connection: AsyncConnection,
    scope: dict[str, UUID],
    rule_id: UUID,
    kind: str,
    body: str,
    **columns: object,
) -> UUID:
    return await _insert(
        connection,
        "policy_rule_wordings",
        **scope,
        policy_rule_id=rule_id,
        language=columns.pop("language", "en"),
        kind=kind,
        text=body,
        **columns,
    )


def _scope(graph: Graph) -> dict[str, UUID]:
    return {"tenant_id": graph.tenant_id, "policy_bundle_id": graph.bundle_id}


async def _bundle(
    connection: AsyncConnection,
    tenant: SeededTenant,
    version: int,
    state: BundleLifecycleState = BundleLifecycleState.DRAFT,
) -> UUID:
    async with as_tenant_owner(connection, tenant.tenant_id):
        return await add_policy_bundle(connection, tenant.tenant_id, version, state)


async def _count(connection: AsyncConnection, table: str, tenant_id: UUID | None) -> int:
    await bind_test_tenant(connection, tenant_id) if tenant_id else await connection.execute(
        text("SELECT set_config('app.tenant_id', '', true)")
    )
    return int(await _scalar(connection, f"SELECT count(*) FROM {table}") or 0)  # noqa: S608


async def _denied(connection: AsyncConnection, sql: str) -> bool:
    # Whether the statement fails for lack of privilege; nothing it does is kept.
    savepoint = await connection.begin_nested()
    try:
        await connection.execute(text(sql))
    except DBAPIError as error:
        message = str(error)
    else:
        message = ""
    await savepoint.rollback()
    return "permission denied" in message


# Tenancy: RLS and composite foreign keys on the new tables


@pytest.mark.parametrize(
    ("table", "role"),
    [
        ("policy_rule_wordings", APP_ROLE),
        ("policy_rule_wordings", WORKER_ROLE),
        ("policy_audit_events", APP_ROLE),
    ],
)
async def test_no_role_reads_another_tenants_wordings_or_audit_rows(
    connection: AsyncConnection, world: World, table: str, role: str
) -> None:
    assert world.a.policy_bundle_id is not None
    graph = await _graph(connection, world.a.tenant_id, world.a.policy_bundle_id)
    async with as_tenant_owner(connection, world.a.tenant_id):
        await _insert(
            connection,
            "policy_audit_events",
            tenant_id=world.a.tenant_id,
            policy_bundle_id=graph.bundle_id,
            action="CREATE",
            actor="synthetic-fixture",
        )
    await as_role(connection, role)

    visible = {
        "own": await _count(connection, table, world.a.tenant_id),
        "other": await _count(connection, table, world.b.tenant_id),
        "none": await _count(connection, table, None),
    }

    assert visible == {"own": 1, "other": 0, "none": 0}


async def test_rls_blocks_writing_a_wording_or_audit_row_for_another_tenant(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    graph = await _graph(connection, world.a.tenant_id, world.a.policy_bundle_id)
    await bind_test_tenant(connection, world.b.tenant_id)

    async with rejected(connection, "row-level security"):
        await _insert(
            connection,
            "policy_audit_events",
            tenant_id=world.a.tenant_id,
            policy_bundle_id=graph.bundle_id,
            action="CREATE",
            actor="x",
        )


async def test_wording_cannot_point_at_another_tenants_rule(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    graph = await _graph(connection, world.a.tenant_id, world.a.policy_bundle_id)
    bundle_b = await _bundle(connection, world.b, 1)

    async with (
        as_tenant_owner(connection, world.b.tenant_id),
        rejected(connection, "fk_policy_rule_wordings_tenant_id_policy_rules"),
    ):
        await _wording(
            connection,
            {"tenant_id": world.b.tenant_id, "policy_bundle_id": bundle_b},
            graph.rule,
            "PROHIBITED_PHRASE",
            "x",
        )


async def test_wording_cannot_point_at_a_rule_of_another_bundle(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    graph = await _graph(connection, world.a.tenant_id, world.a.policy_bundle_id)
    other = await _bundle(connection, world.a, 2)

    async with (
        as_tenant_owner(connection, world.a.tenant_id),
        rejected(connection, "fk_policy_rule_wordings_tenant_id_policy_rules"),
    ):
        await _wording(
            connection,
            {"tenant_id": world.a.tenant_id, "policy_bundle_id": other},
            graph.rule,
            "PROHIBITED_PHRASE",
            "x",
        )


async def test_audit_event_cannot_point_at_another_tenants_bundle(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    async with (
        as_tenant_owner(connection, world.b.tenant_id),
        rejected(connection, "fk_policy_audit_events_tenant_id_policy_bundles"),
    ):
        await _insert(
            connection,
            "policy_audit_events",
            tenant_id=world.b.tenant_id,
            policy_bundle_id=world.a.policy_bundle_id,
            action="CREATE",
            actor="x",
        )


async def test_bundle_cannot_be_based_on_another_tenants_bundle(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_b = await add_policy_bundle(
            connection, world.b.tenant_id, 1, BundleLifecycleState.DRAFT
        )
        async with rejected(connection, "fk_policy_bundles_tenant_id_policy_bundles"):
            await connection.execute(
                text("UPDATE policy_bundles SET based_on_bundle_id = :a WHERE id = :b"),
                {"a": world.a.policy_bundle_id, "b": bundle_b},
            )


# Review state on nodes and edges


async def _two_nodes(connection: AsyncConnection, world: World) -> Graph:
    assert world.a.policy_bundle_id is not None
    return await _graph(connection, world.a.tenant_id, world.a.policy_bundle_id)


async def test_new_graph_rows_start_proposed_and_human(
    connection: AsyncConnection, world: World
) -> None:
    graph = await _two_nodes(connection, world)

    rows = [
        (
            await connection.execute(
                text(f"SELECT review_state, origin FROM {table} WHERE id = :id"),  # noqa: S608
                {"id": row},
            )
        ).one()
        for table, row in (
            ("policy_nodes", graph.node_a),
            ("policy_edges", graph.edge),
            ("policy_rule_wordings", graph.wording),
        )
    ]

    assert [(r.review_state, r.origin) for r in rows] == [("PROPOSED", "HUMAN")] * 3


@pytest.mark.parametrize("table", ["policy_nodes", "policy_edges"])
@pytest.mark.parametrize(
    ("columns", "constraint"),
    [
        ({"approved": True, "review_state": "PROPOSED"}, "approved_matches_review_state"),
        (
            {"approved": False, "review_state": "APPROVED", "reviewed_by": "x"},
            "approved_matches_review_state",
        ),
        ({"approved": False, "review_state": "REJECTED"}, "reviewed_when_decided"),
        (
            {"approved": True, "review_state": "APPROVED", "provenance_note": "n"},
            "reviewed_when_decided",
        ),
    ],
    ids=["approved-but-proposed", "approved-state-without-flag", "rejected-unsigned", "unsigned"],
)
async def test_review_state_checks_on_nodes_and_edges(
    connection: AsyncConnection,
    world: World,
    table: str,
    columns: dict[str, object],
    constraint: str,
) -> None:
    graph = await _two_nodes(connection, world)
    if table == "policy_nodes":
        row: dict[str, object] = {
            "node_key": "R-NEW",
            "node_type": "RULE",
            "text": "x",
            "language": "en",
        }
        # provenance_note belongs to edges only.
        columns = {k: v for k, v in columns.items() if k != "provenance_note"}
    else:
        row = {"edge_type": "except_when", "from_node_id": graph.node_a, "to_node_id": graph.node_b}
        if columns.get("approved") and "provenance_note" not in columns:
            columns = {**columns, "provenance_note": "n"}

    async with (
        as_tenant_owner(connection, graph.tenant_id),
        rejected(connection, f"ck_{table}_{constraint}"),
    ):
        await _insert(connection, table, **_scope(graph), **row, **columns)


@pytest.mark.parametrize("table", ["policy_nodes", "policy_edges"])
async def test_a_decided_row_with_a_reviewer_is_accepted(
    connection: AsyncConnection, world: World, table: str
) -> None:
    graph = await _two_nodes(connection, world)
    decided = {"reviewed_by": "reviewer-a", "reviewed_at": APPROVED_ON}
    states: list[dict[str, object]] = [
        {"review_state": "APPROVED", "approved": True},
        {"review_state": "REJECTED", "approved": False},
    ]
    async with as_tenant_owner(connection, graph.tenant_id):
        for n, state in enumerate(states):
            if table == "policy_nodes":
                row: dict[str, object] = {
                    "node_key": f"R-NEW-{n}",
                    "node_type": "RULE",
                    "text": "x",
                    "language": "en",
                }
            else:
                row = {
                    "edge_type": "except_when",
                    "from_node_id": graph.node_a if n == 0 else graph.spare,
                    "to_node_id": graph.node_b,
                    "provenance_note": "n",
                }
            await _insert(connection, table, **_scope(graph), **row, **state, **decided)

        count = await _scalar(
            connection,
            f"SELECT count(*) FROM {table} WHERE reviewed_by = 'reviewer-a'",  # noqa: S608
        )

    assert count == 2


# Bundles: one ACTIVE per tenant, approver, revision


async def test_a_second_active_bundle_of_a_tenant_is_refused(
    connection: AsyncConnection, world: World
) -> None:
    async with (
        as_tenant_owner(connection, world.a.tenant_id),
        rejected(connection, "uq_policy_bundles_one_active"),
    ):
        await _insert_bundle(connection, world.a.tenant_id, 2, "ACTIVE")


async def test_another_tenant_may_have_its_own_active_bundle(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.b.tenant_id):
        await _insert_bundle(connection, world.b.tenant_id, 1, "ACTIVE")
        count = await _scalar(
            connection,
            "SELECT count(*) FROM policy_bundles WHERE lifecycle_state = 'ACTIVE'"
            " AND tenant_id = :t",
            t=world.b.tenant_id,
        )

    assert count == 1


async def _insert_bundle(
    connection: AsyncConnection,
    tenant_id: UUID,
    version: int,
    state: str,
    **columns: object,
) -> UUID:
    values: dict[str, object] = {
        "tenant_id": tenant_id,
        "version": version,
        "name": f"[SYNTHETIC] Policy v{version}",
        "source_ref": "synthetic:test-fixture",
        "rules": "[]",
        "is_synthetic": True,
        "lifecycle_state": state,
        "approved_by": "synthetic-fixture",
        "approved_at": APPROVED_ON,
        **columns,
    }
    # rules is JSONB: pass it as text and cast.
    names = ", ".join(values)
    params = ", ".join("CAST(:rules AS jsonb)" if n == "rules" else f":{n}" for n in values)
    bundle_id = uuid4()
    await connection.execute(
        text(f"INSERT INTO policy_bundles (id, {names}) VALUES (:id, {params})"),  # noqa: S608
        {
            "id": bundle_id,
            **values,
        },
    )
    return bundle_id


async def test_data_step_keeps_the_highest_active_version_per_tenant(
    connection: AsyncConnection, world: World
) -> None:
    # Legacy data: several ACTIVE bundles per tenant. The migration's own SQL, run on that data
    # with the index that forbids it dropped (DDL rolls back with the test).
    sql = str(_migration().RETIRE_SUPERSEDED_ACTIVE)
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(text("DROP INDEX uq_policy_bundles_one_active"))
        await _insert_bundle(connection, world.a.tenant_id, 3, "ACTIVE")
        await _insert_bundle(connection, world.a.tenant_id, 2, "ACTIVE")
        await _insert_bundle(connection, world.a.tenant_id, 4, "DRAFT", approved_by=None)
        await _insert_bundle(connection, world.b.tenant_id, 1, "ACTIVE")
        await _insert_bundle(connection, world.b.tenant_id, 2, "READY")

        await connection.execute(text(sql))
        rows = (
            await connection.execute(
                text(
                    "SELECT tenant_id, version, lifecycle_state FROM policy_bundles"
                    " WHERE tenant_id IN (:a, :b)"
                ),
                {"a": world.a.tenant_id, "b": world.b.tenant_id},
            )
        ).all()

    assert {(r.tenant_id, r.version): r.lifecycle_state for r in rows} == {
        (world.a.tenant_id, 1): "RETIRED",
        (world.a.tenant_id, 2): "RETIRED",
        (world.a.tenant_id, 3): "ACTIVE",
        (world.a.tenant_id, 4): "DRAFT",
        (world.b.tenant_id, 1): "ACTIVE",
        (world.b.tenant_id, 2): "READY",
    }


@pytest.mark.parametrize("state", ["APPROVED", "BUILDING", "READY", "ACTIVE", "RETIRED"])
@pytest.mark.parametrize("missing", ["approved_by", "approved_at"])
async def test_states_from_approved_on_need_an_approver(
    connection: AsyncConnection, world: World, state: str, missing: str
) -> None:
    async with (
        as_tenant_owner(connection, world.b.tenant_id),
        rejected(connection, "ck_policy_bundles_approved_states_have_approver"),
    ):
        await _insert_bundle(connection, world.b.tenant_id, 1, state, **{missing: None})


@pytest.mark.parametrize("state", ["DRAFT", "REVIEW", "REJECTED"])
async def test_drafts_have_no_approver_yet(
    connection: AsyncConnection, world: World, state: str
) -> None:
    async with as_tenant_owner(connection, world.b.tenant_id):
        await _insert_bundle(
            connection, world.b.tenant_id, 1, state, approved_by=None, approved_at=None
        )
        count = await _scalar(
            connection,
            "SELECT count(*) FROM policy_bundles WHERE tenant_id = :t AND approved_by IS NULL",
            t=world.b.tenant_id,
        )

    assert count == 1


async def test_bundle_revision_starts_at_one_and_stays_positive(
    connection: AsyncConnection, world: World
) -> None:
    bundle = await _bundle(connection, world.b, 1)
    async with as_tenant_owner(connection, world.b.tenant_id):
        revision = await _scalar(
            connection, "SELECT revision FROM policy_bundles WHERE id = :b", b=bundle
        )
        async with rejected(connection, "ck_policy_bundles_revision_positive"):
            await connection.execute(
                text("UPDATE policy_bundles SET revision = 0 WHERE id = :b"), {"b": bundle}
            )

    assert revision == 1


# The frozen-graph trigger


async def _insert_spare_node(connection: AsyncConnection, graph: Graph) -> UUID:
    return await _insert(
        connection,
        "policy_nodes",
        **_scope(graph),
        node_key="R-NEW",
        node_type="RULE",
        text="x",
        language="en",
    )


async def _insert_spare_rule(connection: AsyncConnection, graph: Graph) -> UUID:
    return await _rule(connection, _scope(graph), graph.spare, "SYN-02")


async def _insert_spare_edge(connection: AsyncConnection, graph: Graph) -> UUID:
    return await _insert(
        connection,
        "policy_edges",
        **_scope(graph),
        edge_type="except_when",
        from_node_id=graph.node_a,
        to_node_id=graph.node_b,
    )


async def _insert_spare_wording(connection: AsyncConnection, graph: Graph) -> UUID:
    return await _wording(connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", "spare")


INSERTS: dict[str, Callable[[AsyncConnection, Graph], Awaitable[UUID]]] = {
    "policy_nodes": _insert_spare_node,
    "policy_rules": _insert_spare_rule,
    "policy_edges": _insert_spare_edge,
    "policy_rule_wordings": _insert_spare_wording,
}
UPDATES = {
    "policy_nodes": "UPDATE policy_nodes SET review_note = 'x' WHERE id = :id",
    "policy_rules": "UPDATE policy_rules SET title = 'changed' WHERE id = :id",
    "policy_edges": "UPDATE policy_edges SET review_note = 'x' WHERE id = :id",
    "policy_rule_wordings": "UPDATE policy_rule_wordings SET review_note = 'x' WHERE id = :id",
}


def _row_of(graph: Graph, table: str) -> UUID:
    return {
        "policy_nodes": graph.node_a,
        "policy_rules": graph.rule,
        "policy_edges": graph.edge,
        "policy_rule_wordings": graph.wording,
    }[table]


@pytest.mark.parametrize("state", STATES, ids=lambda s: s.value)
@pytest.mark.parametrize("table", INSERTS)
async def test_api_role_inserts_graph_rows_only_into_draft_or_review_bundles(
    connection: AsyncConnection, world: World, table: str, state: BundleLifecycleState
) -> None:
    bundle = await _bundle(connection, world.a, 2, state)
    graph = await _graph(connection, world.a.tenant_id, bundle)

    if state in OPEN_STATES:
        await INSERTS[table](connection, graph)
    else:
        async with rejected(connection, "policy graph is frozen"):
            await INSERTS[table](connection, graph)


@pytest.mark.parametrize("state", STATES, ids=lambda s: s.value)
@pytest.mark.parametrize("table", UPDATES)
async def test_api_role_updates_graph_rows_only_in_draft_or_review_bundles(
    connection: AsyncConnection, world: World, table: str, state: BundleLifecycleState
) -> None:
    bundle = await _bundle(connection, world.a, 2, state)
    graph = await _graph(connection, world.a.tenant_id, bundle)
    statement = text(UPDATES[table])

    if state in OPEN_STATES:
        result = await connection.execute(statement, {"id": _row_of(graph, table)})
        assert result.rowcount == 1
    else:
        async with rejected(connection, "policy graph is frozen"):
            await connection.execute(statement, {"id": _row_of(graph, table)})


async def test_frozen_graph_error_is_p0001(connection: AsyncConnection, world: World) -> None:
    bundle = await _bundle(connection, world.a, 2, BundleLifecycleState.READY)
    graph = await _graph(connection, world.a.tenant_id, bundle)

    with pytest.raises(DBAPIError, match="policy graph is frozen") as caught:
        async with connection.begin_nested():
            await _insert_spare_node(connection, graph)

    assert getattr(caught.value.orig, "sqlstate", None) == "P0001"


async def test_graph_write_is_refused_when_the_bundle_is_not_visible(
    connection: AsyncConnection, world: World
) -> None:
    bundle = await _bundle(connection, world.a, 2)
    graph = await _graph(connection, world.a.tenant_id, bundle)
    # Bound to B, writing a row of A's draft bundle: B cannot see the bundle, so the trigger
    # refuses before row-level security would.
    await bind_test_tenant(connection, world.b.tenant_id)

    async with rejected(connection, "policy bundle is not visible to this role"):
        await _insert_spare_node(connection, graph)


async def test_owner_may_write_graph_rows_of_an_active_bundle(
    connection: AsyncConnection, world: World
) -> None:
    # Import and seed scripts run as the owner and backfill legacy ACTIVE bundles.
    assert world.a.policy_bundle_id is not None
    graph = await _graph(connection, world.a.tenant_id, world.a.policy_bundle_id)

    async with as_tenant_owner(connection, world.a.tenant_id):
        await _insert_spare_node(connection, graph)
        result = await connection.execute(text(UPDATES["policy_nodes"]), {"id": graph.node_a})

    assert result.rowcount == 1


async def test_graph_rows_cannot_move_into_a_frozen_bundle(
    connection: AsyncConnection, world: World
) -> None:
    draft = await _bundle(connection, world.a, 2)
    frozen = await _bundle(connection, world.a, 3, BundleLifecycleState.READY)
    graph = await _graph(connection, world.a.tenant_id, draft)
    await _graph(connection, world.a.tenant_id, frozen)

    # The edge has no composite key to the new bundle's nodes, so only the trigger can refuse.
    async with rejected(connection, "policy graph is frozen"):
        await connection.execute(
            text("UPDATE policy_nodes SET policy_bundle_id = :f WHERE id = :n"),
            {"f": frozen, "n": graph.spare},
        )


# Wordings


@pytest.fixture
async def graph(connection: AsyncConnection, world: World) -> Graph:
    return await _two_nodes(connection, world)


async def test_wording_text_is_stored_with_its_sha256(
    connection: AsyncConnection, graph: Graph
) -> None:
    stored = await _scalar(
        connection, "SELECT text_sha256 FROM policy_rule_wordings WHERE id = :id", id=graph.wording
    )

    assert stored == hashlib.sha256(b"[SYNTHETIC] exact").hexdigest()


async def test_the_same_wording_text_twice_for_a_rule_is_refused(
    connection: AsyncConnection, graph: Graph
) -> None:
    async with as_tenant_owner(connection, graph.tenant_id):
        await _wording(connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", "cure")
        async with rejected(connection, "uq_policy_rule_wordings_rule_language_kind_text"):
            await _wording(connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", "cure")


async def test_the_same_text_is_allowed_in_another_language_or_kind(
    connection: AsyncConnection, graph: Graph
) -> None:
    async with as_tenant_owner(connection, graph.tenant_id):
        await _wording(connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", "cure")
        await _wording(
            connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", "cure", language="es"
        )
        await _wording(
            connection, _scope(graph), graph.rule, "EXACT_WORDING", "cure", language="es"
        )
        count = await _scalar(
            connection, "SELECT count(*) FROM policy_rule_wordings WHERE text = 'cure'"
        )

    assert count == 3


async def test_a_rule_has_one_live_exact_wording_per_language(
    connection: AsyncConnection, graph: Graph
) -> None:
    # The fixture graph already has an English one.
    async with as_tenant_owner(connection, graph.tenant_id):
        async with rejected(connection, "uq_policy_rule_wordings_one_exact"):
            await _wording(connection, _scope(graph), graph.rule, "EXACT_WORDING", "[SYNTHETIC] 2")
        # Another language is fine, and so is a replacement once the old one is rejected.
        await _wording(
            connection, _scope(graph), graph.rule, "EXACT_WORDING", "[SYNTHETIC] es", language="es"
        )
        await connection.execute(
            text(
                "UPDATE policy_rule_wordings SET review_state = 'REJECTED', reviewed_by = 'r',"
                " reviewed_at = now() WHERE id = :id"
            ),
            {"id": graph.wording},
        )
        await _wording(connection, _scope(graph), graph.rule, "EXACT_WORDING", "[SYNTHETIC] 2")


async def test_a_rule_may_have_many_prohibited_phrases(
    connection: AsyncConnection, graph: Graph
) -> None:
    async with as_tenant_owner(connection, graph.tenant_id):
        for phrase in ("cure", "cures", "proven"):
            await _wording(connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", phrase)
        count = await _scalar(
            connection, "SELECT count(*) FROM policy_rule_wordings WHERE kind = 'PROHIBITED_PHRASE'"
        )

    assert count == 3


@pytest.mark.parametrize("body", ["", "   ", "\t\n"])
async def test_a_blank_wording_is_refused(
    connection: AsyncConnection, graph: Graph, body: str
) -> None:
    async with (
        as_tenant_owner(connection, graph.tenant_id),
        rejected(connection, "ck_policy_rule_wordings_text_not_blank"),
    ):
        await _wording(connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", body)


async def test_a_wording_language_must_be_a_policy_language(
    connection: AsyncConnection, graph: Graph
) -> None:
    async with (
        as_tenant_owner(connection, graph.tenant_id),
        rejected(connection, "ck_policy_rule_wordings_language"),
    ):
        await _wording(
            connection, _scope(graph), graph.rule, "PROHIBITED_PHRASE", "x", language="fr"
        )


async def test_a_translation_points_at_a_wording_of_the_same_bundle(
    connection: AsyncConnection, world: World, graph: Graph
) -> None:
    other = await _bundle(connection, world.a, 2)
    other_graph = await _graph(connection, world.a.tenant_id, other)

    async with as_tenant_owner(connection, graph.tenant_id):
        await _wording(
            connection,
            _scope(graph),
            graph.rule,
            "EXACT_WORDING",
            "[SYNTHETIC] es",
            language="es",
            translation_of_id=graph.wording,
        )
        async with rejected(connection, "fk_policy_rule_wordings_tenant_id_policy_rule_wordings"):
            await _wording(
                connection,
                _scope(other_graph),
                other_graph.rule,
                "PROHIBITED_PHRASE",
                "t",
                translation_of_id=graph.wording,
            )


async def test_backfill_of_wordings_matches_what_the_import_writes(
    connection: AsyncConnection, world: World
) -> None:
    # The import writes wording rows; the migration backfills them for graphs imported before.
    # Run the migration's SQL over an imported graph and compare. One rule gets an exact wording
    # and a repeated phrase, one rule node is not approved.
    assert world.a.policy_bundle_id is not None
    tenant_id, bundle_id = world.a.tenant_id, world.a.policy_bundle_id
    graph = load_synthetic_graph()
    rules = list(graph.rules)
    rules[0] = rules[0].model_copy(
        update={
            "exact_wording": "[SYNTHETIC] exact words.",
            "prohibited_phrases": [*rules[0].prohibited_phrases, "  cure "],
        }
    )
    nodes = [
        node.model_copy(update={"approved": False}) if node.node_key == "REC-02" else node
        for node in graph.nodes
    ]
    edges = [
        edge
        for edge in graph.edges
        if "REC-02" not in (edge.from_node_key, edge.to_node_key) or not edge.approved
    ]
    graph = graph.model_copy(update={"rules": rules, "nodes": nodes, "edges": edges})
    sql = str(_migration().BACKFILL_WORDINGS)
    columns = (
        "SELECT r.rule_id, w.language, w.kind, w.text, w.review_state, w.origin,"
        " w.reviewed_by IS NOT NULL AS reviewed FROM policy_rule_wordings AS w"
        " JOIN policy_rules AS r ON r.id = w.policy_rule_id"
    )

    async with owner_session(connection, tenant_id) as db:
        await import_policy_graph(db, tenant_id, bundle_id, graph)
        imported = (await connection.execute(text(columns))).all()
        await connection.execute(text("DELETE FROM policy_rule_wordings"))
        await connection.execute(text(sql))
        backfilled = (await connection.execute(text(columns))).all()

    assert len(imported) > 10
    assert {tuple(row) for row in backfilled} == {tuple(row) for row in imported}
    assert len(backfilled) == len(imported)
    kinds = {(row.rule_id, row.kind, row.review_state) for row in imported}
    assert ("REC-01", "EXACT_WORDING", "APPROVED") in kinds
    assert ("REC-02", "PROHIBITED_PHRASE", "PROPOSED") in kinds


# Outbox, uploads, sources


async def _outbox(connection: AsyncConnection, tenant_id: UUID, event_type: str) -> None:
    await _insert(
        connection,
        "outbox_events",
        tenant_id=tenant_id,
        job_id=None,
        event_type=event_type,
        payload="{}",
        attempts=0,
    )


@pytest.mark.parametrize("event_type", ["policy_bundle.activated", "policy_bundle.retired"])
async def test_policy_bundle_events_have_no_job(
    connection: AsyncConnection, world: World, event_type: str
) -> None:
    await bind_test_tenant(connection, world.a.tenant_id)

    await _outbox(connection, world.a.tenant_id, event_type)

    assert await _scalar(connection, "SELECT count(*) FROM outbox_events WHERE job_id IS NULL") == 1


@pytest.mark.parametrize(
    "event_type", ["job.run", "asset.generated", "policy_bundle", "policy_bundleX"]
)
async def test_other_events_still_need_a_job(
    connection: AsyncConnection, world: World, event_type: str
) -> None:
    await bind_test_tenant(connection, world.a.tenant_id)

    async with rejected(connection, "ck_outbox_events_job_id_unless_policy_bundle"):
        await _outbox(connection, world.a.tenant_id, event_type)


async def _upload(
    connection: AsyncConnection, tenant: SeededTenant, purpose: str, study_id: UUID | None
) -> UUID:
    upload_id = uuid4()
    return await _insert(
        connection,
        "media_uploads",
        id=upload_id,
        tenant_id=tenant.tenant_id,
        study_id=study_id,
        purpose=purpose,
        state="READY",
        name="synthetic.pdf",
        declared_mime_type="application/pdf",
        declared_size_bytes=1024,
        declared_checksum_sha256="0" * 64,
        bucket="synthetic-bucket",
        s3_key=f"uploads/2026/11/{upload_id}.pdf",
        upload_url_expires_at=datetime(2026, 11, 30, tzinfo=UTC),
        identifiable_person="no",
        created_by="creator-a",
    )


async def test_creative_media_needs_a_study(connection: AsyncConnection, world: World) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        await _upload(connection, world.a, "creative_media", world.a.study_id)
        async with rejected(connection, "ck_media_uploads_creative_media_has_study"):
            await _upload(connection, world.a, "creative_media", None)


async def test_policy_source_upload_has_no_study(connection: AsyncConnection, world: World) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        upload = await _upload(connection, world.a, "policy_source", None)
        study = await _scalar(
            connection, "SELECT study_id FROM media_uploads WHERE id = :id", id=upload
        )

    assert study is None


async def _source(connection: AsyncConnection, tenant: SeededTenant, **columns: object) -> UUID:
    return await _insert(
        connection,
        "policy_sources",
        tenant_id=tenant.tenant_id,
        title="[SYNTHETIC] source",
        source_ref="synthetic:test-fixture",
        checksum_sha256=uuid4().hex + uuid4().hex,
        language="en",
        is_synthetic=True,
        created_by="synthetic-fixture",
        **columns,
    )


async def test_a_source_waiting_for_extraction_needs_an_upload(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        upload = await _upload(connection, world.a, "policy_source", None)
        await _source(connection, world.a, extraction_state="PENDING", upload_id=upload)
        async with rejected(connection, "ck_policy_sources_extraction_has_upload"):
            await _source(connection, world.a, extraction_state="PENDING")
        async with rejected(connection, "ck_policy_sources_extraction_has_upload"):
            await _source(connection, world.a, extraction_state="FAILED")


async def test_a_source_that_arrives_with_clauses_needs_no_upload(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        source = await _source(connection, world.a)
        state = await _scalar(
            connection, "SELECT extraction_state FROM policy_sources WHERE id = :id", id=source
        )

    assert state == "EXTRACTED"


async def test_an_upload_feeds_one_source_only(connection: AsyncConnection, world: World) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        upload = await _upload(connection, world.a, "policy_source", None)
        await _source(connection, world.a, extraction_state="PENDING", upload_id=upload)
        async with rejected(connection, "uq_policy_sources_tenant_id_upload_id"):
            await _source(connection, world.a, extraction_state="PENDING", upload_id=upload)


async def test_a_source_cannot_use_another_tenants_upload(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        upload = await _upload(connection, world.a, "policy_source", None)
    async with (
        as_tenant_owner(connection, world.b.tenant_id),
        rejected(connection, "fk_policy_sources_tenant_id_media_uploads"),
    ):
        await _source(connection, world.b, extraction_state="PENDING", upload_id=upload)


# Lease functions


@pytest.mark.parametrize("name", LEASE_FUNCTIONS)
async def test_only_the_worker_role_leases_policy_work(
    connection: AsyncConnection, world: World, name: str
) -> None:
    denied: dict[str, bool] = {}
    for role in (APP_ROLE, WORKER_ROLE):
        await as_role(connection, role)
        denied[role] = await _denied(connection, f"SELECT * FROM {name}('probe', 1, 5)")  # noqa: S608

    assert denied == {APP_ROLE: True, WORKER_ROLE: False}


@pytest.mark.parametrize(("name", "signature"), LEASE_FUNCTIONS.items())
async def test_policy_lease_function_checklist(
    connection: AsyncConnection, name: str, signature: str
) -> None:
    # The same review list as lease_media_uploads: fixed search_path, worker-only EXECUTE, ids out.
    row = (
        await connection.execute(
            text(
                "SELECT p.proconfig, p.prosecdef, pg_get_function_result(p.oid) AS result,"
                " p.prosrc, has_function_privilege('marketing_app', p.oid, 'EXECUTE') AS app,"
                " has_function_privilege('marketing_worker', p.oid, 'EXECUTE') AS worker,"
                " coalesce(array_to_string(p.proacl, ','), '') AS acl"
                " FROM pg_proc AS p WHERE p.oid = CAST(:f AS regprocedure)"
            ),
            {"f": signature},
        )
    ).one()

    assert row.proconfig == ["search_path=pg_catalog, public, pg_temp"]
    id_column = "source_id" if name == "lease_policy_sources" else "bundle_id"
    assert row.result == f"TABLE({id_column} uuid, tenant_id uuid)"
    assert (row.app, row.worker) == (False, True)
    assert not any(entry.startswith("=") for entry in row.acl.split(","))
    assert row.prosecdef
    assert "FOR UPDATE SKIP LOCKED" in row.prosrc


@pytest.mark.parametrize(
    ("table", "policy"),
    [("policy_sources", "policy_source_lease"), ("policy_bundles", "policy_build_lease")],
)
async def test_policy_lease_policies_are_scoped_to_the_function_owner(
    connection: AsyncConnection, table: str, policy: str
) -> None:
    owner = await _scalar(
        connection,
        "SELECT pg_get_userbyid(proowner) FROM pg_proc WHERE oid = CAST(:f AS regprocedure)",
        f=LEASE_FUNCTIONS["lease_policy_sources"],
    )
    roles = await _scalar(
        connection,
        "SELECT array_to_string(roles::regrole[], ',') FROM pg_policies"
        " WHERE tablename = :t AND policyname = :p",
        t=table,
        p=policy,
    )

    assert roles == owner
    assert owner not in (APP_ROLE, WORKER_ROLE)


async def _lease(connection: AsyncConnection, function: str, id_column: str) -> set[UUID]:
    await as_role(connection, WORKER_ROLE)
    rows = (
        await connection.execute(text(f"SELECT {id_column} FROM {function}('w1', 50, 30)"))  # noqa: S608
    ).all()
    await connection.execute(text("RESET ROLE"))
    return {row[0] for row in rows}


async def test_lease_policy_sources_takes_only_due_extractions_across_tenants(
    connection: AsyncConnection, world: World
) -> None:
    due: set[UUID] = set()
    for tenant, due_in in ((world.a, timedelta(0)), (world.b, timedelta(0))):
        async with as_tenant_owner(connection, tenant.tenant_id):
            upload = await _upload(connection, tenant, "policy_source", None)
            source = await _source(
                connection,
                tenant,
                extraction_state="PENDING",
                upload_id=upload,
                available_at=await _scalar(
                    connection, "SELECT clock_timestamp() + CAST(:d AS interval)", d=due_in
                ),
            )
            due.add(source)
    async with as_tenant_owner(connection, world.a.tenant_id):
        later = await _upload(connection, world.a, "policy_source", None)
        await _source(
            connection,
            world.a,
            extraction_state="PENDING",
            upload_id=later,
            available_at=await _scalar(
                connection, "SELECT clock_timestamp() + interval '5 minutes'"
            ),
        )
        await _source(connection, world.a)  # EXTRACTED: never due
        unscheduled = await _upload(connection, world.a, "policy_source", None)
        await _source(connection, world.a, extraction_state="PENDING", upload_id=unscheduled)

    assert await _lease(connection, "lease_policy_sources", "source_id") == due
    # Leased rows are not handed out again while the lease holds.
    assert await _lease(connection, "lease_policy_sources", "source_id") == set()


async def test_lease_policy_builds_takes_only_approved_bundles_due_for_a_build(
    connection: AsyncConnection, world: World
) -> None:
    now_sql = (
        "UPDATE policy_bundles SET available_at = clock_timestamp() + CAST(:d AS interval)"
        " WHERE id = :id"
    )
    due: set[UUID] = set()
    async with as_tenant_owner(connection, world.a.tenant_id):
        for version, state, delay in (
            (2, BundleLifecycleState.APPROVED, timedelta(0)),
            (3, BundleLifecycleState.APPROVED, timedelta(minutes=5)),
            (4, BundleLifecycleState.DRAFT, timedelta(0)),
            (5, BundleLifecycleState.READY, timedelta(0)),
            (6, BundleLifecycleState.APPROVED, None),
        ):
            bundle = await add_policy_bundle(connection, world.a.tenant_id, version, state)
            if delay is not None:
                await connection.execute(text(now_sql), {"d": delay, "id": bundle})
            if (state, delay) == (BundleLifecycleState.APPROVED, timedelta(0)):
                due.add(bundle)
    async with as_tenant_owner(connection, world.b.tenant_id):
        bundle_b = await add_policy_bundle(
            connection, world.b.tenant_id, 1, BundleLifecycleState.APPROVED
        )
        await connection.execute(text(now_sql), {"d": timedelta(0), "id": bundle_b})
        due.add(bundle_b)

    assert await _lease(connection, "lease_policy_builds", "bundle_id") == due
    assert await _lease(connection, "lease_policy_builds", "bundle_id") == set()


async def test_expired_policy_lease_is_leased_again_and_counted_as_a_failure(
    connection: AsyncConnection, world: World
) -> None:
    async with as_tenant_owner(connection, world.a.tenant_id):
        bundle = await add_policy_bundle(
            connection, world.a.tenant_id, 2, BundleLifecycleState.BUILDING
        )
        await connection.execute(
            text("UPDATE policy_bundles SET available_at = clock_timestamp() WHERE id = :id"),
            {"id": bundle},
        )
    assert await _lease(connection, "lease_policy_builds", "bundle_id") == {bundle}
    # The worker crashed mid-build: its lease runs out without being cleared.
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            text(
                "UPDATE policy_bundles SET leased_until = clock_timestamp() - interval '1 second'"
                " WHERE id = :id"
            ),
            {"id": bundle},
        )
    assert await _lease(connection, "lease_policy_builds", "bundle_id") == {bundle}

    async with as_tenant_owner(connection, world.a.tenant_id):
        counts = (
            await connection.execute(
                text("SELECT attempts, failed_attempts FROM policy_bundles WHERE id = :id"),
                {"id": bundle},
            )
        ).one()
    assert (counts.attempts, counts.failed_attempts) == (2, 1)


# Audit: insert-only for the API, insert for the worker


async def test_audit_events_can_be_added_and_read_but_never_changed_by_the_api(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    await bind_test_tenant(connection, world.a.tenant_id)
    await _insert(
        connection,
        "policy_audit_events",
        tenant_id=world.a.tenant_id,
        policy_bundle_id=world.a.policy_bundle_id,
        action="ACTIVATE",
        from_state="READY",
        to_state="ACTIVE",
        actor="reviewer-a",
        reason="synthetic",
    )

    assert await _scalar(connection, "SELECT count(*) FROM policy_audit_events") == 1
    assert await _denied(connection, "UPDATE policy_audit_events SET actor = 'someone else'")
    assert await _denied(connection, "DELETE FROM policy_audit_events")


async def test_worker_adds_audit_events_but_cannot_read_or_change_them(
    connection: AsyncConnection, world: World
) -> None:
    assert world.a.policy_bundle_id is not None
    await as_role(connection, WORKER_ROLE)
    await bind_test_tenant(connection, world.a.tenant_id)

    await _insert(
        connection,
        "policy_audit_events",
        tenant_id=world.a.tenant_id,
        policy_bundle_id=world.a.policy_bundle_id,
        action="BUILD_READY",
        actor="system:policy-worker",
    )

    assert await _denied(connection, "SELECT count(*) FROM policy_audit_events")
    assert await _denied(connection, "UPDATE policy_audit_events SET actor = 'x'")
    assert await _denied(connection, "DELETE FROM policy_audit_events")


# Grants


@pytest.mark.usefixtures("world")
async def test_api_role_privileges_on_policy_tables_and_nobody_may_delete(
    connection: AsyncConnection,
) -> None:
    mutable = {"SELECT", "INSERT", "UPDATE"}
    expected = {
        "policy_bundles": mutable,
        "policy_nodes": mutable,
        "policy_edges": mutable,
        "policy_rules": mutable,
        "policy_rule_wordings": mutable,
        "policy_sources": mutable,
        "policy_audit_events": {"SELECT", "INSERT"},
        # Written by the policy worker (clauses, embeddings, builds), only read by the API.
        "policy_source_clauses": {"SELECT"},
        "policy_node_embeddings": {"SELECT"},
        "policy_index_builds": {"SELECT"},
    }
    rows = (
        await connection.execute(
            text(
                "SELECT grantee, table_name, privilege_type"
                " FROM information_schema.role_table_grants"
                " WHERE table_name LIKE 'policy\\_%' AND grantee IN ('marketing_app',"
                " 'marketing_worker')"
            )
        )
    ).all()
    app: dict[str, set[str]] = {}
    for row in rows:
        assert row.privilege_type != "DELETE", (row.grantee, row.table_name)
        if row.grantee == APP_ROLE:
            app.setdefault(row.table_name, set()).add(row.privilege_type)

    assert app == expected


@pytest.mark.parametrize("role", [APP_ROLE, WORKER_ROLE])
@pytest.mark.parametrize("table", ["policy_rule_wordings", "policy_audit_events"])
async def test_new_tables_have_forced_row_level_security_and_a_tenant_policy(
    connection: AsyncConnection, role: str, table: str
) -> None:
    flags = (
        await connection.execute(
            text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = :t"),
            {"t": table},
        )
    ).one()
    policy = await _scalar(
        connection,
        "SELECT count(*) FROM pg_policies WHERE tablename = :t AND policyname = 'tenant_isolation'"
        " AND qual LIKE '%app.tenant_id%' AND with_check LIKE '%app.tenant_id%'",
        t=table,
    )

    assert (flags[0], flags[1], policy) == (True, True, 1)
    assert role in (APP_ROLE, WORKER_ROLE)
