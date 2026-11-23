# The worker's database role and the lease function: who may do what, checked in the database.

from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.workers.connector import WorkerConfigError, worker_database_url
from tests.conftest import APP_ROLE, WORKER_ROLE, AuthHeaders, World, _creative, _post_job, as_role

LEASE = "lease_outbox_events(text, integer, integer, text[])"
ROLES = (APP_ROLE, WORKER_ROLE)
CHECK_INPUT_TABLES = (
    "creatives",
    "study_snapshots",
    "policy_bundles",
    "policy_sources",
    "policy_source_clauses",
    "policy_nodes",
    "policy_rules",
    "policy_edges",
    "policy_node_embeddings",
    "policy_index_builds",
)


async def _bind(connection: AsyncConnection, tenant_id: UUID | None) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"),
        {"t": "" if tenant_id is None else str(tenant_id)},
    )


@pytest.fixture
async def job_in_a(client: httpx.AsyncClient, auth: AuthHeaders, world: World) -> None:
    creative_id = await _creative(client, auth, world.a)
    assert (await _post_job(client, auth, world, creative_id)).status_code == 202


@pytest.mark.usefixtures("job_in_a")
@pytest.mark.parametrize("role", ROLES)
async def test_no_role_reads_another_tenants_outbox_rows(
    connection: AsyncConnection, world: World, role: str
) -> None:
    await as_role(connection, role)
    visible: dict[str, int] = {}
    for name, tenant in (("own", world.a.tenant_id), ("other", world.b.tenant_id), ("none", None)):
        await _bind(connection, tenant)
        visible[name] = int(
            await connection.scalar(
                text("SELECT count(*) FROM outbox_events WHERE tenant_id = :a"),
                {"a": world.a.tenant_id},
            )
            or 0
        )

    assert visible == {"own": 1, "other": 0, "none": 0}


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


@pytest.mark.usefixtures("world")
async def test_only_the_worker_role_leases_events(connection: AsyncConnection) -> None:
    lease = "SELECT * FROM lease_outbox_events('probe', 1, 5, ARRAY['job.run'])"
    denied: dict[str, bool] = {}
    for role in ROLES:
        await as_role(connection, role)
        denied[role] = await _denied(connection, lease)

    assert denied == {APP_ROLE: True, WORKER_ROLE: False}


@pytest.mark.usefixtures("world")
async def test_only_the_worker_role_reads_and_updates_provider_spend(
    connection: AsyncConnection,
) -> None:
    statements = {
        "select": "SELECT count(*) FROM provider_spend",
        "update": "UPDATE provider_spend SET updated_at = updated_at",
        "insert": "INSERT INTO provider_spend (provider) VALUES ('probe')",
        "delete": "DELETE FROM provider_spend WHERE provider = 'probe'",
    }
    denied: dict[str, dict[str, bool]] = {}
    for role in ROLES:
        await as_role(connection, role)
        denied[role] = {name: await _denied(connection, sql) for name, sql in statements.items()}

    assert denied[APP_ROLE] == {"select": True, "update": True, "insert": True, "delete": True}
    assert denied[WORKER_ROLE] == {"select": False, "update": False, "insert": True, "delete": True}


async def test_lease_function_checklist(connection: AsyncConnection) -> None:
    # Dinesh's review list: fixed search_path, EXECUTE for marketing_worker only, ids out only.
    row = (
        await connection.execute(
            text(
                "SELECT p.proconfig, p.prosecdef, pg_get_function_result(p.oid) AS result,"
                " p.prosrc, pg_get_userbyid(p.proowner) AS owner,"
                " has_function_privilege('marketing_app', p.oid, 'EXECUTE') AS app,"
                " has_function_privilege('marketing_worker', p.oid, 'EXECUTE') AS worker,"
                " coalesce(array_to_string(p.proacl, ','), '') AS acl"
                " FROM pg_proc AS p WHERE p.oid = CAST(:f AS regprocedure)"
            ),
            {"f": LEASE},
        )
    ).one()

    assert row.proconfig == ["search_path=pg_catalog, public, pg_temp"]
    assert row.result == ("TABLE(event_id uuid, tenant_id uuid, job_id uuid, event_type text)")
    assert (row.app, row.worker) == (False, True)
    # No PUBLIC entry ("=X/owner") in the ACL.
    assert not any(entry.startswith("=") for entry in row.acl.split(","))
    assert row.prosecdef
    assert "FOR UPDATE SKIP LOCKED" in row.prosrc


async def test_only_the_four_argument_lease_function_exists(connection: AsyncConnection) -> None:
    # No three-argument overload: it would lease every event type.
    matches = (
        await connection.scalars(
            text(
                "SELECT p.oid = CAST(:f AS regprocedure) FROM pg_proc AS p"
                " WHERE p.proname = 'lease_outbox_events'"
            ),
            {"f": LEASE},
        )
    ).all()

    assert matches == [True]


async def test_outbox_lease_policy_is_scoped_to_the_function_owner(
    connection: AsyncConnection,
) -> None:
    owner = await connection.scalar(
        text("SELECT pg_get_userbyid(proowner) FROM pg_proc WHERE oid = CAST(:f AS regprocedure)"),
        {"f": LEASE},
    )
    roles = await connection.scalar(
        text(
            "SELECT roles FROM pg_policies"
            " WHERE tablename = 'outbox_events' AND policyname = 'outbox_lease'"
        )
    )

    assert roles == [owner]
    assert owner not in ROLES


def test_worker_refuses_to_start_without_its_database_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No .env in this directory, and no variable.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORKER_DATABASE_URL", raising=False)

    with pytest.raises(WorkerConfigError, match="WORKER_DATABASE_URL"):
        worker_database_url()


async def test_worker_role_has_exactly_the_table_privileges_it_uses(
    connection: AsyncConnection,
) -> None:
    expected = {
        "creative_versions": {"SELECT"},
        "jobs": {"SELECT", "UPDATE"},
        "credit_reservations": {"SELECT", "UPDATE"},
        "outbox_events": {"SELECT", "INSERT", "UPDATE"},
        "assets": {"SELECT", "INSERT"},
        "provider_spend": {"SELECT", "UPDATE"},
        # Text validation (NRC-7): the inputs of a check, and its manifest and run, insert-only.
        **{table: {"SELECT"} for table in CHECK_INPUT_TABLES},
        "context_manifests": {"SELECT", "INSERT"},
        "validation_runs": {"SELECT", "INSERT"},
    }
    rows = (
        await connection.execute(
            text(
                "SELECT table_name, privilege_type FROM information_schema.role_table_grants"
                " WHERE grantee = 'marketing_worker'"
            )
        )
    ).all()
    granted: dict[str, set[str]] = {}
    for row in rows:
        granted.setdefault(row.table_name, set()).add(row.privilege_type)

    assert granted == expected


@pytest.mark.usefixtures("job_in_a")
async def test_lease_batch_is_clamped_to_1_through_50(
    connection: AsyncConnection, world: World
) -> None:
    await _bind(connection, world.a.tenant_id)
    # 60 due job.run events in all (the job's own plus 59 copies).
    await connection.execute(
        text(
            "INSERT INTO outbox_events (id, tenant_id, job_id, event_type, payload, attempts)"
            " SELECT gen_random_uuid(), o.tenant_id, o.job_id, o.event_type, o.payload, 0"
            " FROM outbox_events AS o, generate_series(1, 59)"
        )
    )
    await as_role(connection, WORKER_ROLE)
    lease = "SELECT count(*) FROM lease_outbox_events('probe', :batch, 60, ARRAY['job.run'])"

    too_many = await connection.scalar(text(lease), {"batch": 500})
    none = await connection.scalar(text(lease), {"batch": 0})

    assert (too_many, none) == (50, 1)


async def test_every_table_the_worker_touches_for_a_check_forces_tenant_isolation(
    connection: AsyncConnection,
) -> None:
    tables = [*CHECK_INPUT_TABLES, "context_manifests", "validation_runs"]
    unprotected = (
        await connection.scalars(
            text(
                "SELECT c.relname FROM pg_class AS c WHERE c.relname = ANY(:tables)"
                " AND NOT (c.relrowsecurity AND c.relforcerowsecurity AND EXISTS ("
                "SELECT 1 FROM pg_policies AS p WHERE p.tablename = c.relname"
                " AND p.policyname = 'tenant_isolation'))"
            ),
            {"tables": tables},
        )
    ).all()

    assert unprotected == []


@pytest.fixture
async def run_in_a(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> dict[str, Any]:
    # _creative also stores the brief validation run of tenant A, which the tests copy.
    await _creative(client, auth, world.a)
    row = (
        (
            await connection.execute(
                text(
                    "SELECT creative_id, creative_version, input_kind, state, input_hash,"
                    " policy_bundle_id, policy_bundle_version, study_snapshot_version"
                    " FROM validation_runs WHERE tenant_id = :a"
                ),
                {"a": world.a.tenant_id},
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


async def _insert_run(connection: AsyncConnection, tenant_id: UUID, run: dict[str, Any]) -> None:
    await connection.execute(
        text(
            "INSERT INTO validation_runs (id, tenant_id, creative_id, creative_version,"
            " input_kind, state, input_hash, policy_bundle_id, policy_bundle_version,"
            " study_snapshot_version, result, created_by, judge_model_id, judge_attempts,"
            " reuse_key) VALUES (gen_random_uuid(), :tenant_id, :creative_id,"
            " :creative_version, :input_kind, :state, :input_hash, :policy_bundle_id,"
            " :policy_bundle_version, :study_snapshot_version, '{}'::jsonb, 'worker-test',"
            " 'simulator:judge-v1', 1, repeat('a', 64))"
        ),
        {**run, "tenant_id": tenant_id},
    )


async def _count(connection: AsyncConnection, table: str, tenant_id: UUID) -> int:
    # The table is one of the constants above, never input.
    return int(
        await connection.scalar(
            text(f"SELECT count(*) FROM {table} WHERE tenant_id = :t"),  # noqa: S608
            {"t": tenant_id},
        )
        or 0
    )


async def test_worker_reads_runs_and_policy_rules_of_its_own_tenant_only(
    connection: AsyncConnection, world: World, run_in_a: dict[str, Any]
) -> None:
    await as_role(connection, WORKER_ROLE)
    seen: dict[str, dict[str, int]] = {}
    for name, tenant in (("own", world.a.tenant_id), ("other", world.b.tenant_id), ("none", None)):
        await _bind(connection, tenant)
        seen[name] = {
            table: await _count(connection, table, world.a.tenant_id)
            for table in ("validation_runs", "policy_rules")
        }

    assert seen["own"]["validation_runs"] == 1
    assert seen["own"]["policy_rules"] > 0
    assert seen["other"] == seen["none"] == {"validation_runs": 0, "policy_rules": 0}


async def test_worker_inserts_a_run_for_its_tenant_but_not_another(
    connection: AsyncConnection, world: World, run_in_a: dict[str, Any]
) -> None:
    await as_role(connection, WORKER_ROLE)
    await _bind(connection, world.a.tenant_id)
    await _insert_run(connection, world.a.tenant_id, run_in_a)

    assert await _count(connection, "validation_runs", world.a.tenant_id) == 2

    await _bind(connection, world.b.tenant_id)
    with pytest.raises(DBAPIError, match="row-level security"):
        async with connection.begin_nested():
            await _insert_run(connection, world.a.tenant_id, run_in_a)


@pytest.mark.parametrize(
    "statement",
    ["UPDATE validation_runs SET outcome = 'PASS'", "DELETE FROM validation_runs"],
)
async def test_worker_cannot_change_or_delete_a_stored_run(
    connection: AsyncConnection, world: World, run_in_a: dict[str, Any], statement: str
) -> None:
    await as_role(connection, WORKER_ROLE)
    await _bind(connection, world.a.tenant_id)

    with pytest.raises(DBAPIError, match="permission denied"):
        async with connection.begin_nested():
            await connection.execute(text(statement))
