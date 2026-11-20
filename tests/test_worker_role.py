# The worker's database role and the lease function: who may do what, checked in the database.

from pathlib import Path
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.workers.common import WorkerConfigError, worker_database_url
from tests.conftest import APP_ROLE, WORKER_ROLE, AuthHeaders, World, _creative, _post_job, as_role

LEASE = "lease_outbox_events(text, integer, integer, text[])"
ROLES = (APP_ROLE, WORKER_ROLE)


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
        # The media worker (NRC-3) moves uploads through inspection; the API creates them.
        "media_uploads": {"SELECT", "UPDATE"},
        # The policy worker (NRC-4) reads the graph, writes clauses, embeddings and builds, and
        # records audit events; its UPDATEs are column-level (next test).
        "policy_bundles": {"SELECT"},
        "policy_nodes": {"SELECT"},
        "policy_edges": {"SELECT"},
        "policy_rules": {"SELECT"},
        "policy_rule_wordings": {"SELECT"},
        "policy_sources": {"SELECT"},
        "policy_source_clauses": {"SELECT", "INSERT"},
        "policy_node_embeddings": {"SELECT", "INSERT"},
        "policy_index_builds": {"SELECT", "INSERT", "UPDATE"},
        "policy_audit_events": {"INSERT"},
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


async def test_worker_updates_only_the_queue_columns_of_bundles_and_sources(
    connection: AsyncConnection,
) -> None:
    # Column-level UPDATE: the worker moves work through its queue and changes nothing else (not
    # the rules, the approver or the version).
    lease = {"available_at", "leased_until", "lease_owner", "attempts", "failed_attempts"}
    expected = {
        "policy_bundles": {"lifecycle_state", "updated_at", "last_error_code", *lease},
        "policy_sources": {
            "extraction_state",
            "extraction_error_code",
            "extracted_at",
            "page_count",
            *lease,
        },
    }
    granted: dict[str, set[str]] = {}
    for table in expected:
        rows = await connection.scalars(
            text(
                "SELECT a.attname FROM pg_attribute AS a WHERE a.attrelid = CAST(:t AS regclass)"
                " AND a.attnum > 0 AND NOT a.attisdropped"
                " AND has_column_privilege('marketing_worker', a.attrelid, a.attnum, 'UPDATE')"
            ),
            {"t": table},
        )
        granted[table] = set(rows.all())

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
