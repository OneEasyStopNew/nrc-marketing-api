# The worker's database role and the lease function: who may do what, checked in the database.

from pathlib import Path
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.workers.connector import WorkerConfigError, worker_database_url
from tests.conftest import APP_ROLE, WORKER_ROLE, AuthHeaders, World, as_role
from tests.test_jobs_api import _creative, _post_job

LEASE_4 = "lease_outbox_events(text, integer, integer, text[])"
LEASE_3 = "lease_outbox_events(text, integer, integer)"
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
    lease_4 = "SELECT * FROM lease_outbox_events('probe', 1, 5, ARRAY['job.run'])"
    lease_3 = "SELECT * FROM lease_outbox_events('probe', 1, 5)"
    denied: dict[str, tuple[bool, bool]] = {}
    for role in ROLES:
        await as_role(connection, role)
        denied[role] = (await _denied(connection, lease_4), await _denied(connection, lease_3))

    assert denied == {APP_ROLE: (True, True), WORKER_ROLE: (False, False)}


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


@pytest.mark.parametrize("function", [LEASE_4, LEASE_3])
async def test_lease_function_checklist(connection: AsyncConnection, function: str) -> None:
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
            {"f": function},
        )
    ).one()

    assert row.proconfig == ["search_path=pg_catalog, public, pg_temp"]
    assert row.result == ("TABLE(event_id uuid, tenant_id uuid, job_id uuid, event_type text)")
    assert (row.app, row.worker) == (False, True)
    # No PUBLIC entry ("=X/owner") in the ACL.
    assert not any(entry.startswith("=") for entry in row.acl.split(","))
    if function == LEASE_4:
        assert row.prosecdef
        assert "FOR UPDATE SKIP LOCKED" in row.prosrc
    else:
        # The wrapper runs as its caller and only delegates.
        assert not row.prosecdef


async def test_outbox_lease_policy_is_scoped_to_the_function_owner(
    connection: AsyncConnection,
) -> None:
    owner = await connection.scalar(
        text("SELECT pg_get_userbyid(proowner) FROM pg_proc WHERE oid = CAST(:f AS regprocedure)"),
        {"f": LEASE_4},
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
