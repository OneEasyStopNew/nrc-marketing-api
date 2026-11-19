# The media_uploads table, its lease function and the assets lineage check, checked in the
# database (NRC-3). Service and API behaviour are tested with the upload endpoints.

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection

from app.models.enums import UploadState
from tests.conftest import APP_ROLE, WORKER_ROLE, SeededTenant, World, as_role

LEASE = "lease_media_uploads(text, integer, integer)"
ROLES = (APP_ROLE, WORKER_ROLE)
CHECKSUM = "0" * 64


async def _bind(connection: AsyncConnection, tenant_id: UUID | None) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"),
        {"t": "" if tenant_id is None else str(tenant_id)},
    )


async def _upload(
    connection: AsyncConnection,
    tenant: SeededTenant,
    state: UploadState,
    *,
    due_in: timedelta | None = timedelta(0),
) -> UUID:
    # Inserted as the owner under the tenant's binding (FORCE RLS applies to the owner too).
    upload_id = uuid4()
    await _bind(connection, tenant.tenant_id)
    await connection.execute(
        text(
            "INSERT INTO media_uploads (id, tenant_id, study_id, purpose, state, name,"
            " declared_mime_type, declared_size_bytes, declared_checksum_sha256, bucket, s3_key,"
            " upload_url_expires_at, identifiable_person, created_by, available_at)"
            " VALUES (:id, :tenant, :study, 'creative_media', :state, 'synthetic.png',"
            " 'image/png', 1024, :checksum, 'synthetic-bucket', :key,"
            " now() + interval '10 minutes', 'no', 'creator-a',"
            " CASE WHEN CAST(:due AS interval) IS NULL THEN NULL"
            " ELSE clock_timestamp() + CAST(:due AS interval) END)"
        ),
        {
            "id": upload_id,
            "tenant": tenant.tenant_id,
            "study": tenant.study_id,
            "state": state.value,
            "checksum": CHECKSUM,
            "key": f"uploads/2026/11/{upload_id}.png",
            "due": due_in,
        },
    )
    return upload_id


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


@pytest.mark.parametrize("role", ROLES)
async def test_no_role_reads_another_tenants_uploads(
    connection: AsyncConnection, world: World, role: str
) -> None:
    await _upload(connection, world.a, UploadState.QUARANTINED)
    await as_role(connection, role)
    visible: dict[str, int] = {}
    for name, tenant in (("own", world.a.tenant_id), ("other", world.b.tenant_id), ("none", None)):
        await _bind(connection, tenant)
        visible[name] = int(
            await connection.scalar(
                text("SELECT count(*) FROM media_uploads WHERE tenant_id = :a"),
                {"a": world.a.tenant_id},
            )
            or 0
        )

    assert visible == {"own": 1, "other": 0, "none": 0}


@pytest.mark.usefixtures("world")
async def test_upload_privileges_per_role(connection: AsyncConnection) -> None:
    # The API creates and moves uploads; the worker only moves them; nobody deletes.
    statements = {
        "select": "SELECT count(*) FROM media_uploads",
        "update": "UPDATE media_uploads SET updated_at = updated_at",
        "insert": "INSERT INTO media_uploads (id) VALUES (gen_random_uuid())",
        "delete": "DELETE FROM media_uploads",
    }
    denied: dict[str, dict[str, bool]] = {}
    for role in ROLES:
        await as_role(connection, role)
        denied[role] = {name: await _denied(connection, sql) for name, sql in statements.items()}

    assert denied[APP_ROLE] == {"select": False, "update": False, "insert": False, "delete": True}
    assert denied[WORKER_ROLE] == {
        "select": False,
        "update": False,
        "insert": True,
        "delete": True,
    }


async def test_media_upload_lease_function_checklist(connection: AsyncConnection) -> None:
    # Same review list as lease_outbox_events: fixed search_path, worker-only EXECUTE, ids out.
    row = (
        await connection.execute(
            text(
                "SELECT p.proconfig, p.prosecdef, pg_get_function_result(p.oid) AS result,"
                " p.prosrc,"
                " has_function_privilege('marketing_app', p.oid, 'EXECUTE') AS app,"
                " has_function_privilege('marketing_worker', p.oid, 'EXECUTE') AS worker,"
                " coalesce(array_to_string(p.proacl, ','), '') AS acl"
                " FROM pg_proc AS p WHERE p.oid = CAST(:f AS regprocedure)"
            ),
            {"f": LEASE},
        )
    ).one()

    assert row.proconfig == ["search_path=pg_catalog, public, pg_temp"]
    assert row.result == "TABLE(upload_id uuid, tenant_id uuid)"
    assert (row.app, row.worker) == (False, True)
    assert not any(entry.startswith("=") for entry in row.acl.split(","))
    assert row.prosecdef
    assert "FOR UPDATE SKIP LOCKED" in row.prosrc


async def test_lease_policy_is_scoped_to_the_function_owner(connection: AsyncConnection) -> None:
    owner = await connection.scalar(
        text("SELECT pg_get_userbyid(proowner) FROM pg_proc WHERE oid = CAST(:f AS regprocedure)"),
        {"f": LEASE},
    )
    roles = await connection.scalar(
        text(
            "SELECT array_to_string(roles::regrole[], ',') FROM pg_policies"
            " WHERE tablename = 'media_uploads' AND policyname = 'media_upload_lease'"
        )
    )

    assert roles == owner


async def _lease(connection: AsyncConnection) -> set[UUID]:
    await as_role(connection, WORKER_ROLE)
    rows = (
        await connection.execute(text("SELECT upload_id FROM lease_media_uploads('w1', 50, 30)"))
    ).all()
    await connection.execute(text("RESET ROLE"))
    return {row.upload_id for row in rows}


async def test_lease_takes_only_due_inspections_across_tenants(
    connection: AsyncConnection, world: World
) -> None:
    due_a = await _upload(connection, world.a, UploadState.QUARANTINED)
    due_b = await _upload(connection, world.b, UploadState.QUARANTINED)
    await _upload(connection, world.a, UploadState.QUARANTINED, due_in=timedelta(minutes=5))
    await _upload(connection, world.a, UploadState.PENDING_UPLOAD, due_in=None)
    await _upload(connection, world.a, UploadState.READY)
    await _upload(connection, world.a, UploadState.REJECTED)

    assert await _lease(connection) == {due_a, due_b}
    # Leased rows are not handed out again while the lease holds.
    assert await _lease(connection) == set()


async def test_expired_lease_is_leased_again_and_counted_as_a_failure(
    connection: AsyncConnection, world: World
) -> None:
    upload_id = await _upload(connection, world.a, UploadState.SCANNING)
    assert await _lease(connection) == {upload_id}
    # The worker crashed mid-scan: its lease runs out without being cleared.
    await _bind(connection, world.a.tenant_id)
    await connection.execute(
        text(
            "UPDATE media_uploads SET leased_until = clock_timestamp() - interval '1 second'"
            " WHERE id = :id"
        ),
        {"id": upload_id},
    )

    assert await _lease(connection) == {upload_id}
    await _bind(connection, world.a.tenant_id)
    counts = (
        await connection.execute(
            text("SELECT attempts, failed_attempts FROM media_uploads WHERE id = :id"),
            {"id": upload_id},
        )
    ).one()
    assert (counts.attempts, counts.failed_attempts) == (2, 1)


async def _insert_asset(
    connection: AsyncConnection, tenant: SeededTenant, *, asset_id: UUID, **columns: object
) -> None:
    values: dict[str, object] = {
        "id": asset_id,
        "tenant_id": tenant.tenant_id,
        "bucket": "synthetic-bucket",
        "s3_key": f"uploads/2026/11/{asset_id}.png",
        "mime_type": "image/png",
        "width_px": 10,
        "height_px": 10,
        "size_bytes": 1024,
        "checksum_sha256": CHECKSUM,
        "created_by": "creator-a",
        **columns,
    }
    names = ", ".join(values)
    params = ", ".join(f":{name}" for name in values)
    await _bind(connection, tenant.tenant_id)
    savepoint = await connection.begin_nested()
    try:
        await connection.execute(text(f"INSERT INTO assets ({names}) VALUES ({params})"), values)  # noqa: S608  column names are this test's literals
    except Exception:
        await savepoint.rollback()
        raise
    await savepoint.commit()


async def test_an_upload_that_passed_can_become_an_asset(
    connection: AsyncConnection, world: World
) -> None:
    upload_id = await _upload(connection, world.a, UploadState.READY)

    await _insert_asset(
        connection,
        world.a,
        asset_id=upload_id,
        origin="uploaded",
        upload_id=upload_id,
        study_id=world.a.study_id,
        name="synthetic.png",
    )

    assert (
        await connection.scalar(text("SELECT origin FROM assets WHERE id = :id"), {"id": upload_id})
        == "uploaded"
    )


@pytest.mark.parametrize(
    "case",
    [
        "generated-without-job-or-creative",
        "uploaded-without-upload",
        "uploaded-without-study",
        "uploaded-id-differs-from-upload",
    ],
)
async def test_asset_lineage_must_match_its_origin(
    connection: AsyncConnection, world: World, case: str
) -> None:
    upload_id = await _upload(connection, world.a, UploadState.READY)
    study = world.a.study_id
    cases: dict[str, tuple[UUID, dict[str, object]]] = {
        "generated-without-job-or-creative": (uuid4(), {"origin": "generated"}),
        "uploaded-without-upload": (upload_id, {"origin": "uploaded", "study_id": study}),
        "uploaded-without-study": (upload_id, {"origin": "uploaded", "upload_id": upload_id}),
        "uploaded-id-differs-from-upload": (
            uuid4(),
            {"origin": "uploaded", "upload_id": upload_id, "study_id": study},
        ),
    }
    asset_id, columns = cases[case]

    with pytest.raises(IntegrityError, match="ck_assets_lineage_matches_origin"):
        await _insert_asset(connection, world.a, asset_id=asset_id, **columns)
