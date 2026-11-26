# Migration details that a schema comparison (alembic check) does not cover.

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any, Final
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import URL, make_url, text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from tests.conftest import ROOT, IntegrationEnv

# The foundation schema, the first one deployed. Data written there must survive every upgrade.
BASELINE: Final = "de579cd5c63a"
SCRATCH_SUFFIX: Final = "_upgrade"
BASELINE_TABLES: Final = (
    "tenants",
    "memberships",
    "studies",
    "study_snapshots",
    "membership_studies",
    "creatives",
    "creative_versions",
    "idempotency_records",
)


async def test_due_index_leads_with_the_event_type(connection: AsyncConnection) -> None:
    definition = await connection.scalar(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_outbox_events_due'")
    )

    assert "(event_type, available_at)" in str(definition)
    assert "WHERE (processed_at IS NULL)" in str(definition)


def _migrate(owner_url: URL, revision: str) -> None:
    # Never let this point anywhere but the scratch database: a downgrade here drops tables.
    assert str(owner_url.database).endswith(SCRATCH_SUFFIX)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIGRATION_DATABASE_URL", owner_url.render_as_string(hide_password=False))
        command.upgrade(Config(str(ROOT / "alembic.ini")), revision)


@pytest.fixture
async def scratch_database(test_env: IntegrationEnv) -> AsyncIterator[URL]:
    # A database of its own, so a failure here can't leave the shared test database half-migrated.
    owner_url = make_url(test_env.test_migration_database_url.get_secret_value())
    name = f"{owner_url.database}{SCRATCH_SUFFIX}"
    # Database names can't be bound parameters; this one comes from test config, checked here.
    assert re.fullmatch(r"[a-z0-9_]+", name, flags=re.ASCII)
    drop = text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin = create_async_engine(owner_url, poolclass=NullPool, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            await conn.execute(drop)
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield owner_url.set(database=name)
    finally:
        async with admin.connect() as conn:
            await conn.execute(drop)
        await admin.dispose()


async def _seed_baseline(conn: AsyncConnection) -> dict[str, Any]:
    ids = {"tenant": uuid4(), "study": uuid4(), "snapshot": uuid4(), "creative": uuid4()}
    await conn.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(ids["tenant"])}
    )
    statements = (
        "INSERT INTO tenants (id, name) VALUES (:tenant, 'Baseline tenant')",
        "INSERT INTO memberships (tenant_id, user_sub, roles)"
        " VALUES (:tenant, 'creator-baseline', ARRAY['creator'])",
        "INSERT INTO studies (id, tenant_id, title) VALUES (:study, :tenant, '[SYNTHETIC] Study')",
        "INSERT INTO study_snapshots (id, tenant_id, study_id, version, recruitment_status,"
        " language, facts, source_ref, approved_by, approved_at, is_synthetic)"
        " VALUES (:snapshot, :tenant, :study, 1, 'recruiting', 'en',"
        " '{\"claims\": [\"Approved claim\"]}', 'synthetic:baseline', 'approver',"
        " '2026-11-04T10:00:00+00:00', true)",
        "INSERT INTO membership_studies (tenant_id, user_sub, study_id)"
        " VALUES (:tenant, 'creator-baseline', :study)",
        "INSERT INTO creatives (id, tenant_id, study_id, original_prompt, current_version,"
        " created_by)"
        " VALUES (:creative, :tenant, :study, 'Baseline prompt', 1, 'creator-baseline')",
        "INSERT INTO creative_versions (id, tenant_id, creative_id, version, prompt, caption,"
        " study_snapshot_id, language, channel, post_format, created_by)"
        " VALUES (gen_random_uuid(), :tenant, :creative, 1, 'Baseline prompt', 'Caption',"
        " :snapshot, 'en', 'facebook', 'feed_image', 'creator-baseline')",
        "INSERT INTO idempotency_records (tenant_id, route, key, request_hash, resource_id)"
        " VALUES (:tenant, 'POST /creatives', 'baseline-key', repeat('a', 64), :creative)",
    )
    for statement in statements:
        await conn.execute(text(statement), ids)
    return ids


async def _rows(conn: AsyncConnection) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for table in BASELINE_TABLES:
        # Table names are the constants above, never input.
        result = await conn.execute(text(f"SELECT to_jsonb(t) FROM {table} t"))  # noqa: S608
        rows[table] = list(result.scalars())
    return rows


async def test_upgrade_from_the_baseline_keeps_existing_records(
    scratch_database: URL, test_env: IntegrationEnv
) -> None:
    await asyncio.to_thread(_migrate, scratch_database, BASELINE)
    owner = create_async_engine(scratch_database, poolclass=NullPool)
    try:
        async with owner.begin() as conn:
            ids = await _seed_baseline(conn)
            before = await _rows(conn)

        await asyncio.to_thread(_migrate, scratch_database, "head")

        async with owner.begin() as conn:
            after = await _rows(conn)
    finally:
        await owner.dispose()

    for table, (old_row,) in before.items():
        (new_row,) = after[table]
        # Upgrades may add columns, but every existing column keeps exactly its old value.
        assert old_row.items() <= new_row.items(), table
    assert after["memberships"][0]["display_name"] is None

    # The API's role still reaches the old rows through its grants and row-level security.
    app_url = make_url(test_env.test_database_url.get_secret_value()).set(
        database=scratch_database.database
    )
    app = create_async_engine(app_url, poolclass=NullPool)
    try:
        async with app.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(ids["tenant"])}
            )
            prompt = await conn.scalar(text("SELECT prompt FROM creative_versions"))
    finally:
        await app.dispose()
    assert prompt == "Baseline prompt"
