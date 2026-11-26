# Database-level guarantees that hold even if application code forgets a filter.

from typing import Final
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from tests.conftest import World

# Tenant tables without row-level security, and why. Every other table with a tenant_id has it.
TABLES_WITHOUT_RLS: Final = {
    # /me lists a user's tenants before any tenant is chosen, so rows are found by user_sub.
    # The app role may only SELECT, and queries always filter by the token's sub.
    "memberships",
}
# Foreign keys between tenant tables that don't carry tenant_id themselves, and why.
FOREIGN_KEYS_WITHOUT_TENANT: Final = {
    # Same row also has (tenant_id, creative_id) -> creatives, so the version is the same tenant's.
    "fk_jobs_creative_id_creative_versions",
}
INSERT_CREATIVE: Final = text(
    "INSERT INTO creatives (id, tenant_id, study_id, original_prompt, current_version,"
    " created_by) VALUES (:id, :t, :s, 'p', 1, 'creator-a')"
)
INSERT_VERSION: Final = text(
    "INSERT INTO creative_versions (id, tenant_id, creative_id, version, prompt,"
    " study_snapshot_id, language, channel, post_format, created_by)"
    " VALUES (:id, :t, :c, 1, 'p', :snap, 'en', 'facebook', 'feed_image', 'creator-a')"
)
INSERT_MEMBER_STUDY: Final = text(
    "INSERT INTO membership_studies (tenant_id, user_sub, study_id) VALUES (:t, 'creator-a', :s)"
)
# Each tenant table (a public table with a tenant_id column) and whether RLS is forced on it
# with the tenant_isolation policy.
TENANT_TABLE_RLS: Final = """
    SELECT c.relname,
           c.relrowsecurity AND c.relforcerowsecurity AND EXISTS (
               SELECT 1 FROM pg_policies p
               WHERE p.schemaname = 'public' AND p.tablename = c.relname
                 AND p.policyname = 'tenant_isolation')
    FROM pg_class c
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'tenant_id' AND NOT a.attisdropped
    WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'r'
"""
# Each foreign key from one tenant table to another, and whether it maps tenant_id to tenant_id.
TENANT_FOREIGN_KEYS: Final = """
    WITH tenant_tables AS (
        SELECT c.oid
        FROM pg_class c
        JOIN pg_attribute a
          ON a.attrelid = c.oid AND a.attname = 'tenant_id' AND NOT a.attisdropped
        WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'r'
    )
    SELECT k.conname, EXISTS (
               SELECT 1 FROM unnest(k.conkey, k.confkey) AS col(child_col, parent_col)
               JOIN pg_attribute ca ON ca.attrelid = k.conrelid AND ca.attnum = col.child_col
               JOIN pg_attribute pa ON pa.attrelid = k.confrelid AND pa.attnum = col.parent_col
               WHERE ca.attname = 'tenant_id' AND pa.attname = 'tenant_id')
    FROM pg_constraint k
    WHERE k.contype = 'f'
      AND k.conrelid IN (SELECT oid FROM tenant_tables)
      AND k.confrelid IN (SELECT oid FROM tenant_tables)
"""


async def _insert_creative(connection: AsyncConnection, world: World) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    creative_id = uuid4()
    await connection.execute(
        text(
            "INSERT INTO creatives (id, tenant_id, study_id, original_prompt, current_version,"
            " created_by) VALUES (:id, :t, :s, 'p', 1, 'creator-a')"
        ),
        {"id": creative_id, "t": world.a.tenant_id, "s": world.a.study_id},
    )
    await connection.execute(
        text(
            "INSERT INTO creative_versions (id, tenant_id, creative_id, version, prompt,"
            " study_snapshot_id, language, channel, post_format, created_by)"
            " VALUES (:id, :t, :c, 1, 'p', :snap, 'en', 'facebook', 'feed_image', 'creator-a')"
        ),
        {"id": uuid4(), "t": world.a.tenant_id, "c": creative_id, "snap": world.a.snapshot_id},
    )


async def _count(connection: AsyncConnection, tenant: str) -> int:
    await connection.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant})
    return int((await connection.execute(text("SELECT count(*) FROM creatives"))).scalar_one())


async def test_rls_hides_other_tenants_rows_from_unfiltered_queries(
    connection: AsyncConnection, world: World
) -> None:
    await _insert_creative(connection, world)

    assert await _count(connection, str(world.a.tenant_id)) == 1
    assert await _count(connection, str(world.b.tenant_id)) == 0
    assert await _count(connection, "") == 0


async def test_rls_blocks_writing_rows_for_another_tenant(
    connection: AsyncConnection, world: World
) -> None:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.b.tenant_id)}
    )

    with pytest.raises(DBAPIError, match="row-level security"):
        await connection.execute(
            text(
                "INSERT INTO creatives (id, tenant_id, study_id, original_prompt,"
                " current_version, created_by) VALUES (:id, :t, :s, 'p', 1, 'x')"
            ),
            {"id": uuid4(), "t": world.a.tenant_id, "s": world.a.study_id},
        )


async def test_app_role_cannot_rewrite_creative_versions(
    connection: AsyncConnection, world: World
) -> None:
    await _insert_creative(connection, world)

    with pytest.raises(DBAPIError, match="permission denied"):
        await connection.execute(text("UPDATE creative_versions SET prompt = 'changed'"))


async def _as_owner_in(connection: AsyncConnection, tenant_id: UUID) -> None:
    # Leave the app role and act for tenant A: only the foreign key can stop a link to tenant B.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )


async def test_creative_cannot_use_another_tenants_study(
    connection: AsyncConnection, world: World
) -> None:
    await _as_owner_in(connection, world.a.tenant_id)
    row = {"id": uuid4(), "t": world.a.tenant_id, "s": world.b.study_id}

    with pytest.raises(DBAPIError, match="fk_creatives_tenant_id_studies"):
        await connection.execute(INSERT_CREATIVE, row)


async def test_creative_version_cannot_pin_another_tenants_snapshot(
    connection: AsyncConnection, world: World
) -> None:
    await _as_owner_in(connection, world.a.tenant_id)
    creative_id = uuid4()
    await connection.execute(
        INSERT_CREATIVE, {"id": creative_id, "t": world.a.tenant_id, "s": world.a.study_id}
    )
    row = {"id": uuid4(), "t": world.a.tenant_id, "c": creative_id, "snap": world.b.snapshot_id}

    with pytest.raises(DBAPIError, match="fk_creative_versions_tenant_id_study_snapshots"):
        await connection.execute(INSERT_VERSION, row)


async def test_member_cannot_be_given_another_tenants_study(
    connection: AsyncConnection, world: World
) -> None:
    await _as_owner_in(connection, world.a.tenant_id)
    row = {"t": world.a.tenant_id, "s": world.b.study_id}

    with pytest.raises(DBAPIError, match="fk_membership_studies_tenant_id_studies"):
        await connection.execute(INSERT_MEMBER_STUDY, row)


async def test_every_tenant_table_has_forced_row_level_security(
    connection: AsyncConnection,
) -> None:
    # Guards new tables too: a migration that forgets RLS fails here.
    rows = (await connection.execute(text(TENANT_TABLE_RLS))).all()
    unprotected = {name for name, protected in rows if not protected}

    assert {"creatives", "jobs", "validation_runs"} <= {name for name, _ in rows}
    assert unprotected == TABLES_WITHOUT_RLS


async def test_every_reference_between_tenant_tables_carries_tenant_id(
    connection: AsyncConnection,
) -> None:
    # Guards new tables too: a foreign key on id alone could link rows of two tenants.
    rows = (await connection.execute(text(TENANT_FOREIGN_KEYS))).all()
    without_tenant = {name for name, carries_tenant in rows if not carries_tenant}

    assert "fk_creatives_tenant_id_studies" in {name for name, _ in rows}
    assert without_tenant == FOREIGN_KEYS_WITHOUT_TENANT
