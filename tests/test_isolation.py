# Database-level guarantees that hold even if application code forgets a filter.

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from tests.conftest import World


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
