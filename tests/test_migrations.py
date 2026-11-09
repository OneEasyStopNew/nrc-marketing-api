# Migration behaviour that a schema comparison cannot show: compatibility and safe downgrades.

import asyncio

import httpx
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.main import create_app
from tests.conftest import (
    ROOT,
    WORKER_ROLE,
    AuthHeaders,
    IntegrationEnv,
    SeededTenant,
    StaticKeys,
    World,
    as_role,
)
from tests.test_jobs_api import _creative, _post_job

# The revision before the derived-key change; downgrading to it runs that change's downgrade.
BEFORE_DERIVED_KEYS = "a3c5e7b91d24"


async def _add_asset_event(connection: AsyncConnection, world: World) -> None:
    # An event type that workers before NRC-10 never handled.
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(world.a.tenant_id)}
    )
    await connection.execute(
        text(
            "INSERT INTO outbox_events (id, tenant_id, job_id, event_type, payload, attempts)"
            " SELECT gen_random_uuid(), tenant_id, job_id, 'asset.generated', '{}', 0"
            " FROM outbox_events"
        )
    )


async def test_three_argument_lease_still_works_and_leases_only_job_runs(
    client: httpx.AsyncClient, auth: AuthHeaders, world: World, connection: AsyncConnection
) -> None:
    creative_id = await _creative(client, auth, world.a)
    assert (await _post_job(client, auth, world, creative_id)).status_code == 202
    await _add_asset_event(connection, world)
    await as_role(connection, WORKER_ROLE)

    leased = (
        await connection.execute(text("SELECT * FROM lease_outbox_events('old-worker', 10, 60)"))
    ).all()

    assert [row.event_type for row in leased] == ["job.run"]


async def test_due_index_leads_with_the_event_type(connection: AsyncConnection) -> None:
    definition = await connection.scalar(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_outbox_events_due'")
    )

    assert "(event_type, available_at)" in str(definition)
    assert "WHERE (processed_at IS NULL)" in str(definition)


def _alembic(monkeypatch: pytest.MonkeyPatch, test_env: IntegrationEnv) -> Config:
    monkeypatch.setenv(
        "MIGRATION_DATABASE_URL", test_env.test_migration_database_url.get_secret_value()
    )
    return Config(str(ROOT / "alembic.ini"))


async def _current_revision(test_env: IntegrationEnv) -> str | None:
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.connect() as conn:
        revision = await conn.scalar(text("SELECT version_num FROM alembic_version"))
    await engine.dispose()
    return None if revision is None else str(revision)


async def test_downgrade_refuses_while_an_unfinished_job_holds_a_derived_key(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    auth = AuthHeaders(signing_key)
    world = World(committed_tenant, committed_tenant)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        creative_id = await _creative(http, auth, committed_tenant)
        accepted = await _post_job(http, auth, world, creative_id)
    assert accepted.status_code == 202
    config = _alembic(monkeypatch, test_env)
    head = ScriptDirectory.from_config(config).get_current_head()

    # alembic's env.py runs its own event loop, so it gets a thread of its own.
    with pytest.raises(RuntimeError, match="Refusing to downgrade 8f8284b27c6f"):
        await asyncio.to_thread(command.downgrade, config, BEFORE_DERIVED_KEYS)

    # One transaction for the whole run: nothing was downgraded.
    assert await _current_revision(test_env) == head
