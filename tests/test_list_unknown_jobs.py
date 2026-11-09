# The operator listing of UNKNOWN jobs, against committed data in its read-only session.

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.main import create_app
from scripts.list_unknown_jobs import list_unknown_jobs, read_only_engine
from tests.conftest import (
    AuthHeaders,
    IntegrationEnv,
    SeededTenant,
    StaticKeys,
    World,
    _creative,
    _post_job,
)


async def _accept_jobs(tenant: SeededTenant, signing_key: RSAPrivateKey, count: int) -> list[str]:
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    auth = AuthHeaders(signing_key)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        creative_id = await _creative(http, auth, tenant)
        responses = [
            await _post_job(http, auth, World(tenant, tenant), creative_id) for _ in range(count)
        ]
    return [str(response.json()["job_id"]) for response in responses]


async def test_lists_only_unknown_jobs_the_worker_stopped_on(
    committed_tenant: SeededTenant, signing_key: RSAPrivateKey, test_env: IntegrationEnv
) -> None:
    stopped, reconciling = await _accept_jobs(committed_tenant, signing_key, 2)
    url = test_env.test_migration_database_url.get_secret_value()
    engine = create_async_engine(url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE jobs SET state = 'UNKNOWN' WHERE tenant_id = :t"),
            {"t": committed_tenant.tenant_id},
        )
        await conn.execute(
            text("UPDATE outbox_events SET processed_at = now() WHERE job_id = :j"),
            {"j": stopped},
        )
    await engine.dispose()

    lines = await list_unknown_jobs(url)

    mine = [line for line in lines if str(committed_tenant.tenant_id) in line]
    assert mine == [f"tenant={committed_tenant.tenant_id} job={stopped}"]
    assert all(reconciling not in line for line in lines)


async def test_listing_session_is_read_only(test_env: IntegrationEnv) -> None:
    engine = read_only_engine(test_env.test_migration_database_url.get_secret_value())

    async with engine.connect() as conn:
        with pytest.raises(DBAPIError, match="read-only transaction"):
            await conn.execute(text("UPDATE tenants SET name = name"))
    await engine.dispose()
