# Concurrent requests against committed data (no shared test transaction).

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.db.base import Base
from app.main import create_app
from app.models import Creative
from tests.conftest import AuthHeaders, IntegrationEnv, SeededTenant, StaticKeys, seed_tenant


@pytest.fixture
async def committed_tenant(test_env: IntegrationEnv) -> AsyncIterator[SeededTenant]:
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with AsyncSession(engine, expire_on_commit=False) as db:
        seeded = await seed_tenant(
            db, name="Concurrency tenant", members={"racer": ["creator"]}, with_policy=False
        )
        await db.commit()
    try:
        yield seeded
    finally:
        async with AsyncSession(engine) as db:
            # Children before parents; every table except tenants carries tenant_id.
            for table in reversed(Base.metadata.sorted_tables):
                if table.name == "provider_spend":
                    # Not tenant data (one counter per provider).
                    continue
                column = table.c.id if table.name == "tenants" else table.c.tenant_id
                await db.execute(table.delete().where(column == seeded.tenant_id))
            await db.commit()
        await engine.dispose()


async def test_parallel_duplicate_creates_make_one_creative(
    committed_tenant: SeededTenant, signing_key: RSAPrivateKey, test_env: IntegrationEnv
) -> None:
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    headers = AuthHeaders(signing_key)(
        "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "parallel-key-1"}
    )
    body = {
        "study_id": str(committed_tenant.study_id),
        "prompt": "Double-clicked save.",
        "language": "en",
        "channel": "facebook",
        "post_format": "feed_image",
    }

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        responses = await asyncio.gather(
            *(http.post("/v1/marketing/creatives", headers=headers, json=body) for _ in range(5))
        )

    assert {r.status_code for r in responses} == {201}
    assert len({r.json()["id"] for r in responses}) == 1
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with AsyncSession(engine) as db:
        count = await db.scalar(
            select(func.count())
            .select_from(Creative)
            .where(Creative.tenant_id == committed_tenant.tenant_id)
        )
    await engine.dispose()
    assert count == 1
