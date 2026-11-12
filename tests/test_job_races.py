# Real races on committed data with independent connections (simulated provider only).

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import get_settings
from app.db.session import create_engine, create_sessionmaker
from app.main import create_app
from app.services.providers.base import ImageRequest, SubmitResult
from app.services.providers.simulator import SimulatedImageProvider
from app.workers.connector import run_once
from tests.conftest import AuthHeaders, IntegrationEnv, SeededTenant, StaticKeys
from tests.test_job_worker import FakeSink
from tests.test_jobs_api import JOBS, _creative

pytest_plugins = ("tests.test_jobs_api",)  # committed_tenant fixture


class SlowProvider(SimulatedImageProvider):
    # SIMULATED; a slow submit keeps the race window open.

    async def submit(self, request: ImageRequest) -> SubmitResult:
        await asyncio.sleep(0.3)
        return await super().submit(request)


@pytest.fixture
async def http(
    committed_tenant: SeededTenant, signing_key: RSAPrivateKey
) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


async def _post(
    http: httpx.AsyncClient, auth: AuthHeaders, tenant: SeededTenant, creative_id: str, key: str
) -> httpx.Response:
    return await http.post(
        JOBS,
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": key}),
        json={"creative_id": creative_id, "creative_version": 1, "kind": "image_generation"},
    )


async def _scalar(test_env: IntegrationEnv, sql: str, tenant: SeededTenant) -> object:
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.connect() as conn:
        value = await conn.scalar(text(sql), {"t": tenant.tenant_id})
    await engine.dispose()
    return value


async def test_two_workers_on_one_event_submit_once(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
) -> None:
    auth = AuthHeaders(signing_key)
    creative_id = await _creative(http, auth, committed_tenant)
    assert (await _post(http, auth, committed_tenant, creative_id, "race-w-1")).status_code == 202
    settings = get_settings()
    provider, sink = SlowProvider(), FakeSink()
    engines = [create_engine(settings), create_engine(settings)]

    async def worker(sessions: async_sessionmaker[AsyncSession], owner: str) -> int:
        # Keep polling while the other worker may be mid-submit.
        leased = 0
        for _ in range(5):
            leased += await run_once(sessions, {"simulator": provider}, sink, settings, owner=owner)
            await asyncio.sleep(0.1)
        return leased

    try:
        leased = await asyncio.gather(
            *(worker(create_sessionmaker(e), f"worker-{n}") for n, e in enumerate(engines))
        )
    finally:
        for engine in engines:
            await engine.dispose()

    assert sum(leased) == 1
    assert sum(provider.submits.values()) == 1
    assert len(sink.stored) == 1
    state = await _scalar(test_env, "SELECT state FROM jobs WHERE tenant_id = :t", committed_tenant)
    assert state == "SUCCEEDED"


async def test_parallel_admission_respects_the_active_job_limit(
    http: httpx.AsyncClient,
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
) -> None:
    auth = AuthHeaders(signing_key)
    limit = get_settings().generation_max_active_jobs_per_tenant
    creative_id = await _creative(http, auth, committed_tenant)
    for n in range(limit - 1):
        response = await _post(http, auth, committed_tenant, creative_id, f"fill-{n:04d}")
        assert response.status_code == 202

    last_slot = await asyncio.gather(
        *(_post(http, auth, committed_tenant, creative_id, f"last-slot-{n}") for n in range(2))
    )

    assert sorted(r.status_code for r in last_slot) == [202, 429]
    count = await _scalar(
        test_env, "SELECT count(*) FROM jobs WHERE tenant_id = :t", committed_tenant
    )
    assert count == limit
