# Generation job API (simulated provider mode): admission, idempotency, limits and isolation.

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import get_settings
from app.db.base import Base
from app.main import create_app
from app.models import PolicyBundle
from scripts.seed_demo import demo_policy_rules
from tests.conftest import AuthHeaders, IntegrationEnv, SeededTenant, StaticKeys, World, seed_tenant

JOBS = "/v1/marketing/creative-jobs"


async def _creative(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    tenant: SeededTenant,
    *,
    prompt: str = "Invite adults with high blood pressure to learn about the study.",
) -> str:
    response = await client.post(
        "/v1/marketing/creatives",
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={
            "study_id": str(tenant.study_id),
            "prompt": prompt,
            "language": "en",
            "channel": "facebook",
            "post_format": "feed_image",
        },
    )
    assert response.status_code == 201, response.text
    creative_id: str = response.json()["id"]
    # Generation needs a stored, passing brief validation of this version (NRC-7 gate).
    validation = await client.post(
        "/v1/marketing/validations",
        headers=auth("creator-a", tenant.tenant_id, **{"Idempotency-Key": uuid4().hex}),
        json={"creative_id": creative_id, "creative_version": 1, "input_kind": "brief"},
    )
    assert validation.status_code == 202, validation.text
    return creative_id


async def _post_job(
    client: httpx.AsyncClient,
    auth: AuthHeaders,
    world: World,
    creative_id: str,
    *,
    key: str | None = None,
    sub: str = "creator-a",
    version: int = 1,
) -> httpx.Response:
    headers = auth(sub, world.a.tenant_id, **{"Idempotency-Key": key or uuid4().hex})
    body = {"creative_id": creative_id, "creative_version": version, "kind": "image_generation"}
    return await client.post(JOBS, headers=headers, json=body)


async def _count(connection: AsyncConnection, tenant_id: UUID, table: str) -> int:
    await connection.execute(
        text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)}
    )
    result = await connection.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608  table is a literal name passed by the calling test, never input
    return int(result.scalar_one())


def _error(response: httpx.Response) -> dict[str, Any]:
    body: dict[str, Any] = response.json()
    return body


async def test_simulated_accept_returns_202_and_duplicate_key_replays_same_job(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, connection: AsyncConnection
) -> None:
    creative_id = await _creative(client, auth, world.a)

    first = await _post_job(client, auth, world, creative_id, key="job-key-0001")
    again = await _post_job(client, auth, world, creative_id, key="job-key-0001")
    polled = await client.get(
        first.headers["Location"], headers=auth("creator-a", world.a.tenant_id)
    )

    assert first.status_code == 202
    assert first.json()["state"] == "QUEUED"
    job_id = first.json()["job_id"]
    assert first.headers["Location"] == f"/v1/marketing/jobs/{job_id}"
    assert "Idempotent-Replayed" not in first.headers
    assert again.status_code == 202
    assert again.json()["job_id"] == job_id
    assert again.headers["Idempotent-Replayed"] == "true"
    assert polled.status_code == 200
    assert polled.json()["state"] == "QUEUED"
    assert polled.json()["creative_id"] == creative_id
    for table in ("jobs", "credit_reservations", "outbox_events"):
        assert await _count(connection, world.a.tenant_id, table) == 1


async def test_same_key_with_changed_body_is_rejected(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    first_creative = await _creative(client, auth, world.a)
    second_creative = await _creative(client, auth, world.a)
    await _post_job(client, auth, world, first_creative, key="job-key-0002")

    response = await _post_job(client, auth, world, second_creative, key="job-key-0002")

    assert response.status_code == 409
    assert _error(response)["code"] == "IDEMPOTENCY_KEY_REUSED"


async def test_missing_idempotency_key_is_rejected(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)

    response = await client.post(
        JOBS,
        headers=auth("creator-a", world.a.tenant_id),
        json={"creative_id": creative_id, "creative_version": 1, "kind": "image_generation"},
    )

    assert response.status_code == 422


async def test_stale_version_returns_412_with_current_version(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)
    edit = await client.patch(
        f"/v1/marketing/creatives/{creative_id}",
        headers=auth("creator-a", world.a.tenant_id, **{"If-Match": '"1"'}),
        json={"current_prompt": "Revised prompt."},
    )
    assert edit.status_code == 200

    response = await _post_job(client, auth, world, creative_id, version=1)

    assert response.status_code == 412
    assert _error(response)["details"] == {"current_version": 2}


async def test_closed_study_is_not_eligible(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, connection: AsyncConnection
) -> None:
    creative_id = await _creative(client, auth, world.a)
    # The study closes after the draft was written: the owner publishes snapshot version 2.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        text(
            "INSERT INTO study_snapshots (id, tenant_id, study_id, version, recruitment_status,"
            " language, facts, source_ref, approved_by, approved_at, is_synthetic)"
            " SELECT gen_random_uuid(), tenant_id, study_id, 2, 'closed', language, facts,"
            " source_ref, approved_by, approved_at, true FROM study_snapshots WHERE id = :id"
        ),
        {"id": world.a.snapshot_id},
    )
    await connection.execute(text("SET LOCAL ROLE marketing_app"))

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 409
    assert _error(response)["code"] == "GENERATION_NOT_ELIGIBLE"
    assert _error(response)["details"] == {"reason": "STUDY_CLOSED"}


async def test_brief_that_failed_validation_is_not_eligible(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, connection: AsyncConnection
) -> None:
    creative_id = await _creative(
        client, auth, world.a, prompt="Guaranteed results! Try this new drug for free treatment."
    )

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 409
    body = _error(response)
    assert (body["code"], body["retryable"]) == ("VALIDATION_NOT_PASSED", False)
    assert body["details"]["status"] == "VALIDATION_FAILED"
    assert body["details"]["validation_run_id"] is not None
    assert await _count(connection, world.a.tenant_id, "jobs") == 0


async def test_pass_under_an_older_policy_is_not_eligible(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders, connection: AsyncConnection
) -> None:
    creative_id = await _creative(client, auth, world.a)  # passes against policy v1
    # A new policy version is approved after the brief passed: it must validate again.
    await connection.execute(text("RESET ROLE"))
    await connection.execute(
        insert(PolicyBundle).values(
            id=uuid4(),
            tenant_id=world.a.tenant_id,
            version=2,
            name="[SYNTHETIC] Policy v2",
            source_ref="synthetic:test-fixture",
            rules=demo_policy_rules(),
            is_synthetic=True,
            approved_by="synthetic-fixture",
            approved_at=datetime(2026, 11, 2, tzinfo=UTC),
        )
    )
    await connection.execute(text("SET LOCAL ROLE marketing_app"))

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 409
    body = _error(response)
    assert body["code"] == "VALIDATION_NOT_PASSED"
    assert body["details"]["status"] == "NOT_VALIDATED"
    assert await _count(connection, world.a.tenant_id, "jobs") == 0


async def test_reviewer_cannot_start_generation(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)

    response = await _post_job(client, auth, world, creative_id, sub="reviewer-a")

    assert response.status_code == 403


async def test_other_tenant_cannot_read_the_job(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)
    accepted = await _post_job(client, auth, world, creative_id)

    response = await client.get(
        accepted.headers["Location"], headers=auth("creator-b", world.b.tenant_id)
    )

    assert response.status_code == 404


async def test_active_job_limit_returns_429(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)
    for _ in range(get_settings().generation_max_active_jobs_per_tenant):
        assert (await _post_job(client, auth, world, creative_id)).status_code == 202

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 429
    assert _error(response)["code"] == "GENERATION_LIMIT_REACHED"
    assert _error(response)["retryable"] is True
    assert _error(response)["details"] == {"reason": "ACTIVE_JOB_LIMIT"}


@pytest.fixture
def credit_ceiling_of_one() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GENERATION_DAILY_CREDIT_CEILING_PER_TENANT", "1")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("credit_ceiling_of_one")
async def test_daily_credit_ceiling_returns_429(
    client: httpx.AsyncClient, world: World, auth: AuthHeaders
) -> None:
    creative_id = await _creative(client, auth, world.a)
    assert (await _post_job(client, auth, world, creative_id)).status_code == 202

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 429
    assert _error(response)["details"] == {"reason": "CREDIT_CEILING"}


@pytest.fixture
async def committed_tenant(test_env: IntegrationEnv) -> AsyncIterator[SeededTenant]:
    # Committed data (no shared test transaction), so parallel requests really race.
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with AsyncSession(engine, expire_on_commit=False) as db:
        seeded = await seed_tenant(
            db, name="Job race tenant", members={"creator-a": ["creator"]}, with_policy=True
        )
        await db.commit()
    try:
        yield seeded
    finally:
        async with AsyncSession(engine) as db:
            for table in reversed(Base.metadata.sorted_tables):
                column = table.c.id if table.name == "tenants" else table.c.tenant_id
                await db.execute(table.delete().where(column == seeded.tenant_id))
            await db.commit()
        await engine.dispose()


async def test_parallel_duplicate_jobs_make_one_job(
    committed_tenant: SeededTenant, signing_key: RSAPrivateKey, test_env: IntegrationEnv
) -> None:
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    auth = AuthHeaders(signing_key)
    headers = auth("creator-a", committed_tenant.tenant_id, **{"Idempotency-Key": "race-job-1"})

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        creative_id = await _creative(http, auth, committed_tenant)
        body = {"creative_id": creative_id, "creative_version": 1, "kind": "image_generation"}
        responses = await asyncio.gather(
            *(http.post(JOBS, headers=headers, json=body) for _ in range(5))
        )

    assert {r.status_code for r in responses} == {202}
    assert len({r.json()["job_id"] for r in responses}) == 1
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.connect() as conn:
        for table in ("jobs", "credit_reservations", "outbox_events"):
            count = await conn.scalar(
                text(f"SELECT count(*) FROM {table} WHERE tenant_id = :t"),  # noqa: S608  table comes from the fixed tuple in the loop above
                {"t": committed_tenant.tenant_id},
            )
            assert count == 1
    await engine.dispose()
