# Generation job API (simulated provider mode): admission, idempotency, limits and isolation.

import asyncio
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_signing_keys
from app.core.config import get_settings
from app.core.errors import ValidationNotPassedError
from app.main import create_app
from app.services import eligibility
from tests.conftest import (
    JOBS,
    AuthHeaders,
    IntegrationEnv,
    SeededTenant,
    StaticKeys,
    World,
    _creative,
    _post_job,
)


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
    assert _error(response)["details"] == {"reason": "STUDY_NOT_RECRUITING"}


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


async def test_validation_gate_refusal_is_a_409_and_queues_nothing(
    client: httpx.AsyncClient,
    world: World,
    auth: AuthHeaders,
    connection: AsyncConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creative_id = await _creative(client, auth, world.a)
    calls: list[tuple[UUID, UUID, int]] = []

    # Stands in for Dinesh's gate (NRC-7), which has the same signature as the stub.
    async def not_passed(
        session: AsyncSession, tenant_id: UUID, creative: UUID, creative_version: int
    ) -> None:
        calls.append((tenant_id, creative, creative_version))
        raise ValidationNotPassedError(details={"validation_run_id": "run-1", "status": "FAILED"})

    monkeypatch.setattr(eligibility, "assert_generation_allowed", not_passed)

    response = await _post_job(client, auth, world, creative_id)

    assert response.status_code == 409
    body = _error(response)
    assert (body["code"], body["retryable"]) == ("VALIDATION_NOT_PASSED", False)
    assert body["details"] == {"validation_run_id": "run-1", "status": "FAILED"}
    assert calls == [(world.a.tenant_id, UUID(creative_id), 1)]
    assert await _count(connection, world.a.tenant_id, "jobs") == 0
