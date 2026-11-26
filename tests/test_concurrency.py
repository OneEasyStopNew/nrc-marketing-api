# Concurrent requests against committed data (no shared test transaction).

import asyncio
import hashlib
import io
from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import FastAPI
from PIL import Image
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_optional_storage, get_signing_keys, get_storage
from app.core.config import get_settings
from app.db.base import Base
from app.main import create_app
from app.models import Creative
from app.services.storage import ObjectHead, ObjectStorage
from tests.conftest import (
    BUCKET,
    ENDPOINT,
    AuthHeaders,
    IntegrationEnv,
    SeededTenant,
    StaticKeys,
    needs_seaweedfs,
    seed_tenant,
)

UPLOADS = "/v1/marketing/assets/uploads"


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


def _upload_app(signing_key: RSAPrivateKey, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    # Local-only values that open the SeaweedFS container, not AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "localdev")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "localdev-password")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    storage = ObjectStorage(
        get_settings().model_copy(
            update={
                "aws_s3_bucket": BUCKET,
                "aws_endpoint_url_s3": ENDPOINT,
                "s3_force_path_style": True,
                "aws_profile": None,
            }
        )
    )
    app = create_app()
    app.dependency_overrides[get_signing_keys] = lambda: StaticKeys(signing_key)
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_optional_storage] = lambda: storage
    return app


def _upload_body(tenant: SeededTenant, content: bytes) -> dict[str, object]:
    return {
        "study_id": str(tenant.study_id),
        "name": "[SYNTHETIC] photo.png",
        "mime_type": "image/png",
        "size_bytes": len(content),
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "authorization": {"identifiable_person": "no", "permitted_channels": ["facebook"]},
    }


def _synthetic_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 200, 200)).save(buffer, "PNG")
    return buffer.getvalue()


async def _owner_count(test_env: IntegrationEnv, statement: str, tenant: SeededTenant) -> int:
    # As the table owner (outside RLS), counting only this tenant's rows.
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.connect() as conn:
        count = await conn.scalar(text(statement), {"t": tenant.tenant_id})
    await engine.dispose()
    return int(count or 0)


async def test_parallel_duplicate_upload_creates_make_one_upload(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Signing a URL needs no SeaweedFS; only the PUT does.
    app = _upload_app(signing_key, monkeypatch)
    headers = AuthHeaders(signing_key)(
        "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "parallel-upload-1"}
    )
    body = _upload_body(committed_tenant, _synthetic_png())

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        responses = await asyncio.gather(
            *(http.post(UPLOADS, headers=headers, json=body) for _ in range(5))
        )

    assert {r.status_code for r in responses} == {201}
    assert len({r.json()["asset_id"] for r in responses}) == 1
    count = await _owner_count(
        test_env, "SELECT count(*) FROM media_uploads WHERE tenant_id = :t", committed_tenant
    )
    assert count == 1


async def test_parallel_new_uploads_cannot_both_take_the_last_pending_slot(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cap check is serialised per tenant, like generation's active-job limit.
    app = _upload_app(signing_key, monkeypatch)
    one_slot = get_settings().model_copy(update={"upload_max_pending_per_tenant": 1})
    app.dependency_overrides[get_settings] = lambda: one_slot
    auth = AuthHeaders(signing_key)
    body = _upload_body(committed_tenant, _synthetic_png())

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        responses = await asyncio.gather(
            *(
                http.post(
                    UPLOADS,
                    headers=auth(
                        "racer", committed_tenant.tenant_id, **{"Idempotency-Key": f"slot-race-{n}"}
                    ),
                    json=body,
                )
                for n in range(5)
            )
        )

    assert sorted(r.status_code for r in responses) == [201, 429, 429, 429, 429]
    count = await _owner_count(
        test_env, "SELECT count(*) FROM media_uploads WHERE tenant_id = :t", committed_tenant
    )
    assert count == 1


@needs_seaweedfs
async def test_parallel_completes_queue_one_inspection(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _upload_app(signing_key, monkeypatch)
    auth = AuthHeaders(signing_key)
    content = _synthetic_png()

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        created = await http.post(
            UPLOADS,
            headers=auth(
                "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "parallel-complete-1"}
            ),
            json=_upload_body(committed_tenant, content),
        )
        assert created.status_code == 201, created.text
        target = created.json()["upload"]
        async with httpx.AsyncClient() as s3:
            stored = await s3.request(
                target["method"], target["url"], content=content, headers=target["headers"]
            )
        assert stored.status_code == 200, stored.text
        asset_id = created.json()["asset_id"]
        responses = await asyncio.gather(
            *(
                http.post(
                    f"/v1/marketing/assets/{asset_id}/complete",
                    headers=auth("racer", committed_tenant.tenant_id),
                )
                for _ in range(5)
            )
        )

    assert {r.status_code for r in responses} == {202}
    assert {r.json()["status"] for r in responses} == {"QUARANTINED"}
    # One transition: the upload is queued once, and nothing leased or retried it yet.
    count = await _owner_count(
        test_env,
        "SELECT count(*) FROM media_uploads WHERE tenant_id = :t AND state = 'QUARANTINED'"
        " AND completed_at IS NOT NULL AND attempts = 0",
        committed_tenant,
    )
    assert count == 1


# What UPLOAD_MAX_PENDING_PER_TENANT counts (app.services.uploads._in_progress).
_IN_PROGRESS = (
    "SELECT count(*) FROM media_uploads WHERE tenant_id = :t AND (state IN ('QUARANTINED',"
    " 'SCANNING') OR (state = 'PENDING_UPLOAD' AND upload_url_expires_at > now()))"
)


async def _owner_execute(test_env: IntegrationEnv, statement: str, **params: object) -> None:
    # As the table owner, committed, so the API's own connections see it.
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    async with engine.begin() as conn:
        await conn.execute(text(statement), params)
    await engine.dispose()


def _stored_as_declared(monkeypatch: pytest.MonkeyPatch, content: bytes) -> None:
    # HEAD reports the declared file, so complete needs no PUT; the race is in the database.
    async def head(self: ObjectStorage, key: str, *, version_id: str | None = None) -> ObjectHead:
        return ObjectHead(
            size_bytes=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
            version_id="v1",
            content_type="image/png",
        )

    monkeypatch.setattr(ObjectStorage, "head", head)


def _one_slot_app(signing_key: RSAPrivateKey, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = _upload_app(signing_key, monkeypatch)
    one_slot = get_settings().model_copy(update={"upload_max_pending_per_tenant": 1})
    app.dependency_overrides[get_settings] = lambda: one_slot
    return app


async def _expire_link(test_env: IntegrationEnv, asset_id: str) -> None:
    await _owner_execute(
        test_env,
        "UPDATE media_uploads SET upload_url_expires_at = now() - interval '1 second'"
        " WHERE id = CAST(:id AS uuid)",
        id=asset_id,
    )


async def test_parallel_create_and_complete_of_an_expired_upload_share_the_last_slot(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (review, api#23): completing an expired upload skipped admission.
    app = _one_slot_app(signing_key, monkeypatch)
    auth = AuthHeaders(signing_key)
    content = _synthetic_png()
    _stored_as_declared(monkeypatch, content)

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        created = await http.post(
            UPLOADS,
            headers=auth(
                "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "expired-old-1"}
            ),
            json=_upload_body(committed_tenant, content),
        )
        assert created.status_code == 201, created.text
        asset_id = created.json()["asset_id"]
        await _expire_link(test_env, asset_id)

        completed, *new = await asyncio.gather(
            http.post(
                f"/v1/marketing/assets/{asset_id}/complete",
                headers=auth("racer", committed_tenant.tenant_id),
            ),
            *(
                http.post(
                    UPLOADS,
                    headers=auth(
                        "racer",
                        committed_tenant.tenant_id,
                        **{"Idempotency-Key": f"last-slot-new-{n}"},
                    ),
                    json=_upload_body(committed_tenant, content),
                )
                for n in range(4)
            ),
        )

    # Exactly one request gets the slot; every other one is refused, none fails.
    codes = [completed.status_code, *(r.status_code for r in new)]
    assert sorted(codes) in ([201, 429, 429, 429, 429], [202, 429, 429, 429, 429]), codes
    assert await _owner_count(test_env, _IN_PROGRESS, committed_tenant) == 1


async def test_parallel_reissue_and_completes_of_an_expired_upload_stay_under_the_cap(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _one_slot_app(signing_key, monkeypatch)
    auth = AuthHeaders(signing_key)
    content = _synthetic_png()
    _stored_as_declared(monkeypatch, content)
    headers = auth("racer", committed_tenant.tenant_id)

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        created = await http.post(
            UPLOADS,
            headers=auth(
                "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "expired-old-2"}
            ),
            json=_upload_body(committed_tenant, content),
        )
        assert created.status_code == 201, created.text
        asset_id = created.json()["asset_id"]
        await _expire_link(test_env, asset_id)

        responses = await asyncio.gather(
            http.post(f"/v1/marketing/assets/{asset_id}/upload-url", headers=headers),
            *(
                http.post(f"/v1/marketing/assets/{asset_id}/complete", headers=headers)
                for _ in range(3)
            ),
        )

    # The reissue either revives the upload first or finds it already completed (409).
    assert responses[0].status_code in {200, 409}, responses[0].text
    assert {r.status_code for r in responses[1:]} == {202}
    assert {r.json()["status"] for r in responses[1:]} == {"QUARANTINED"}
    assert await _owner_count(test_env, _IN_PROGRESS, committed_tenant) == 1
    queued = await _owner_count(
        test_env,
        "SELECT count(*) FROM media_uploads WHERE tenant_id = :t AND state = 'QUARANTINED'"
        " AND attempts = 0",
        committed_tenant,
    )
    assert queued == 1


async def _wait_for_advisory_lock_waiter(test_env: IntegrationEnv) -> None:
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    try:
        async with asyncio.timeout(10), engine.connect() as conn:
            while not await conn.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory'"
                    " AND NOT granted AND database = (SELECT oid FROM pg_database"
                    " WHERE datname = current_database()))"
                )
            ):
                await conn.rollback()
                await asyncio.sleep(0.02)
    finally:
        await engine.dispose()


async def test_complete_rechecks_expiry_after_waiting_for_the_admission_lock(
    committed_tenant: SeededTenant,
    signing_key: RSAPrivateKey,
    test_env: IntegrationEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The link expires, and another upload takes the last slot, while this complete waits for
    # the tenant's admission lock. Deciding from what was read before the lock would let it in.
    app = _one_slot_app(signing_key, monkeypatch)
    auth = AuthHeaders(signing_key)
    content = _synthetic_png()
    _stored_as_declared(monkeypatch, content)
    headers = auth("racer", committed_tenant.tenant_id)
    engine = create_async_engine(
        test_env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        other = await http.post(
            UPLOADS,
            headers=auth(
                "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "lock-wait-other-3"}
            ),
            json=_upload_body(committed_tenant, content),
        )
        assert other.status_code == 201, other.text
        await _owner_execute(
            test_env,
            "UPDATE media_uploads SET state = 'REJECTED', reason_code = 'SIZE_MISMATCH'"
            " WHERE id = CAST(:id AS uuid)",
            id=other.json()["asset_id"],
        )
        created = await http.post(
            UPLOADS,
            headers=auth(
                "racer", committed_tenant.tenant_id, **{"Idempotency-Key": "lock-wait-mine-3"}
            ),
            json=_upload_body(committed_tenant, content),
        )
        assert created.status_code == 201, created.text
        asset_id = created.json()["asset_id"]

        try:
            async with engine.begin() as holder:
                await holder.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                    {"k": f"uploads:{committed_tenant.tenant_id}"},
                )
                completing = asyncio.create_task(
                    http.post(f"/v1/marketing/assets/{asset_id}/complete", headers=headers)
                )
                await _wait_for_advisory_lock_waiter(test_env)
                # Expires now, while the complete waits: after its transaction began, so a
                # recheck against the transaction's start time (now()) would still see it valid.
                await holder.execute(
                    text(
                        "UPDATE media_uploads SET upload_url_expires_at = clock_timestamp()"
                        " WHERE id = CAST(:id AS uuid)"
                    ),
                    {"id": asset_id},
                )
                await holder.execute(
                    text(
                        "UPDATE media_uploads SET state = 'PENDING_UPLOAD', reason_code = NULL,"
                        " upload_url_expires_at = now() + interval '10 minutes'"
                        " WHERE id = CAST(:id AS uuid)"
                    ),
                    {"id": other.json()["asset_id"]},
                )
            response = await completing
        finally:
            await engine.dispose()

    assert response.status_code == 429, response.text
    assert response.json()["code"] == "UPLOAD_LIMIT_REACHED"
    assert await _owner_count(test_env, _IN_PROGRESS, committed_tenant) == 1
    still_pending = await _owner_count(
        test_env,
        "SELECT count(*) FROM media_uploads WHERE tenant_id = :t AND state = 'PENDING_UPLOAD'"
        " AND completed_at IS NULL",
        committed_tenant,
    )
    assert still_pending == 2
