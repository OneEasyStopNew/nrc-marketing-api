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
from app.services.storage import ObjectStorage
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
