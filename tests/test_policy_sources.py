"""Registering policy sources from READY policy uploads (NRC-4). Real Postgres with RLS.

Fixture data is synthetic. The parallel-duplicate test runs on committed data with independent
connections, so the unique constraint really decides the race.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TypedDict
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.auth.access import Capability, Role, TenantContext
from app.auth.tokens import Principal
from app.core.errors import NotFoundError, PolicySourceUploadNotReadyError
from app.db.session import TenantSession, bind_tenant
from app.models import MediaUpload, PolicySource, PolicySourceClause
from app.models.enums import (
    IdentifiablePerson,
    PolicyLanguage,
    SourceExtractionState,
    UploadPurpose,
    UploadState,
)
from app.services.policy_sources import get_source, list_source_clauses, register_source
from tests.conftest import (
    IntegrationEnv,
    SeededTenant,
    World,
    add_policy_upload,
    app_session,
    as_tenant_owner,
)

ADMIN = "policy-admin-a"


class SourceArgs(TypedDict):
    title: str
    source_ref: str
    language: PolicyLanguage
    is_synthetic: bool


SOURCE = SourceArgs(
    title="[SYNTHETIC] Policy source",
    source_ref="synthetic:nrc-4-test",
    language=PolicyLanguage.EN,
    is_synthetic=True,
)


def ctx_for(tenant_id: UUID, sub: str = ADMIN) -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        principal=Principal(sub=sub, username=sub, scopes=frozenset()),
        roles=frozenset({Role.POLICY_ADMIN}),
        capabilities=frozenset({Capability.POLICY_MANAGE}),
        study_ids=frozenset(),
    )


@pytest.fixture
async def db(connection: AsyncConnection, world: World) -> AsyncIterator[AsyncSession]:
    async with app_session(connection) as session:
        await bind_tenant(session, world.a.tenant_id)
        yield session


async def register(db: AsyncSession, world: World, upload_id: UUID) -> tuple[PolicySource, bool]:
    # Seeding other tenants' rows on the shared connection moves its tenant setting; bind A again.
    await bind_tenant(db, world.a.tenant_id)
    return await register_source(db, ctx_for(world.a.tenant_id), upload_id=upload_id, **SOURCE)


async def count_sources(connection: AsyncConnection, world: World) -> int:
    async with as_tenant_owner(connection, world.a.tenant_id):
        return int(await connection.scalar(select(func.count()).select_from(PolicySource)) or 0)


async def test_a_ready_upload_becomes_a_pending_source(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    upload_id = await add_policy_upload(connection, world.a.tenant_id)

    source, created = await register(db, world, upload_id)

    await db.refresh(source)
    assert created is True
    assert source.extraction_state is SourceExtractionState.PENDING
    assert source.upload_id == upload_id
    assert source.tenant_id == world.a.tenant_id
    assert source.created_by == ADMIN
    assert source.title == "[SYNTHETIC] Policy source"
    assert source.is_synthetic is True
    assert source.language is PolicyLanguage.EN
    assert source.storage_key == f"policy/{upload_id}"
    assert source.checksum_sha256 == upload_id.hex * 2
    assert (source.extracted_at, source.page_count, source.extraction_error_code) == (
        None,
        None,
        None,
    )
    # Due at once: the worker's lease needs available_at <= now.
    assert source.available_at is not None
    assert source.available_at <= datetime.now(UTC)


async def test_the_same_checksum_returns_the_existing_source(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    checksum = hashlib.sha256(b"[SYNTHETIC] same file").hexdigest()
    first_upload = await add_policy_upload(connection, world.a.tenant_id, checksum=checksum)
    second_upload = await add_policy_upload(connection, world.a.tenant_id, checksum=checksum)
    first, _ = await register(db, world, first_upload)
    await db.commit()

    again, created = await register_source(
        db,
        ctx_for(world.a.tenant_id, "someone-else"),
        upload_id=second_upload,
        title="[SYNTHETIC] Renamed",
        source_ref="synthetic:other",
        language=PolicyLanguage.ES,
        is_synthetic=False,
    )

    assert created is False
    assert again.id == first.id
    # Provenance is untouched: nothing of the second request was written.
    assert (again.title, again.source_ref, again.created_by, again.upload_id) == (
        "[SYNTHETIC] Policy source",
        "synthetic:nrc-4-test",
        ADMIN,
        first_upload,
    )
    assert again.language is PolicyLanguage.EN
    assert await count_sources(connection, world) == 1


async def test_the_same_upload_twice_returns_the_existing_source(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    upload_id = await add_policy_upload(connection, world.a.tenant_id)
    first, first_created = await register(db, world, upload_id)

    again, created = await register(db, world, upload_id)

    assert (first_created, created) == (True, False)
    assert again.id == first.id
    assert await count_sources(connection, world) == 1


async def test_registering_a_failed_source_again_queues_its_extraction_again(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    # Otherwise the checksum would tie the file to a dead source: no way to retry after a fix.
    checksum = hashlib.sha256(b"[SYNTHETIC] failed once").hexdigest()
    first_upload = await add_policy_upload(connection, world.a.tenant_id, checksum=checksum)
    second_upload = await add_policy_upload(connection, world.a.tenant_id, checksum=checksum)
    first, _ = await register(db, world, first_upload)
    await db.commit()
    async with as_tenant_owner(connection, world.a.tenant_id):
        await connection.execute(
            update(PolicySource)
            .where(PolicySource.id == first.id)
            .values(
                extraction_state=SourceExtractionState.FAILED,
                extraction_error_code="STORAGE_UNAVAILABLE",
                available_at=None,
                attempts=3,
                failed_attempts=2,
            )
        )
    db.expire_all()

    again, created = await register(db, world, second_upload)
    await db.commit()
    await db.refresh(again)

    assert (created, again.id) == (False, first.id)
    assert again.extraction_state is SourceExtractionState.PENDING
    assert (again.extraction_error_code, again.attempts, again.failed_attempts) == (None, 0, 0)
    assert again.available_at is not None
    # Read from the new upload (same bytes); the first registration's provenance stays.
    assert (again.upload_id, again.storage_key) == (second_upload, f"policy/{second_upload}")
    assert (again.title, again.created_by) == ("[SYNTHETIC] Policy source", ADMIN)
    assert await count_sources(connection, world) == 1


async def test_another_tenant_may_register_the_same_checksum(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    checksum = hashlib.sha256(b"[SYNTHETIC] shared bytes").hexdigest()
    a_upload = await add_policy_upload(connection, world.a.tenant_id, checksum=checksum)
    b_upload = await add_policy_upload(connection, world.b.tenant_id, checksum=checksum)
    a_source, _ = await register(db, world, a_upload)
    await db.commit()

    async with app_session(connection) as b_db:
        await bind_tenant(b_db, world.b.tenant_id)
        b_source, created = await register_source(
            b_db,
            ctx_for(world.b.tenant_id),
            upload_id=b_upload,
            **SOURCE,
        )

    assert created is True
    assert b_source.id != a_source.id


async def test_an_upload_of_another_tenant_is_not_found(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    other = await add_policy_upload(connection, world.b.tenant_id)

    with pytest.raises(NotFoundError, match="Upload not found"):
        await register(db, world, other)
    assert await count_sources(connection, world) == 0


async def test_a_creative_upload_is_not_found(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    creative = await add_policy_upload(
        connection,
        world.a.tenant_id,
        purpose=UploadPurpose.CREATIVE_MEDIA,
        mime_type="image/png",
        study_id=world.a.study_id,
    )

    with pytest.raises(NotFoundError, match="Upload not found"):
        await register(db, world, creative)


async def test_an_unknown_upload_is_not_found(world: World, db: AsyncSession) -> None:
    with pytest.raises(NotFoundError, match="Upload not found"):
        await register(db, world, uuid4())


@pytest.mark.parametrize(
    "state",
    [
        UploadState.PENDING_UPLOAD,
        UploadState.QUARANTINED,
        UploadState.SCANNING,
        UploadState.REJECTED,
        UploadState.EXPIRED,
    ],
)
async def test_an_upload_that_is_not_ready_is_refused(
    connection: AsyncConnection, world: World, db: AsyncSession, state: UploadState
) -> None:
    upload_id = await add_policy_upload(connection, world.a.tenant_id, state=state)

    with pytest.raises(PolicySourceUploadNotReadyError, match="not passed its checks") as error:
        await register(db, world, upload_id)
    assert error.value.details == {"status": state.value}
    assert await count_sources(connection, world) == 0


async def test_get_source_hides_other_tenants(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    source, _ = await register(db, world, await add_policy_upload(connection, world.a.tenant_id))
    await db.commit()

    assert (await get_source(db, ctx_for(world.a.tenant_id), source.id)).id == source.id
    # The caller's tenant filter alone ...
    with pytest.raises(NotFoundError, match="Policy source not found"):
        await get_source(db, ctx_for(world.b.tenant_id), source.id)
    # ... and row-level security alone (a session bound to the other tenant).
    async with app_session(connection) as b_db:
        await bind_tenant(b_db, world.b.tenant_id)
        with pytest.raises(NotFoundError, match="Policy source not found"):
            await get_source(b_db, ctx_for(world.a.tenant_id), source.id)
        with pytest.raises(NotFoundError, match="Policy source not found"):
            await list_source_clauses(b_db, ctx_for(world.a.tenant_id), source.id)


async def test_clauses_are_listed_by_page_then_reading_order(
    connection: AsyncConnection, world: World, db: AsyncSession
) -> None:
    source, _ = await register(db, world, await add_policy_upload(connection, world.a.tenant_id))
    await db.commit()
    async with as_tenant_owner(connection, world.a.tenant_id):
        for ref, page in (("p2-c11", 2), ("p1-c10", 1), ("c1", None), ("p1-c9", 1)):
            text = f"[SYNTHETIC] {ref}"
            await connection.execute(
                insert(PolicySourceClause).values(
                    id=uuid4(),
                    tenant_id=world.a.tenant_id,
                    source_id=source.id,
                    clause_ref=ref,
                    page=page,
                    text=text,
                    checksum_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    language=PolicyLanguage.EN,
                )
            )

    clauses = await list_source_clauses(db, ctx_for(world.a.tenant_id), source.id)

    assert [c.clause_ref for c in clauses] == ["p1-c9", "p1-c10", "p2-c11", "c1"]


# ---- Parallel duplicates, on committed data


async def _committed_uploads(
    env: IntegrationEnv, seeded: SeededTenant, checksums: list[str]
) -> list[UUID]:
    engine = create_async_engine(
        env.test_migration_database_url.get_secret_value(), poolclass=NullPool
    )
    ids = [uuid4() for _ in checksums]
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as owner:
            await bind_tenant(owner, seeded.tenant_id)
            owner.add_all(
                MediaUpload(
                    id=upload_id,
                    tenant_id=seeded.tenant_id,
                    purpose=UploadPurpose.POLICY_SOURCE,
                    state=UploadState.READY,
                    name="[SYNTHETIC] policy source",
                    declared_mime_type="text/plain",
                    declared_size_bytes=10,
                    declared_checksum_sha256=checksum,
                    mime_type="text/plain",
                    bucket="nrc-test-policy",
                    s3_key=f"policy/{upload_id}",
                    upload_url_expires_at=datetime(2026, 11, 20, tzinfo=UTC),
                    identifiable_person=IdentifiablePerson.UNKNOWN,
                    created_by=ADMIN,
                )
                for upload_id, checksum in zip(ids, checksums, strict=True)
            )
            await owner.commit()
    finally:
        await engine.dispose()
    return ids


@pytest.mark.parametrize("same_upload", [False, True], ids=["same-checksum", "same-upload"])
async def test_parallel_duplicate_registrations_make_one_source(
    committed_tenant: SeededTenant, test_env: IntegrationEnv, same_upload: bool
) -> None:
    checksum = hashlib.sha256(b"[SYNTHETIC] parallel").hexdigest()
    first, second = await _committed_uploads(test_env, committed_tenant, [checksum, checksum])
    uploads = [first, first] if same_upload else [first, second]
    ctx = ctx_for(committed_tenant.tenant_id)
    api = create_async_engine(test_env.test_database_url.get_secret_value(), poolclass=NullPool)

    async def register_one(upload_id: UUID) -> tuple[UUID, bool]:
        # Each request on its own API-role connection, as two requests would be.
        async with AsyncSession(api, sync_session_class=TenantSession) as session:
            await bind_tenant(session, ctx.tenant_id)
            source, created = await register_source(session, ctx, upload_id=upload_id, **SOURCE)
            source_id = source.id
            await session.commit()
            return source_id, created

    try:
        results = await asyncio.gather(*(register_one(u) for u in uploads))
        async with AsyncSession(api, sync_session_class=TenantSession) as session:
            await bind_tenant(session, ctx.tenant_id)
            rows = await session.scalar(select(func.count()).select_from(PolicySource))
    finally:
        await api.dispose()

    assert results[0][0] == results[1][0]
    assert sorted(created for _, created in results) == [False, True]
    assert rows == 1
