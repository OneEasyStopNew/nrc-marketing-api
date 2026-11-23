"""Policy worker (NRC-4): clause extraction from registered sources.

Real Postgres (RLS, the lease function, the worker role's grants). Storage is a small fake that
serves bytes from memory, so no SeaweedFS is needed. All sources and files are synthetic.
"""

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Row, func, insert, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import bind_tenant
from app.models import PolicySource, PolicySourceClause
from app.models.enums import PolicyLanguage, SourceExtractionState
from app.services.storage import StorageError
from app.workers.common import WorkerConfigError
from app.workers.policy import build_storage, run_once
from tests.conftest import (
    Sessions,
    World,
    add_policy_upload,
    as_tenant_owner,
    make_pdf,
    owner_sql,
)

BUCKET = "nrc-test-policy"
MARKDOWN = b"# [SYNTHETIC] Claims\n\nFirst clause.\n\nSecond  clause, kept verbatim.\n"


class FakeStorage:
    # The policy storage the worker reads through: bytes by key, with an outage switch.

    def __init__(self, objects: dict[str, bytes], *, bucket: str = BUCKET) -> None:
        self.objects = objects
        self._bucket = bucket
        self.outage = False
        self.reads = 0

    @property
    def bucket(self) -> str:
        return self._bucket

    async def read(self, key: str, *, version_id: str | None = None) -> bytes:
        self.reads += 1
        if self.outage:
            raise StorageError
        return self.objects[key]


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(
        update={
            "worker_backoff_base_seconds": 30.0,
            "worker_backoff_max_seconds": 60.0,
            "policy_worker_max_attempts": 3,
            "policy_extraction_timeout_seconds": 5.0,
        }
    )


async def add_source(
    connection: AsyncConnection,
    storage: FakeStorage,
    tenant_id: UUID,
    content: bytes,
    mime_type: str = "text/markdown",
    **upload: Any,
) -> UUID:
    # A PENDING source on a READY upload whose bytes the fake storage holds.
    upload_id = await add_policy_upload(connection, tenant_id, mime_type=mime_type, **upload)
    storage.objects[f"policy/{upload_id}"] = content
    source_id = uuid4()
    async with as_tenant_owner(connection, tenant_id):
        await connection.execute(
            insert(PolicySource).values(
                id=source_id,
                tenant_id=tenant_id,
                title="[SYNTHETIC] Policy source",
                source_ref="synthetic:nrc-4-test",
                checksum_sha256=upload_id.hex * 2,
                storage_key=f"policy/{upload_id}",
                language=PolicyLanguage.EN,
                is_synthetic=True,
                created_by="policy-admin-a",
                upload_id=upload_id,
                extraction_state=SourceExtractionState.PENDING,
                available_at=func.now(),
            )
        )
    return source_id


async def row(connection: AsyncConnection, tenant_id: UUID, source_id: UUID) -> Row[Any]:
    rows = await owner_sql(
        connection,
        tenant_id,
        "SELECT extraction_state, extraction_error_code, extracted_at, page_count, available_at,"
        " leased_until, lease_owner, attempts, failed_attempts,"
        " coalesce(available_at > clock_timestamp(), false) AS deferred"
        " FROM policy_sources WHERE id = :id",
        {"id": source_id},
    )
    assert len(rows) == 1
    return rows[0]


async def clauses(connection: AsyncConnection, tenant_id: UUID, source_id: UUID) -> list[Row[Any]]:
    return await owner_sql(
        connection,
        tenant_id,
        "SELECT clause_ref, section, page, text, checksum_sha256, language, tenant_id"
        " FROM policy_source_clauses WHERE source_id = :id ORDER BY length(clause_ref), clause_ref",
        {"id": source_id},
    )


async def make_due(connection: AsyncConnection, tenant_id: UUID, source_id: UUID) -> None:
    await owner_sql(
        connection,
        tenant_id,
        "UPDATE policy_sources SET available_at = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": source_id},
    )


async def expire_lease(connection: AsyncConnection, tenant_id: UUID, source_id: UUID) -> None:
    await owner_sql(
        connection,
        tenant_id,
        "UPDATE policy_sources SET leased_until = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": source_id},
    )


async def work(
    sessions: Sessions, storage: FakeStorage, settings: Settings, owner: str = "w1"
) -> int:
    return await run_once(sessions, storage, settings, owner=owner)


# ---- Extraction


async def test_a_markdown_source_is_extracted(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    source_id = await add_source(connection, storage, world.a.tenant_id, MARKDOWN)

    assert await work(sessions, storage, settings) == 1

    state = await row(connection, world.a.tenant_id, source_id)
    assert state.extraction_state == "EXTRACTED"
    assert state.extracted_at is not None
    assert state.page_count is None
    assert (state.extraction_error_code, state.available_at) == (None, None)
    assert (state.leased_until, state.lease_owner) == (None, None)
    found = await clauses(connection, world.a.tenant_id, source_id)
    assert [(c.clause_ref, c.section, c.page, c.text, c.language) for c in found] == [
        ("c1", "[SYNTHETIC] Claims", None, "First clause.", "en"),
        ("c2", "[SYNTHETIC] Claims", None, "Second  clause, kept verbatim.", "en"),
    ]
    assert all(c.tenant_id == world.a.tenant_id for c in found)
    assert all(len(c.checksum_sha256) == 64 for c in found)
    # Done: nothing left to lease.
    assert await work(sessions, storage, settings) == 0


async def test_a_pdf_source_is_extracted_with_pages(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    pdf = make_pdf([["1. Scope", None, "[SYNTHETIC] On page one."], ["On page two."]])
    source_id = await add_source(
        connection, storage, world.a.tenant_id, pdf, mime_type="application/pdf"
    )

    await work(sessions, storage, settings)

    state = await row(connection, world.a.tenant_id, source_id)
    assert (state.extraction_state, state.page_count) == ("EXTRACTED", 2)
    found = await clauses(connection, world.a.tenant_id, source_id)
    assert [(c.clause_ref, c.section, c.page, c.text) for c in found] == [
        ("p1-c1", "1. Scope", 1, "[SYNTHETIC] On page one."),
        ("p2-c2", "1. Scope", 2, "On page two."),
    ]


async def test_each_tenants_source_is_processed_under_its_own_tenant(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    a_source = await add_source(connection, storage, world.a.tenant_id, b"[SYNTHETIC] A.")
    b_source = await add_source(connection, storage, world.b.tenant_id, b"[SYNTHETIC] B.")

    assert await work(sessions, storage, settings) == 2

    for tenant, source, expected in (
        (world.a.tenant_id, a_source, "[SYNTHETIC] A."),
        (world.b.tenant_id, b_source, "[SYNTHETIC] B."),
    ):
        (only,) = await clauses(connection, tenant, source)
        assert (only.text, only.tenant_id) == (expected, tenant)


@pytest.mark.parametrize(
    ("content", "mime", "code"),
    [
        pytest.param(b"   \n\n", "text/plain", "SOURCE_EMPTY", id="empty-text"),
        pytest.param(b"%PDF-1.7 damaged", "application/pdf", "SOURCE_UNREADABLE", id="damaged-pdf"),
        pytest.param(b"caf\xe9", "text/plain", "SOURCE_UNREADABLE", id="not-utf8"),
    ],
)
async def test_an_unextractable_file_fails_with_its_code_and_no_clauses(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
    content: bytes,
    mime: str,
    code: str,
) -> None:
    storage = FakeStorage({})
    source_id = await add_source(connection, storage, world.a.tenant_id, content, mime)

    await work(sessions, storage, settings)

    state = await row(connection, world.a.tenant_id, source_id)
    assert (state.extraction_state, state.extraction_error_code) == ("FAILED", code)
    assert (state.leased_until, state.lease_owner, state.available_at) == (None, None, None)
    assert state.extracted_at is None
    assert await clauses(connection, world.a.tenant_id, source_id) == []
    # Terminal: never leased again.
    assert await work(sessions, storage, settings) == 0


async def test_a_source_over_the_clause_cap_fails_without_clauses(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    source_id = await add_source(connection, storage, world.a.tenant_id, b"One.\n\nTwo.\n\nThree.")

    await work(sessions, storage, settings.model_copy(update={"policy_extraction_max_clauses": 2}))

    state = await row(connection, world.a.tenant_id, source_id)
    assert (state.extraction_state, state.extraction_error_code) == (
        "FAILED",
        "SOURCE_TOO_MANY_CLAUSES",
    )
    assert await clauses(connection, world.a.tenant_id, source_id) == []


# ---- Storage and time


async def test_a_storage_outage_is_retried_then_fails_after_the_attempt_cap(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    storage.outage = True
    tenant = world.a.tenant_id
    source_id = await add_source(connection, storage, tenant, MARKDOWN)

    await work(sessions, storage, settings)

    waiting = await row(connection, tenant, source_id)
    assert (waiting.extraction_state, waiting.extraction_error_code) == (
        "PENDING",
        "SOURCE_STORAGE_UNAVAILABLE",
    )
    assert waiting.deferred is True
    assert (waiting.leased_until, waiting.lease_owner, waiting.attempts) == (None, None, 1)
    # Not due yet: nothing to lease until the backoff passes.
    assert await work(sessions, storage, settings) == 0

    for attempt in (2, 3):
        await make_due(connection, tenant, source_id)
        assert await work(sessions, storage, settings) == 1
        assert (await row(connection, tenant, source_id)).attempts == attempt

    failed = await row(connection, tenant, source_id)
    assert (failed.extraction_state, failed.extraction_error_code) == (
        "FAILED",
        "SOURCE_STORAGE_UNAVAILABLE",
    )
    assert (failed.available_at, failed.leased_until, failed.lease_owner) == (None, None, None)
    assert await clauses(connection, tenant, source_id) == []


async def test_a_source_waits_and_then_succeeds_when_storage_returns(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    storage.outage = True
    tenant = world.a.tenant_id
    source_id = await add_source(connection, storage, tenant, MARKDOWN)
    await work(sessions, storage, settings)

    storage.outage = False
    await make_due(connection, tenant, source_id)
    await work(sessions, storage, settings)

    state = await row(connection, tenant, source_id)
    assert (state.extraction_state, state.extraction_error_code) == ("EXTRACTED", None)
    assert len(await clauses(connection, tenant, source_id)) == 2


async def test_a_bucket_that_is_not_the_uploads_defers_and_reads_nothing(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    # The worker holds the policy storage; the upload says another bucket (for example the media
    # bucket): the object is never read from the wrong place.
    storage = FakeStorage({})
    source_id = await add_source(
        connection, storage, world.a.tenant_id, MARKDOWN, bucket="nrc-test-media"
    )

    await work(sessions, storage, settings)

    state = await row(connection, world.a.tenant_id, source_id)
    assert (state.extraction_state, state.extraction_error_code) == (
        "PENDING",
        "SOURCE_STORAGE_UNAVAILABLE",
    )
    assert state.deferred is True
    assert storage.reads == 0


async def test_a_slow_read_times_out_and_is_retried(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    class SlowStorage(FakeStorage):
        async def read(self, key: str, *, version_id: str | None = None) -> bytes:

            await asyncio.sleep(10)
            return await super().read(key, version_id=version_id)

    storage = SlowStorage({})
    source_id = await add_source(connection, storage, world.a.tenant_id, MARKDOWN)

    await work(
        sessions, storage, settings.model_copy(update={"policy_extraction_timeout_seconds": 1.0})
    )

    state = await row(connection, world.a.tenant_id, source_id)
    assert (state.extraction_state, state.extraction_error_code) == (
        "PENDING",
        "SOURCE_EXTRACTION_TIMEOUT",
    )
    assert state.deferred is True


# ---- Interrupted work, leases and the attempt cap


class Boom(Exception):
    # Stands in for a crash between the clause insert and the commit.
    pass


async def test_a_failure_after_the_clause_insert_leaves_no_clauses_and_is_recovered(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = FakeStorage({})
    tenant = world.a.tenant_id
    source_id = await add_source(connection, storage, tenant, MARKDOWN)
    real_commit = AsyncSession.commit

    async def commit_unless_clauses_pending(self: AsyncSession) -> None:
        # The worker's own session sees its uncommitted inserts: fail exactly after them.
        if await self.scalar(select(func.count()).select_from(PolicySourceClause)):
            raise Boom
        await real_commit(self)

    monkeypatch.setattr(AsyncSession, "commit", commit_unless_clauses_pending)

    await work(sessions, storage, settings)

    crashed = await row(connection, tenant, source_id)
    assert crashed.extraction_state == "EXTRACTING"
    assert crashed.extracted_at is None
    assert crashed.lease_owner == "w1"
    assert await clauses(connection, tenant, source_id) == []
    assert f"policy work failed item={source_id}" in caplog.text

    # While the lease holds nobody else takes it; once it expires the next worker redoes it all.
    assert await work(sessions, storage, settings, owner="w2") == 0
    monkeypatch.setattr(AsyncSession, "commit", real_commit)
    await expire_lease(connection, tenant, source_id)
    assert await work(sessions, storage, settings, owner="w2") == 1

    done = await row(connection, tenant, source_id)
    assert (done.extraction_state, done.failed_attempts, done.attempts) == ("EXTRACTED", 1, 2)
    assert len(await clauses(connection, tenant, source_id)) == 2


async def test_a_source_whose_lease_expired_is_leased_again_and_counted(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    tenant = world.a.tenant_id
    source_id = await add_source(connection, storage, tenant, MARKDOWN)
    await owner_sql(
        connection,
        tenant,
        "UPDATE policy_sources SET extraction_state = 'EXTRACTING', lease_owner = 'dead-worker',"
        " attempts = 1, leased_until = clock_timestamp() - interval '1 second' WHERE id = :id",
        {"id": source_id},
    )

    assert await work(sessions, storage, settings) == 1

    state = await row(connection, tenant, source_id)
    assert (state.extraction_state, state.attempts, state.failed_attempts) == ("EXTRACTED", 2, 1)


async def test_a_source_that_keeps_crashing_is_failed_without_being_read(
    connection: AsyncConnection, world: World, sessions: Sessions, settings: Settings
) -> None:
    storage = FakeStorage({})
    tenant = world.a.tenant_id
    source_id = await add_source(connection, storage, tenant, MARKDOWN)
    await owner_sql(
        connection,
        tenant,
        "UPDATE policy_sources SET extraction_state = 'EXTRACTING', lease_owner = 'dead-worker',"
        " attempts = 3, failed_attempts = 2, leased_until = clock_timestamp() - interval '1 second'"
        " WHERE id = :id",
        {"id": source_id},
    )

    await work(sessions, storage, settings)

    state = await row(connection, tenant, source_id)
    # The lease counted the expired one: 2 + 1 reaches the cap of 3.
    assert (state.extraction_state, state.extraction_error_code, state.failed_attempts) == (
        "FAILED",
        "SOURCE_EXTRACTION_FAILED",
        3,
    )
    assert storage.reads == 0
    assert await clauses(connection, tenant, source_id) == []


async def test_a_worker_that_lost_its_lease_writes_nothing(
    connection: AsyncConnection,
    world: World,
    sessions: Sessions,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = world.a.tenant_id

    class LosingStorage(FakeStorage):
        async def read(self, key: str, *, version_id: str | None = None) -> bytes:
            # The lease runs out while the file is being read; another worker is not involved.
            await expire_lease(connection, tenant, source_id)
            return await super().read(key, version_id=version_id)

    storage = LosingStorage({})
    source_id = await add_source(connection, storage, tenant, MARKDOWN)

    await work(sessions, storage, settings)

    state = await row(connection, tenant, source_id)
    assert state.extraction_state == "EXTRACTING"
    assert await clauses(connection, tenant, source_id) == []
    assert f"policy lease lost item={source_id}" in caplog.text


# ---- The worker role and tenants


async def test_a_worker_session_cannot_touch_another_tenants_source(
    connection: AsyncConnection, world: World, sessions: Sessions
) -> None:
    storage = FakeStorage({})
    b_source = await add_source(connection, storage, world.b.tenant_id, MARKDOWN)

    async with sessions() as db:
        await bind_tenant(db, world.a.tenant_id)
        assert await db.get(PolicySource, b_source) is None
        changed = await db.execute(
            update(PolicySource)
            .where(PolicySource.id == b_source)
            .values(extraction_state=SourceExtractionState.FAILED)
        )
        assert changed.rowcount == 0  # type: ignore[attr-defined]
        with pytest.raises(DBAPIError, match="row-level security"):
            await db.execute(
                insert(PolicySourceClause).values(
                    tenant_id=world.b.tenant_id,
                    source_id=b_source,
                    clause_ref="c1",
                    text="[SYNTHETIC] x",
                    checksum_sha256="0" * 64,
                    language=PolicyLanguage.EN,
                )
            )

    assert (await row(connection, world.b.tenant_id, b_source)).extraction_state == "PENDING"


# ---- Start-up


def test_the_worker_refuses_to_start_without_the_policy_bucket(settings: Settings) -> None:
    with pytest.raises(WorkerConfigError, match="POLICY_SOURCE_BUCKET"):
        build_storage(settings.model_copy(update={"policy_source_bucket": None}))


def test_the_worker_refuses_a_timeout_that_outlasts_its_lease(settings: Settings) -> None:
    long_timeout = settings.model_copy(
        update={
            "policy_source_bucket": BUCKET,
            "worker_lease_seconds": 60,
            "policy_extraction_timeout_seconds": 60.0,
        }
    )

    with pytest.raises(WorkerConfigError, match="below WORKER_LEASE_SECONDS"):
        build_storage(long_timeout)
