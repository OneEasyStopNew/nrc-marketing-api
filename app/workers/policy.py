"""Policy worker (NRC-4): extracts clauses from registered policy sources.

Run with `uv run python -m app.workers.policy`. It connects as marketing_worker and leases due
sources one at a time through lease_policy_sources(), which returns ids only; everything else
runs in a session bound to the source's tenant, so row-level security applies.

For each source it reads exactly the pinned object version from the policy bucket, extracts the
clauses off the event loop (app.services.policy_extraction), and then, in ONE transaction, inserts
every clause and marks the source EXTRACTED. A crash or failure before that commit leaves no
clauses and a source that is not EXTRACTED: a source never shows a partial clause set (AC4).

A file that cannot be turned into clauses ends as FAILED with a code. Temporary storage problems
and timeouts send the source back to PENDING with a backoff, up to POLICY_WORKER_MAX_ATTEMPTS
leases, then FAILED. A crash leaves the lease to expire; the lease function counts that as a failed
attempt, with the same cap. Every change is fenced on this worker's lease, so a slow or duplicate
worker can never write twice.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol
from uuid import UUID

from sqlalchemy import TextClause, Update, func, insert, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import bind_tenant, create_engine, create_sessionmaker
from app.models import MediaUpload, PolicySource, PolicySourceClause
from app.models.enums import SourceExtractionState, UploadState
from app.services.policy_extraction import (
    Extraction,
    ExtractionCode,
    ExtractionError,
    extract_clauses,
)
from app.services.storage import ObjectStorage, StorageError
from app.workers.common import (
    DEFAULT_OWNER,
    WorkerConfigError,
    backoff_seconds,
    lease_seconds,
    stop_on_signals,
    worker_database_url,
)

logger = logging.getLogger(__name__)

_LEASE_SOURCES = text("SELECT source_id, tenant_id FROM lease_policy_sources(:owner, 1, :lease)")
_IN_PROGRESS: Final = (SourceExtractionState.PENDING, SourceExtractionState.EXTRACTING)
_RELEASE: Final[Mapping[str, object]] = {
    "leased_until": None,
    "lease_owner": None,
    "available_at": None,
}


class LeaseLostError(Exception):
    # Another worker leased the source after this one's lease ran out; leave it to that worker.
    pass


class _TransientError(Exception):
    # Storage or time ran out; worth another lease. The code is stored while the source waits.
    def __init__(self, code: ExtractionCode) -> None:
        super().__init__(code.value)
        self.code = code


class ObjectReader(Protocol):
    # What the worker needs of storage (ObjectStorage; tests use a fake).
    @property
    def bucket(self) -> str: ...

    async def read(self, key: str, *, version_id: str | None = None) -> bytes: ...


@dataclass(frozen=True, slots=True)
class _Run:
    sessions: async_sessionmaker[AsyncSession]
    storage: ObjectReader
    settings: Settings
    owner: str


type _Handler = Callable[[_Run, UUID, UUID], Awaitable[None]]


async def run_once(
    sessions: async_sessionmaker[AsyncSession],
    storage: ObjectReader,
    settings: Settings,
    *,
    owner: str = DEFAULT_OWNER,
    max_items: int | None = None,
) -> int:
    # Lease and process up to max_items (default WORKER_MAX_EVENTS_PER_RUN) due sources, one lease
    # each. Returns how many were leased.
    limit = settings.worker_max_events_per_run if max_items is None else max_items
    run = _Run(sessions, storage, settings, owner)
    return await _drain(run, _LEASE_SOURCES, _extract, limit)


async def _drain(run: _Run, lease: TextClause, handle: _Handler, limit: int) -> int:
    # One queue: lease an item, handle it. A second queue is another lease statement and handler.
    leased = 0
    while leased < limit:
        async with run.sessions() as db:
            row = (
                await db.execute(lease, {"owner": run.owner, "lease": lease_seconds(run.settings)})
            ).one_or_none()
            await db.commit()
        if row is None:
            break
        leased += 1
        item_id, tenant_id = row[0], row[1]
        try:
            await handle(run, item_id, tenant_id)
        except LeaseLostError:
            logger.warning("policy lease lost item=%s tenant=%s", item_id, tenant_id)
        except Exception:
            # The lease expires unfinished, which the next lease counts as a failed attempt.
            logger.exception("policy work failed item=%s tenant=%s", item_id, tenant_id)
    return leased


def _fenced(source_id: UUID, owner: str) -> Update:
    # Only the worker whose lease is still valid may change the source, and only while it is
    # unfinished: terminal states never change again.
    return (
        update(PolicySource)
        .where(
            PolicySource.id == source_id,
            PolicySource.lease_owner == owner,
            PolicySource.leased_until > func.clock_timestamp(),
            PolicySource.extraction_state.in_(_IN_PROGRESS),
        )
        .execution_options(synchronize_session=False)
    )


async def _change(
    db: AsyncSession, run: _Run, source_id: UUID, values: Mapping[str, object]
) -> None:
    changed = await db.scalar(
        _fenced(source_id, run.owner).values(**values).returning(PolicySource.id)
    )
    if changed is None:
        raise LeaseLostError


async def _extract(run: _Run, source_id: UUID, tenant_id: UUID) -> None:
    settings = run.settings
    async with run.sessions() as db:
        await bind_tenant(db, tenant_id)
        source = await db.get(PolicySource, source_id, populate_existing=True)
        if (
            source is None
            or source.lease_owner != run.owner
            or source.extraction_state not in _IN_PROGRESS
        ):
            raise LeaseLostError
        if source.failed_attempts >= settings.policy_worker_max_attempts:
            await _fail(db, run, source, ExtractionCode.FAILED)
            return
        await _change(db, run, source.id, {"extraction_state": SourceExtractionState.EXTRACTING})
        await db.commit()
        try:
            extraction = await _read_and_extract(run, db, source)
        except ExtractionError as error:
            await _fail(db, run, source, error.failure_code)
        except _TransientError as error:
            await _retry_later(db, run, source, error.code)
        else:
            await _complete(db, run, source, extraction)


async def _read_and_extract(run: _Run, db: AsyncSession, source: PolicySource) -> Extraction:
    settings = run.settings
    upload = await db.get(MediaUpload, source.upload_id) if source.upload_id else None
    if upload is None or upload.state is not UploadState.READY or upload.mime_type is None:
        raise ExtractionError(ExtractionCode.UNREADABLE, "The upload of this source is not usable.")
    # Only the bucket recorded on the upload is read, and only through the policy storage: if it
    # is not the configured one the source waits (and fails after the attempt cap).
    if run.storage.bucket != upload.bucket:
        raise _TransientError(ExtractionCode.STORAGE_UNAVAILABLE)
    try:
        # shortcut: a timed-out extraction thread cannot be stopped and runs on until it ends,
        # bounded by the page and clause caps; move extraction to a subprocess if that ever hurts.
        async with asyncio.timeout(settings.policy_extraction_timeout_seconds):
            content = await run.storage.read(upload.s3_key, version_id=upload.s3_version_id)
            return await asyncio.to_thread(
                extract_clauses,
                content,
                upload.mime_type,
                max_pages=settings.policy_source_max_pages,
                max_clauses=settings.policy_extraction_max_clauses,
            )
    except StorageError:
        raise _TransientError(ExtractionCode.STORAGE_UNAVAILABLE) from None
    except TimeoutError:
        raise _TransientError(ExtractionCode.TIMEOUT) from None


async def _complete(
    db: AsyncSession, run: _Run, source: PolicySource, extraction: Extraction
) -> None:
    # The fenced state change comes first, so a worker that lost its lease inserts nothing. Both
    # commit together: no clauses without EXTRACTED, no EXTRACTED without clauses.
    await _change(
        db,
        run,
        source.id,
        {
            "extraction_state": SourceExtractionState.EXTRACTED,
            "extraction_error_code": None,
            "extracted_at": func.now(),
            "page_count": extraction.page_count,
            **_RELEASE,
        },
    )
    await db.execute(
        insert(PolicySourceClause),
        [
            {
                "tenant_id": source.tenant_id,
                "source_id": source.id,
                "clause_ref": clause.clause_ref,
                "section": clause.section,
                "page": clause.page,
                "text": clause.text,
                "checksum_sha256": clause.checksum_sha256,
                "language": source.language,
            }
            for clause in extraction.clauses
        ],
    )
    await db.commit()
    logger.info(
        "policy source extracted source=%s tenant=%s clauses=%d",
        source.id,
        source.tenant_id,
        len(extraction.clauses),
    )


async def _fail(db: AsyncSession, run: _Run, source: PolicySource, code: ExtractionCode) -> None:
    await _change(
        db,
        run,
        source.id,
        {
            "extraction_state": SourceExtractionState.FAILED,
            "extraction_error_code": code.value,
            **_RELEASE,
        },
    )
    await db.commit()
    logger.info(
        "policy source failed source=%s tenant=%s code=%s", source.id, source.tenant_id, code.value
    )


async def _retry_later(
    db: AsyncSession, run: _Run, source: PolicySource, code: ExtractionCode
) -> None:
    if source.attempts >= run.settings.policy_worker_max_attempts:
        await _fail(db, run, source, code)
        return
    delay = backoff_seconds(source.attempts, run.settings)
    await _change(
        db,
        run,
        source.id,
        {
            "extraction_state": SourceExtractionState.PENDING,
            "extraction_error_code": code.value,
            "leased_until": None,
            "lease_owner": None,
            "available_at": func.clock_timestamp() + timedelta(seconds=delay),
        },
    )
    await db.commit()
    logger.warning(
        "policy source deferred source=%s tenant=%s code=%s retry_in=%.0fs",
        source.id,
        source.tenant_id,
        code.value,
        delay,
    )


def build_storage(settings: Settings) -> ObjectStorage:
    # Refuses to start without the policy bucket: it never falls back to the media bucket.
    if not settings.policy_source_bucket:
        raise WorkerConfigError("POLICY_SOURCE_BUCKET is missing (policy sources live there).")
    if settings.policy_extraction_timeout_seconds >= lease_seconds(settings):
        raise WorkerConfigError(
            "POLICY_EXTRACTION_TIMEOUT_SECONDS must be below WORKER_LEASE_SECONDS."
        )
    return ObjectStorage(
        settings,
        bucket=settings.policy_source_bucket,
        max_read_bytes=settings.policy_source_max_bytes,
    )


async def run_forever(settings: Settings) -> None:
    database_url = worker_database_url()
    storage = build_storage(settings)
    engine = create_engine(settings, database_url)
    sessions = create_sessionmaker(engine)
    stop = asyncio.Event()
    stop_on_signals(stop)
    logger.info("policy worker started")
    try:
        while not stop.is_set():
            try:
                leased = await run_once(sessions, storage, settings)
            except Exception:
                logger.exception("policy worker iteration failed")
                leased = 0
            if leased == 0:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), settings.worker_poll_interval_seconds)
    finally:
        await engine.dispose()
    logger.info("policy worker stopped")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        asyncio.run(run_forever(settings))
    except WorkerConfigError as error:
        logger.error("policy worker refused to start: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
