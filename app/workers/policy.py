"""Policy worker (NRC-4): extracts clauses from policy sources and builds the index of bundles.

Run with `uv run python -m app.workers.policy`. It connects as marketing_worker and leases due
sources and bundles one at a time through lease_policy_sources() and lease_policy_builds(), which
return ids only; everything else runs in a session bound to the item's tenant, so row-level
security applies.

For each source it reads exactly the pinned object version from the policy bucket, extracts the
clauses in a child process that is killed on timeout (app.services.policy_extraction), and then,
in ONE transaction, inserts every clause and marks the source EXTRACTED. A crash or failure before
that commit leaves no clauses and a source that is not EXTRACTED: a source never shows a partial
clause set (AC4).

A file that cannot be turned into clauses ends as FAILED with a code. Temporary storage problems
and timeouts send the source back to PENDING with a backoff, up to POLICY_WORKER_MAX_ATTEMPTS
leases, then FAILED. A crash leaves the lease to expire; the lease function counts that as a failed
attempt, with the same cap. Every change is fenced on this worker's lease, so a slow or duplicate
worker can never write twice.

Build queue: approval (or the BUILD command) makes an APPROVED bundle due. The worker sets it
BUILDING (committed, so progress shows), then in ONE transaction builds the embedding index
(app.services.policy_index; none when EMBEDDING_MODE=off, labelled lexical_only in the audit row),
marks the bundle READY and writes the BUILD_READY audit row. A crash or failure before that commit
leaves no READY build and a bundle that is not READY: interrupted indexing never exposes a
half-built version (AC4). A permanent failure sends the bundle back to APPROVED, not queued, with
a FAILED build row and a BUILD_FAILED audit row; it is built again only when someone runs BUILD.
Provider outages back off like extraction, up to the same attempt cap.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Final, Protocol
from uuid import UUID

from sqlalchemy import TextClause, Update, func, insert, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import InstrumentedAttribute

from app.core.config import Settings, get_settings
from app.core.errors import PolicyIndexError
from app.core.logging import configure_logging
from app.db.session import bind_tenant, create_engine, create_sessionmaker
from app.models import MediaUpload, PolicyAuditEvent, PolicyBundle, PolicySource, PolicySourceClause
from app.models.enums import (
    BundleLifecycleState,
    ExtractionCode,
    PolicyAuditAction,
    PolicyBuildFailure,
    PolicyIndexReason,
    SourceExtractionState,
    UploadState,
)
from app.services.policy_extraction import (
    Extraction,
    ExtractionError,
    extract_in_child_process,
)
from app.services.policy_index import build_policy_index, record_failed_build
from app.services.providers.base import ProviderConfigError, ProviderError
from app.services.providers.embeddings import EmbeddingProvider, embedding_provider_from_settings
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
_LEASE_BUILDS = text("SELECT bundle_id, tenant_id FROM lease_policy_builds(:owner, 1, :lease)")
_IN_PROGRESS: Final = (SourceExtractionState.PENDING, SourceExtractionState.EXTRACTING)
_BUILD_STATES: Final = (BundleLifecycleState.APPROVED, BundleLifecycleState.BUILDING)
BUILD_ACTOR: Final = "system:policy-worker"
# Audit label (Q5): with EMBEDDING_MODE=off the bundle is READY with full-text search only.
LEXICAL_ONLY: Final = "lexical_only"
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


@dataclass(frozen=True, slots=True)
class _Queue:
    # What the lease/fence helpers need to know about one queue: its table, its state column and
    # the states in which an item is still unfinished.
    model: type[PolicySource] | type[PolicyBundle]
    state: InstrumentedAttribute[Any]
    unfinished: tuple[Enum, ...]


_SOURCES: Final = _Queue(PolicySource, PolicySource.extraction_state, _IN_PROGRESS)
_BUILDS: Final = _Queue(PolicyBundle, PolicyBundle.lifecycle_state, _BUILD_STATES)


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
    # None is EMBEDDING_MODE=off: bundles become READY with full-text search only.
    embedder: EmbeddingProvider | None = None
    # Set on SIGTERM: no new lease after it, or the kill that follows would turn each one into a
    # failed attempt.
    stop: asyncio.Event | None = None

    def stopping(self) -> bool:
        return self.stop is not None and self.stop.is_set()


type _Handler = Callable[[_Run, UUID, UUID], Awaitable[None]]


async def run_once(
    sessions: async_sessionmaker[AsyncSession],
    storage: ObjectReader,
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    owner: str = DEFAULT_OWNER,
    max_items: int | None = None,
    stop: asyncio.Event | None = None,
) -> int:
    # Lease and process up to max_items (default WORKER_MAX_EVENTS_PER_RUN) due sources and then
    # as many due bundles, one lease each, until stop is set. Returns how many were leased.
    limit = settings.worker_max_events_per_run if max_items is None else max_items
    run = _Run(sessions, storage, settings, owner, embedder, stop)
    extracted = await _drain(run, _LEASE_SOURCES, _extract, limit)
    return extracted + await _drain(run, _LEASE_BUILDS, _build, limit)


async def _drain(run: _Run, lease: TextClause, handle: _Handler, limit: int) -> int:
    # One queue: lease an item, handle it. A second queue is another lease statement and handler.
    leased = 0
    while leased < limit and not run.stopping():
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


def _fenced(queue: _Queue, item_id: UUID, owner: str) -> Update:
    # Only the worker whose lease is still valid may change the item, and only while it is
    # unfinished: terminal states never change again.
    model = queue.model
    return (
        update(model)
        .where(
            model.id == item_id,
            model.lease_owner == owner,
            model.leased_until > func.clock_timestamp(),
            queue.state.in_(queue.unfinished),
        )
        .execution_options(synchronize_session=False)
    )


async def _change(
    db: AsyncSession,
    run: _Run,
    item_id: UUID,
    values: Mapping[str, object],
    queue: _Queue = _SOURCES,
) -> None:
    changed = await db.scalar(
        _fenced(queue, item_id, run.owner).values(**values).returning(queue.model.id)
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
    # One deadline covers the read and the extraction, so together they stay under the lease.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.policy_extraction_timeout_seconds
    try:
        async with asyncio.timeout_at(deadline):
            content = await run.storage.read(upload.s3_key, version_id=upload.s3_version_id)
        return await extract_in_child_process(
            content,
            upload.mime_type,
            max_pages=settings.policy_source_max_pages,
            max_clauses=settings.policy_extraction_max_clauses,
            max_chars=settings.policy_extraction_max_chars,
            timeout_seconds=max(deadline - loop.time(), 0.0),
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


# ---- Build queue


async def _build(run: _Run, bundle_id: UUID, tenant_id: UUID) -> None:
    async with run.sessions() as db:
        await bind_tenant(db, tenant_id)
        bundle = await db.get(PolicyBundle, bundle_id, populate_existing=True)
        if (
            bundle is None
            or bundle.lease_owner != run.owner
            or bundle.lifecycle_state not in _BUILD_STATES
        ):
            raise LeaseLostError
        attempts = bundle.attempts
        model_id = run.embedder.model_id if run.embedder else None
        if bundle.failed_attempts >= run.settings.policy_worker_max_attempts:
            # Its leases kept expiring unfinished (crashes), up to the cap.
            await _fail_build(
                db, run, bundle_id, tenant_id, PolicyBuildFailure.BUILD_ABANDONED, None
            )
            return
        await _change(
            db,
            run,
            bundle_id,
            {"lifecycle_state": BundleLifecycleState.BUILDING, "updated_at": func.now()},
            _BUILDS,
        )
        await db.commit()
        # From here the session holds plain ids and numbers only: a rollback expires the bundle.
        try:
            await _index_and_publish(db, run, bundle_id, tenant_id)
        except PolicyIndexError as error:
            # A precondition (nothing to index, content changed): no build row, nothing to retry.
            await db.rollback()
            await _fail_build(db, run, bundle_id, tenant_id, _failure_code(error), None)
        except ProviderError as error:
            await db.rollback()
            # The response carries the grouped code only; the exact provider code is logged here
            # and kept on the internal build row.
            code = _failure_code(error)
            logger.warning(
                "policy bundle build embedding error bundle=%s tenant=%s provider_code=%s",
                bundle_id,
                tenant_id,
                error.code,
            )
            if error.retryable and attempts < run.settings.policy_worker_max_attempts:
                await _retry_build_later(db, run, bundle_id, tenant_id, attempts, code)
            else:
                failed_build = (model_id, error.code) if model_id is not None else None
                await _fail_build(db, run, bundle_id, tenant_id, code, failed_build)


def _failure_code(error: PolicyIndexError | ProviderError) -> PolicyBuildFailure:
    if isinstance(error, ProviderError):
        if error.retryable:
            return PolicyBuildFailure.EMBEDDING_UNAVAILABLE
        return PolicyBuildFailure.EMBEDDING_FAILED
    # PolicyBuildFailure reuses every PolicyIndexReason value.
    reason = (error.details or {}).get("reason")
    if isinstance(reason, str) and reason in PolicyIndexReason:
        return PolicyBuildFailure(reason)
    return PolicyBuildFailure.INDEX_REFUSED


async def _index_and_publish(db: AsyncSession, run: _Run, bundle_id: UUID, tenant_id: UUID) -> None:
    # The index rows, READY and its audit row commit together; anything raised before the commit
    # leaves none of them, and a worker whose lease ran out writes nothing (LeaseLostError).
    # shortcut: no lease heartbeat, so a build that outlasts WORKER_LEASE_SECONDS is abandoned and
    # retried until the attempt cap; add a heartbeat if real bundles ever take that long.
    details: dict[str, object] = {"index": LEXICAL_ONLY}
    if run.embedder is not None:
        built = await build_policy_index(db, tenant_id, bundle_id, run.embedder)
        details = {
            "build_id": str(built.build_id),
            "model_id": built.model_id,
            "node_count": built.node_count,
        }
    await _change(
        db,
        run,
        bundle_id,
        {
            "lifecycle_state": BundleLifecycleState.READY,
            "last_error_code": None,
            "updated_at": func.now(),
            **_RELEASE,
        },
        _BUILDS,
    )
    await _audit(
        db,
        tenant_id,
        bundle_id,
        PolicyAuditAction.BUILD_READY,
        BundleLifecycleState.BUILDING,
        BundleLifecycleState.READY,
        details,
    )
    await db.commit()
    logger.info("policy bundle ready bundle=%s tenant=%s", bundle_id, tenant_id)


async def _audit(
    db: AsyncSession,
    tenant_id: UUID,
    bundle_id: UUID,
    action: PolicyAuditAction,
    from_state: BundleLifecycleState,
    to_state: BundleLifecycleState,
    details: Mapping[str, object],
) -> None:
    # Core insert: the worker role may insert audit rows but not read them back (no RETURNING).
    await db.execute(
        insert(PolicyAuditEvent).values(
            tenant_id=tenant_id,
            policy_bundle_id=bundle_id,
            action=action,
            from_state=from_state,
            to_state=to_state,
            actor=BUILD_ACTOR,
            details=dict(details),
        )
    )


async def _fail_build(
    db: AsyncSession,
    run: _Run,
    bundle_id: UUID,
    tenant_id: UUID,
    code: PolicyBuildFailure,
    failed_build: tuple[str, str] | None,
) -> None:
    # Back to APPROVED and out of the queue (available_at NULL) until BUILD runs again. The
    # fenced change comes first, so a worker that lost its lease writes nothing. failed_build is
    # (model id, exact provider code) when an embedding call failed: the internal build row keeps
    # the exact code, the bundle shows the grouped one.
    await _change(
        db,
        run,
        bundle_id,
        {
            "lifecycle_state": BundleLifecycleState.APPROVED,
            "last_error_code": code,
            "updated_at": func.now(),
            **_RELEASE,
        },
        _BUILDS,
    )
    if failed_build is not None:
        await record_failed_build(db, tenant_id, bundle_id, *failed_build)
    await _audit(
        db,
        tenant_id,
        bundle_id,
        PolicyAuditAction.BUILD_FAILED,
        BundleLifecycleState.BUILDING,
        BundleLifecycleState.APPROVED,
        {"error_code": code},
    )
    await db.commit()
    logger.warning(
        "policy bundle build failed bundle=%s tenant=%s code=%s", bundle_id, tenant_id, code
    )


async def _retry_build_later(
    db: AsyncSession,
    run: _Run,
    bundle_id: UUID,
    tenant_id: UUID,
    attempts: int,
    code: PolicyBuildFailure,
) -> None:
    delay = backoff_seconds(attempts, run.settings)
    await _change(
        db,
        run,
        bundle_id,
        {
            "lifecycle_state": BundleLifecycleState.APPROVED,
            "last_error_code": code,
            "updated_at": func.now(),
            "leased_until": None,
            "lease_owner": None,
            "available_at": func.clock_timestamp() + timedelta(seconds=delay),
        },
        _BUILDS,
    )
    await db.commit()
    logger.warning(
        "policy bundle build deferred bundle=%s tenant=%s code=%s retry_in=%.0fs",
        bundle_id,
        tenant_id,
        code,
        delay,
    )


def build_embedder(settings: Settings) -> EmbeddingProvider | None:
    # None when EMBEDDING_MODE=off. Live mode with missing settings refuses to start; it never
    # falls back to the simulator.
    try:
        return embedding_provider_from_settings(settings)
    except ProviderConfigError as error:
        raise WorkerConfigError(error.message) from None


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
    embedder = build_embedder(settings)
    engine = create_engine(settings, database_url)
    sessions = create_sessionmaker(engine)
    stop = asyncio.Event()
    stop_on_signals(stop)
    logger.info("policy worker started")
    try:
        while not stop.is_set():
            try:
                leased = await run_once(sessions, storage, settings, embedder=embedder, stop=stop)
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
