"""Media worker (NRC-3): inspects completed client uploads and turns clean ones into assets.

Run with `uv run python -m app.workers.media`. It connects as marketing_worker and leases due
uploads one at a time through lease_media_uploads(), which returns ids only; everything else runs
in a session bound to the upload's tenant, so row-level security applies.

For each upload it reads exactly the pinned S3 version, then checks, in order: size and SHA-256
against the declaration, the real image type and dimensions (decoded, with the pixel cap), the
type against the declaration and the accepted types, and finally the malware scan. Only a clean
upload becomes READY, in the same transaction as its immutable assets row; any failed check ends
it as REJECTED with a safe reason. Nothing reaches validation or generation before that.

Delivery is at least once. Every change is fenced on this worker's lease and guarded on the
state, so a crashed, slow or duplicate worker can never move an upload twice or regress it. A
crash leaves the lease to expire; the lease function counts that as a failed attempt, and after
MEDIA_INSPECTION_MAX_ATTEMPTS failures the upload is rejected as INSPECTION_FAILED. Temporary
scanner or storage outages keep it QUARANTINED with a retryable reason until
MEDIA_INSPECTION_RETRY_WINDOW_SECONDS after completion.
"""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from uuid import UUID

from sqlalchemy import ColumnElement, Update, func, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import bind_tenant, create_engine, create_sessionmaker
from app.models import Asset, MediaUpload
from app.models.enums import UploadState
from app.schemas.assets import AssetOrigin
from app.services import media
from app.services.media import MediaRejectedError, configure_pillow
from app.services.scanning import (
    MalwareScanner,
    ScannerNotConfiguredError,
    ScanOutcome,
    ScanUnavailableError,
    ScanVerdict,
    build_scanner,
)
from app.services.storage import ObjectStorage, StorageError
from app.services.uploads import reason_columns
from app.workers.common import (
    DEFAULT_OWNER,
    WorkerConfigError,
    backoff_seconds,
    lease_seconds,
    stop_on_signals,
    worker_database_url,
)

logger = logging.getLogger(__name__)

_LEASE = text("SELECT upload_id, tenant_id FROM lease_media_uploads(:owner, 1, :lease)")
_INSPECTING: Final = (UploadState.QUARANTINED, UploadState.SCANNING)


class LeaseLostError(Exception):
    # Another worker leased the upload after this one's lease ran out; leave it to that worker.
    pass


@dataclass(frozen=True, slots=True)
class _Run:
    sessions: async_sessionmaker[AsyncSession]
    storage: ObjectStorage
    scanner: MalwareScanner
    settings: Settings
    owner: str
    lease_seconds: int


async def run_once(
    sessions: async_sessionmaker[AsyncSession],
    storage: ObjectStorage,
    scanner: MalwareScanner,
    settings: Settings,
    *,
    owner: str = DEFAULT_OWNER,
    max_uploads: int | None = None,
) -> int:
    # Lease and inspect up to max_uploads (default WORKER_MAX_EVENTS_PER_RUN) due uploads, one
    # lease each. Returns how many were leased.
    if max_uploads is None:
        max_uploads = settings.worker_max_events_per_run
    run = _Run(sessions, storage, scanner, settings, owner, lease_seconds(settings))
    leased = 0
    while leased < max_uploads:
        async with sessions() as db:
            row = (
                await db.execute(_LEASE, {"owner": owner, "lease": run.lease_seconds})
            ).one_or_none()
            await db.commit()
        if row is None:
            break
        leased += 1
        try:
            await _inspect(run, row.upload_id, row.tenant_id)
        except LeaseLostError:
            logger.warning("upload lease lost upload=%s tenant=%s", row.upload_id, row.tenant_id)
        except Exception:
            # The lease expires unfinished, which the next lease counts as a failed attempt.
            logger.exception(
                "upload inspection failed upload=%s tenant=%s", row.upload_id, row.tenant_id
            )
    return leased


def _fenced(upload_id: UUID, owner: str) -> Update:
    # Only the worker whose lease is still valid may change the upload, and only while it is
    # being inspected: terminal states never change again.
    return (
        update(MediaUpload)
        .where(
            MediaUpload.id == upload_id,
            MediaUpload.lease_owner == owner,
            MediaUpload.leased_until > func.clock_timestamp(),
            MediaUpload.state.in_(_INSPECTING),
        )
        .execution_options(synchronize_session=False)
    )


async def _change(
    db: AsyncSession, run: _Run, upload_id: UUID, values: Mapping[str, object]
) -> None:
    changed = await db.scalar(
        _fenced(upload_id, run.owner)
        .values(updated_at=func.now(), **values)
        .returning(MediaUpload.id)
    )
    if changed is None:
        raise LeaseLostError


_RELEASE: Final[Mapping[str, object]] = {"leased_until": None, "lease_owner": None}


async def _inspect(run: _Run, upload_id: UUID, tenant_id: UUID) -> None:
    async with run.sessions() as db:
        await bind_tenant(db, tenant_id)
        upload = await db.get(MediaUpload, upload_id, populate_existing=True)
        if upload is None or upload.lease_owner != run.owner or upload.state not in _INSPECTING:
            raise LeaseLostError
        if upload.failed_attempts >= run.settings.media_inspection_max_attempts:
            await _reject(db, run, upload, "INSPECTION_FAILED")
            return
        # Visible progress, and a fresh lease before the slow part.
        await _change(
            db,
            run,
            upload.id,
            {"state": UploadState.SCANNING, "leased_until": _renewed_lease(run)},
        )
        await db.commit()
        async with _kept_leased(run, upload.id, upload.tenant_id):
            found = await _examine(run, upload)
        await _record(db, run, upload, found)


def _renewed_lease(run: _Run) -> ColumnElement[datetime]:
    return func.clock_timestamp() + timedelta(seconds=run.lease_seconds)


@asynccontextmanager
async def _kept_leased(run: _Run, upload_id: UUID, tenant_id: UUID) -> AsyncIterator[None]:
    """Renew the lease in the background while the upload is read, decoded and scanned.

    A slow object read (the S3 client retries) or a real scanner can outlast one lease; without
    renewal another worker would take the upload over and count a failure, and enough of those
    reject a valid file. Renewal stops on a lost lease or a database error; the fenced writes
    afterwards then fail safely instead of acting on a lease this worker no longer holds.
    """

    async def renew() -> None:
        while True:
            await asyncio.sleep(run.lease_seconds / 3)
            try:
                async with run.sessions() as db:
                    await bind_tenant(db, tenant_id)
                    await _change(db, run, upload_id, {"leased_until": _renewed_lease(run)})
                    await db.commit()
            except LeaseLostError:
                return
            except Exception:
                logger.exception(
                    "upload lease renewal failed upload=%s tenant=%s", upload_id, tenant_id
                )
                return

    task = asyncio.create_task(renew())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@dataclass(frozen=True, slots=True)
class _Found:
    # What examining the object found. Exactly one of reject, retry or (info and verdict) is set.
    reject: str | None = None
    retry: str | None = None
    info: media.ImageInfo | None = None
    verdict: ScanVerdict | None = None


def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def _examine(run: _Run, upload: MediaUpload) -> _Found:
    # No database access here: this part may take long and runs while the lease is renewed.
    settings = run.settings
    try:
        content = await run.storage.read(upload.s3_key, version_id=upload.s3_version_id)
    except StorageError:
        return _Found(retry="STORAGE_UNAVAILABLE")
    if len(content) != upload.declared_size_bytes:
        return _Found(reject="SIZE_MISMATCH")
    # CPU-bound on files up to UPLOAD_IMAGE_MAX_BYTES: off the event loop, like the decode.
    if await asyncio.to_thread(_sha256_hex, content) != upload.declared_checksum_sha256:
        return _Found(reject="CHECKSUM_MISMATCH")
    try:
        # The real type comes from decoding the file, never from its name or declared type.
        info = await asyncio.to_thread(
            media.inspect_image, content, max_pixels=settings.max_image_pixels
        )
    except MediaRejectedError as error:
        return _Found(reject=error.code)
    if info.mime_type != upload.declared_mime_type:
        return _Found(reject="TYPE_MISMATCH")
    if info.mime_type not in settings.upload_image_types:
        return _Found(reject="MEDIA_UNSUPPORTED_TYPE")
    try:
        verdict = await run.scanner.scan(
            content, bucket=upload.bucket, key=upload.s3_key, version_id=upload.s3_version_id
        )
    except ScanUnavailableError:
        return _Found(retry="SCAN_UNAVAILABLE")
    return _Found(info=info, verdict=verdict)


async def _record(db: AsyncSession, run: _Run, upload: MediaUpload, found: _Found) -> None:
    if found.retry is not None:
        await _retry_later(db, run, upload, found.retry)
        return
    if found.reject is not None or found.info is None or found.verdict is None:
        await _reject(db, run, upload, found.reject or "INSPECTION_FAILED")
        return
    verdict = found.verdict
    scan = {
        "scan_engine": verdict.engine,
        "scan_simulated": verdict.simulated,
        "scanned_at": func.now(),
    }
    if verdict.outcome is ScanOutcome.INFECTED:
        await _reject(db, run, upload, "MALWARE_DETECTED", extra=scan)
        return
    if verdict.outcome is not ScanOutcome.CLEAN:
        # Only an explicit CLEAN verdict passes; anything else is a scanner fault.
        await _retry_later(db, run, upload, "SCAN_UNAVAILABLE")
        return
    await _ready(db, run, upload, found.info, verdict, scan)


async def _ready(
    db: AsyncSession,
    run: _Run,
    upload: MediaUpload,
    info: media.ImageInfo,
    verdict: ScanVerdict,
    scan: Mapping[str, object],
) -> None:
    await _change(
        db,
        run,
        upload.id,
        {
            "state": UploadState.READY,
            "mime_type": info.mime_type,
            "width_px": info.width_px,
            "height_px": info.height_px,
            "reason_code": None,
            "reason_message": None,
            "available_at": None,
            **scan,
            **_RELEASE,
        },
    )
    # The asset is what later steps use; it exists only for a READY upload, with the same id.
    db.add(
        Asset(
            id=upload.id,
            tenant_id=upload.tenant_id,
            creative_id=upload.creative_id,
            study_id=upload.study_id,
            upload_id=upload.id,
            name=upload.name,
            origin=AssetOrigin.UPLOADED.value,
            bucket=upload.bucket,
            s3_key=upload.s3_key,
            s3_version_id=upload.s3_version_id,
            mime_type=info.mime_type,
            width_px=info.width_px,
            height_px=info.height_px,
            size_bytes=upload.declared_size_bytes,
            checksum_sha256=upload.declared_checksum_sha256,
            created_by=upload.created_by,
        )
    )
    await db.commit()
    logger.info(
        "upload ready upload=%s tenant=%s scan_engine=%s simulated=%s",
        upload.id,
        upload.tenant_id,
        verdict.engine,
        verdict.simulated,
    )
    if verdict.simulated:
        # Never let a simulated scan pass unnoticed in the logs either.
        logger.warning(
            "upload passed a SIMULATED malware scan upload=%s tenant=%s",
            upload.id,
            upload.tenant_id,
        )


async def _reject(
    db: AsyncSession,
    run: _Run,
    upload: MediaUpload,
    code: str,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    await _change(
        db,
        run,
        upload.id,
        {
            "state": UploadState.REJECTED,
            "available_at": None,
            **reason_columns(code),
            **(extra or {}),
            **_RELEASE,
        },
    )
    await db.commit()
    logger.info("upload rejected upload=%s tenant=%s code=%s", upload.id, upload.tenant_id, code)


async def _retry_later(db: AsyncSession, run: _Run, upload: MediaUpload, code: str) -> None:
    # A temporary outage: wait in QUARANTINED with a retryable reason, within the retry window.
    completed_at = upload.completed_at or upload.created_at
    window = timedelta(seconds=run.settings.media_inspection_retry_window_seconds)
    if datetime.now(UTC) >= completed_at + window:
        await _reject(db, run, upload, "INSPECTION_FAILED")
        return
    delay = backoff_seconds(upload.attempts, run.settings)
    await _change(
        db,
        run,
        upload.id,
        {
            "state": UploadState.QUARANTINED,
            "available_at": func.clock_timestamp() + timedelta(seconds=delay),
            **reason_columns(code),
            **_RELEASE,
        },
    )
    await db.commit()
    logger.warning(
        "upload inspection deferred upload=%s tenant=%s code=%s retry_in=%.0fs",
        upload.id,
        upload.tenant_id,
        code,
        delay,
    )


def build_dependencies(settings: Settings) -> tuple[ObjectStorage, MalwareScanner]:
    # Refuses to start without storage or a scanner; never falls back to passing files.
    if not settings.aws_s3_bucket:
        raise WorkerConfigError("AWS_S3_BUCKET is missing (uploads live in object storage).")
    try:
        scanner = build_scanner(settings)
    except ScannerNotConfiguredError as error:
        raise WorkerConfigError(str(error)) from None
    return ObjectStorage(settings), scanner


async def run_forever(settings: Settings) -> None:
    database_url = worker_database_url()
    storage, scanner = build_dependencies(settings)
    engine = create_engine(settings, database_url)
    sessions = create_sessionmaker(engine)
    stop = asyncio.Event()
    stop_on_signals(stop)
    logger.info(
        "media worker started scan_engine=%s simulated=%s", scanner.engine, scanner.simulated
    )
    try:
        while not stop.is_set():
            try:
                leased = await run_once(sessions, storage, scanner, settings)
            except Exception:
                logger.exception("media worker iteration failed")
                leased = 0
            if leased == 0:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), settings.worker_poll_interval_seconds)
    finally:
        await engine.dispose()
    logger.info("media worker stopped")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_pillow(settings)
    try:
        asyncio.run(run_forever(settings))
    except WorkerConfigError as error:
        logger.error("media worker refused to start: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
