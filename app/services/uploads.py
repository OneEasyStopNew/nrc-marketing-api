# Client uploads (NRC-3): issue an upload URL, accept the completed file into quarantine, and
# read upload status. Inspection (type, size, malware) runs in the media worker.

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.errors import (
    DependencyUnavailableError,
    NotFoundError,
    PermissionDeniedError,
    UploadNotAcceptedError,
    UploadNotPendingError,
    UploadNotReceivedError,
)
from app.models import Creative, MediaUpload, Study
from app.models.enums import Channel, IdentifiablePerson, UploadPurpose, UploadState
from app.models.uploads import UPLOAD_REASON_MAX_CHARS
from app.schemas.uploads import (
    AuthorizationStatus,
    MediaAuthorization,
    UploadCreate,
    UploadLimits,
    UploadPage,
    UploadRead,
    UploadReason,
    UploadScan,
    UploadTarget,
)
from app.services import idempotency
from app.services.paging import newest_first_page
from app.services.storage import ObjectStorage, PresignedUpload, StorageError, new_upload_key

logger = logging.getLogger(__name__)

CREATE_ROUTE: Final = "POST /assets/uploads"


@dataclass(frozen=True, slots=True)
class ReasonText:
    message: str
    retryable: bool


# Every reason an upload can carry, with the text shown to the uploader. Retryable reasons keep
# the upload QUARANTINED while the worker tries again; the others end it as REJECTED.
REASONS: Final[Mapping[str, ReasonText]] = MappingProxyType(
    {
        "MEDIA_EMPTY": ReasonText("The file is empty.", False),
        "MEDIA_UNSUPPORTED_TYPE": ReasonText(
            "This file type is not accepted. See the accepted types in the upload limits.", False
        ),
        "MEDIA_TOO_LARGE": ReasonText("The image has too many pixels.", False),
        "MEDIA_CORRUPT": ReasonText("The file is not a valid, complete image.", False),
        "TYPE_MISMATCH": ReasonText(
            "The file's real type does not match the type declared for it.", False
        ),
        "SIZE_MISMATCH": ReasonText("The file size does not match the size declared.", False),
        "CHECKSUM_MISMATCH": ReasonText(
            "The file does not match the checksum declared for it.", False
        ),
        "MALWARE_DETECTED": ReasonText("The malware scan flagged this file.", False),
        "INSPECTION_FAILED": ReasonText("The file could not be checked. Upload it again.", False),
        "SCAN_UNAVAILABLE": ReasonText(
            "The malware scan is temporarily unavailable. The check will be retried.", True
        ),
        "STORAGE_UNAVAILABLE": ReasonText(
            "Storage is temporarily unavailable. The check will be retried.", True
        ),
    }
)


def upload_limits(settings: Settings) -> UploadLimits:
    return UploadLimits(
        image_types=list(settings.upload_image_types),
        image_max_bytes=settings.upload_image_max_bytes,
        upload_url_ttl_seconds=settings.upload_url_ttl_seconds,
    )


def authorization_status(
    identifiable_person: IdentifiablePerson,
    usage_authorization_ref: str | None,
    expires_at: datetime | None,
    now: datetime,
) -> AuthorizationStatus:
    # Derived on every read: expiry depends on the time of the read, not of the upload.
    if expires_at is not None and expires_at <= now:
        return AuthorizationStatus.EXPIRED
    if identifiable_person is IdentifiablePerson.NO:
        return AuthorizationStatus.NO_PERSON_DECLARED
    if usage_authorization_ref is None:
        return AuthorizationStatus.MISSING
    return AuthorizationStatus.RECORDED


def _known_channels(upload: MediaUpload) -> list[Channel]:
    # A stored value that is no longer a Channel is dropped, never trusted: fewer permitted
    # channels only narrows what the gates allow. Logged so the stale value gets noticed.
    known = [Channel(value) for value in upload.permitted_channels if value in Channel]
    dropped = len(upload.permitted_channels) - len(known)
    if dropped:
        logger.warning(
            "upload has unknown permitted channels upload=%s tenant=%s dropped=%s",
            upload.id,
            upload.tenant_id,
            dropped,
        )
    return known


def media_authorization(upload: MediaUpload, now: datetime) -> MediaAuthorization:
    return MediaAuthorization(
        identifiable_person=upload.identifiable_person,
        usage_authorization_ref=upload.usage_authorization_ref,
        permitted_channels=_known_channels(upload),
        expires_at=upload.authorization_expires_at,
        status=authorization_status(
            upload.identifiable_person,
            upload.usage_authorization_ref,
            upload.authorization_expires_at,
            now,
        ),
    )


def reason_columns(code: str) -> dict[str, object]:
    # Column values for a reason from REASONS; an unknown code is a programming error.
    return {"reason_code": code, "reason_message": REASONS[code].message[:UPLOAD_REASON_MAX_CHARS]}


def _visible(ctx: TenantContext) -> tuple[ColumnElement[bool], ...]:
    # RLS already hides other tenants; the study check hides studies the caller cannot use.
    return (MediaUpload.tenant_id == ctx.tenant_id, MediaUpload.study_id.in_(ctx.study_ids))


def _to_read(
    upload: MediaUpload, now: datetime, target: PresignedUpload | None = None
) -> UploadRead:
    reason = None
    if upload.reason_code is not None:
        known = REASONS.get(upload.reason_code)
        reason = UploadReason(
            code=upload.reason_code,
            message=upload.reason_message or "",
            retryable=known.retryable if known else False,
        )
    scan = None
    if upload.scan_engine is not None and upload.scanned_at is not None:
        scan = UploadScan(
            engine=upload.scan_engine,
            simulated=bool(upload.scan_simulated),
            scanned_at=upload.scanned_at,
        )
    return UploadRead(
        asset_id=upload.id,
        study_id=upload.study_id,
        creative_id=upload.creative_id,
        purpose=upload.purpose,
        status=upload.state,
        reason=reason,
        name=upload.name,
        mime_type=upload.declared_mime_type,
        size_bytes=upload.declared_size_bytes,
        checksum_sha256=upload.declared_checksum_sha256,
        width_px=upload.width_px,
        height_px=upload.height_px,
        scan=scan,
        authorization=media_authorization(upload, now),
        created_by=upload.created_by,
        created_at=upload.created_at,
        completed_at=upload.completed_at,
        upload=(
            UploadTarget(
                url=target.url,
                method="PUT",
                headers=dict(target.headers),
                expires_at=target.expires_at,
            )
            if target is not None
            else None
        ),
    )


async def _load(db: AsyncSession, ctx: TenantContext, asset_id: UUID) -> MediaUpload:
    upload = await db.scalar(select(MediaUpload).where(*_visible(ctx), MediaUpload.id == asset_id))
    if upload is None:
        raise NotFoundError("Upload not found.")
    return upload


def _check_uploader(ctx: TenantContext, upload: MediaUpload) -> None:
    # The URL and the completion belong to whoever asked for them, like the object they write.
    if upload.created_by != ctx.principal.sub:
        raise PermissionDeniedError("Only the person who started this upload can do this.")


def _check_limits(body: UploadCreate, settings: Settings) -> None:
    if body.mime_type not in settings.upload_image_types:
        raise UploadNotAcceptedError(
            "This file type is not accepted.",
            details={"field": "mime_type", "accepted": list(settings.upload_image_types)},
        )
    if body.size_bytes > settings.upload_image_max_bytes:
        raise UploadNotAcceptedError(
            "This file is larger than the upload limit.",
            details={"field": "size_bytes", "max_bytes": settings.upload_image_max_bytes},
        )


async def _check_scope(db: AsyncSession, ctx: TenantContext, body: UploadCreate) -> None:
    if body.study_id not in ctx.study_ids:
        raise NotFoundError("Study not found.")
    study = await db.scalar(
        select(Study.id).where(Study.tenant_id == ctx.tenant_id, Study.id == body.study_id)
    )
    if study is None:
        raise NotFoundError("Study not found.")
    if body.creative_id is not None:
        creative = await db.scalar(
            select(Creative.id).where(
                Creative.tenant_id == ctx.tenant_id,
                Creative.id == body.creative_id,
                Creative.study_id == body.study_id,
            )
        )
        if creative is None:
            raise NotFoundError("Creative not found in this study.")


async def _presign(
    storage: ObjectStorage, upload: MediaUpload, settings: Settings
) -> PresignedUpload:
    try:
        return await storage.presigned_put(
            upload.s3_key,
            content_type=upload.declared_mime_type,
            size_bytes=upload.declared_size_bytes,
            checksum_sha256=upload.declared_checksum_sha256,
            expires_in=settings.upload_url_ttl_seconds,
        )
    except StorageError:
        raise DependencyUnavailableError("Image storage is temporarily unavailable.") from None


async def _fresh_target(
    db: AsyncSession, upload: MediaUpload, storage: ObjectStorage, settings: Settings
) -> PresignedUpload:
    # A new URL for the same key, checksum and owner; the old one simply expires.
    target = await _presign(storage, upload, settings)
    await db.execute(
        update(MediaUpload)
        .where(MediaUpload.id == upload.id, MediaUpload.state == UploadState.PENDING_UPLOAD)
        .values(upload_url_expires_at=target.expires_at, updated_at=func.now())
        .execution_options(synchronize_session=False)
    )
    return target


async def create_upload(
    db: AsyncSession,
    ctx: TenantContext,
    body: UploadCreate,
    idempotency_key: str,
    storage: ObjectStorage,
    settings: Settings,
) -> tuple[UploadRead, bool]:
    """Record a pending upload and sign a URL for exactly the declared file.

    Returns the upload and whether this was a replay. A replay of a still-pending upload gets a
    fresh URL for the same object.

    Raises:
        UploadNotAcceptedError: the declared type or size breaks a configured limit.
        NotFoundError: the study (or creative) does not exist or is not visible to the caller.
        IdempotencyKeyReusedError: the key was used before with a different payload.
        DependencyUnavailableError: storage could not sign the URL.
    """
    fingerprint = idempotency.request_hash(ctx.principal.sub, body)
    existing_id = await idempotency.claim(
        db, ctx.tenant_id, CREATE_ROUTE, idempotency_key, fingerprint
    )
    if existing_id is not None:
        upload = await _load(db, ctx, existing_id)
        target = None
        if upload.state is UploadState.PENDING_UPLOAD and upload.created_by == ctx.principal.sub:
            target = await _fresh_target(db, upload, storage, settings)
            await db.commit()
        return _to_read(upload, datetime.now(UTC), target), True

    _check_limits(body, settings)
    await _check_scope(db, ctx, body)
    authorization = body.authorization
    upload = MediaUpload(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        study_id=body.study_id,
        creative_id=body.creative_id,
        purpose=UploadPurpose.CREATIVE_MEDIA,
        state=UploadState.PENDING_UPLOAD,
        name=body.name,
        declared_mime_type=body.mime_type,
        declared_size_bytes=body.size_bytes,
        declared_checksum_sha256=body.checksum_sha256,
        bucket=storage.bucket,
        s3_key=new_upload_key(body.mime_type),
        identifiable_person=authorization.identifiable_person,
        usage_authorization_ref=authorization.usage_authorization_ref,
        # Order kept, duplicates dropped.
        permitted_channels=list(dict.fromkeys(c.value for c in authorization.permitted_channels)),
        authorization_expires_at=authorization.expires_at,
        created_by=ctx.principal.sub,
    )
    target = await _presign(storage, upload, settings)
    upload.upload_url_expires_at = target.expires_at
    db.add(upload)
    await db.flush()
    await idempotency.complete(db, ctx.tenant_id, CREATE_ROUTE, idempotency_key, upload.id)
    await db.commit()
    await db.refresh(upload)
    logger.info("upload created upload=%s tenant=%s", upload.id, upload.tenant_id)
    return _to_read(upload, datetime.now(UTC), target), False


async def reissue_upload_url(
    db: AsyncSession,
    ctx: TenantContext,
    asset_id: UUID,
    storage: ObjectStorage,
    settings: Settings,
) -> UploadRead:
    """A fresh URL for a pending upload whose URL expired or was lost. Same object and owner."""
    upload = await _load(db, ctx, asset_id)
    _check_uploader(ctx, upload)
    if upload.state is not UploadState.PENDING_UPLOAD:
        raise UploadNotPendingError(details={"status": upload.state.value})
    target = await _fresh_target(db, upload, storage, settings)
    await db.commit()
    await db.refresh(upload)
    return _to_read(upload, datetime.now(UTC), target)


async def complete_upload(
    db: AsyncSession, ctx: TenantContext, asset_id: UUID, storage: ObjectStorage
) -> UploadRead:
    """Check the stored file against the declaration and queue it for inspection.

    Idempotent: a repeated or concurrent call returns the current status, and only one call
    moves the upload to QUARANTINED, so it is inspected once.

    Raises:
        UploadNotReceivedError: the file is not in storage yet (retryable).
        DependencyUnavailableError: storage could not be reached.
    """
    upload = await _load(db, ctx, asset_id)
    _check_uploader(ctx, upload)
    if upload.state is not UploadState.PENDING_UPLOAD:
        return _to_read(upload, datetime.now(UTC))
    try:
        head = await storage.head(upload.s3_key)
    except StorageError:
        raise DependencyUnavailableError("Image storage is temporarily unavailable.") from None
    if head is None:
        raise UploadNotReceivedError()

    # Storage refuses a PUT that breaks the signed size or checksum; these are a second check.
    values: dict[str, object]
    if head.size_bytes != upload.declared_size_bytes:
        values = {"state": UploadState.REJECTED, **reason_columns("SIZE_MISMATCH")}
    elif head.checksum_sha256 not in (None, upload.declared_checksum_sha256):
        values = {"state": UploadState.REJECTED, **reason_columns("CHECKSUM_MISMATCH")}
    else:
        values = {"state": UploadState.QUARANTINED, "available_at": func.now()}
    moved = await db.scalar(
        update(MediaUpload)
        .where(
            MediaUpload.tenant_id == ctx.tenant_id,
            MediaUpload.id == upload.id,
            MediaUpload.state == UploadState.PENDING_UPLOAD,
        )
        .values(
            s3_version_id=head.version_id,
            completed_by=ctx.principal.sub,
            completed_at=func.now(),
            updated_at=func.now(),
            **values,
        )
        .returning(MediaUpload.id)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    if moved is not None:
        logger.info(
            "upload completed upload=%s tenant=%s state=%s",
            upload.id,
            upload.tenant_id,
            values["state"],
        )
    await db.refresh(upload)
    return _to_read(upload, datetime.now(UTC))


async def get_upload(db: AsyncSession, ctx: TenantContext, asset_id: UUID) -> UploadRead:
    return _to_read(await _load(db, ctx, asset_id), datetime.now(UTC))


async def list_uploads(
    db: AsyncSession,
    ctx: TenantContext,
    *,
    statuses: Collection[UploadState],
    study_id: UUID | None,
    before: datetime | None,
    limit: int,
) -> UploadPage:
    """Uploads the caller can see, newest first, with the same tie-safe cursor as GET /media."""
    query = select(MediaUpload).where(*_visible(ctx))
    if statuses:
        query = query.where(MediaUpload.state.in_(statuses))
    if study_id is not None:
        query = query.where(MediaUpload.study_id == study_id)
    page, next_cursor = await newest_first_page(
        db,
        query,
        created_at=MediaUpload.created_at,
        row_id=MediaUpload.id,
        sort_key=lambda row: (row[0].created_at, row[0].id),
        before=before,
        limit=limit,
    )
    now = datetime.now(UTC)
    return UploadPage(items=[_to_read(row[0], now) for row in page], next_cursor=next_cursor)
