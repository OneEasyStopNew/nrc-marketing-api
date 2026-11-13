# Generated-image assets: the worker's storage sink and the read path behind GET /assets/{id}.

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Final
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.errors import DependencyUnavailableError, NotFoundError
from app.models import Asset, Creative, Job, OutboxEvent
from app.models.enums import ExecutionState
from app.schemas.assets import AssetOrigin, AssetRead, MediaItem, MediaPage
from app.services import media, tenancy
from app.services.jobs import JobContentMissingError, job_content
from app.services.media import MediaRejectedError
from app.services.providers.base import FetchedImage, ProviderError, ProviderOutputError
from app.services.storage import ObjectStorage, StorageError

logger = logging.getLogger(__name__)

ASSET_GENERATED_EVENT: Final = "asset.generated"
# Payload contract version, agreed with the NRC-7 handler; bump it on any incompatible change.
ASSET_GENERATED_EVENT_VERSION: Final = 1


class ImageStorageUnavailableError(ProviderError):
    # Object storage failed: the worker keeps the event and tries again later.
    code = "STORAGE_UNAVAILABLE"
    retryable = True


class StoredImageSink:
    """Verifies a generated image, stores it and adds its asset row (GeneratedImageSink).

    It never commits: the worker commits the asset together with the job's SUCCEEDED transition.
    """

    def __init__(self, storage: ObjectStorage, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    async def store(
        self, db: AsyncSession, job: Job, image: FetchedImage, provider_name: str, model: str
    ) -> UUID:
        try:
            spec = (await job_content(db, job)).spec
        except JobContentMissingError as error:
            # Ends the job like any unusable output: billed, nothing stored, no retry.
            raise ProviderOutputError(str(error)) from error
        # The stored bytes are what image validation reads: exactly the format's size, as PNG.
        try:
            fitted = await asyncio.to_thread(
                media.fit_to_format,
                image.content,
                width_px=spec.width_px,
                height_px=spec.height_px,
                max_pixels=self._settings.max_image_pixels,
            )
        except MediaRejectedError as error:
            raise ProviderOutputError(str(error)) from error
        if fitted.source.mime_type != image.content_type:
            raise ProviderOutputError("The provider output type does not match its content.")
        info = fitted.stored
        try:
            stored = await self._storage.save_generated_image(
                fitted.content, ext=info.ext, content_type=info.mime_type
            )
        except StorageError as error:
            raise ImageStorageUnavailableError("Image storage is temporarily unavailable.") from (
                error
            )
        # If the worker crashes after the put but before its commit, the object is an orphan with
        # no asset row. A cleanup job for such objects is a production follow-up.
        asset = Asset(
            tenant_id=job.tenant_id,
            creative_id=job.creative_id,
            creative_version=job.creative_version,
            job_id=job.id,
            origin=AssetOrigin.GENERATED.value,
            bucket=stored.bucket,
            s3_key=stored.key,
            s3_version_id=stored.version_id,
            mime_type=info.mime_type,
            width_px=info.width_px,
            height_px=info.height_px,
            size_bytes=stored.size_bytes,
            checksum_sha256=stored.checksum_sha256,
            provider=provider_name,
            model=model,
            provider_request_id=job.provider_request_id,
            created_by=job.created_by,
        )
        db.add(asset)
        # Flush so a duplicate for the job fails here, not at the worker's commit.
        await db.flush()
        logger.info(
            "asset stored asset=%s job=%s tenant=%s provider_size=%sx%s stored_size=%sx%s",
            asset.id,
            job.id,
            job.tenant_id,
            fitted.source.width_px,
            fitted.source.height_px,
            info.width_px,
            info.height_px,
        )
        source = fitted.source
        if source.width_px < info.width_px or source.height_px < info.height_px:
            # Upscaling loses sharpness; validation and review should know. Ids and sizes only.
            logger.warning(
                "provider image smaller than its format, upscaled asset=%s job=%s tenant=%s "
                "provider_size=%sx%s stored_size=%sx%s",
                asset.id,
                job.id,
                job.tenant_id,
                source.width_px,
                source.height_px,
                info.width_px,
                info.height_px,
            )
        return asset.id


async def emit_asset_generated(db: AsyncSession, asset_id: UUID) -> None:
    """Queue the hand-off to image validation (NRC-7) for a stored asset.

    Called in the transaction that marks its job SUCCEEDED. The payload is ids and the storage
    location only, never a URL: validation reads our stored copy, never the provider's.
    """
    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise LookupError("The stored asset is not visible in this transaction")
    db.add(
        OutboxEvent(
            tenant_id=asset.tenant_id,
            job_id=asset.job_id,
            event_type=ASSET_GENERATED_EVENT,
            payload={
                "event_version": ASSET_GENERATED_EVENT_VERSION,
                "job_id": str(asset.job_id),
                "asset_id": str(asset.id),
                "tenant_id": str(asset.tenant_id),
                "creative_id": str(asset.creative_id),
                "creative_version": asset.creative_version,
                "bucket": asset.bucket,
                "s3_key": asset.s3_key,
                "s3_version_id": asset.s3_version_id,
                "checksum_sha256": asset.checksum_sha256,
                "mime_type": asset.mime_type,
                "width_px": asset.width_px,
                "height_px": asset.height_px,
            },
            attempts=0,
        )
    )
    logger.info(
        "asset.generated queued asset=%s job=%s tenant=%s", asset.id, asset.job_id, asset.tenant_id
    )


async def get_asset(
    db: AsyncSession,
    ctx: TenantContext,
    asset_id: UUID,
    storage: ObjectStorage,
    settings: Settings,
) -> AssetRead:
    # Tenant isolation comes from RLS plus the explicit tenant filter; the study check hides
    # assets of studies the caller cannot see. Both look like "not found".
    asset = (
        await db.execute(
            select(Asset)
            .join(
                Creative,
                (Creative.tenant_id == Asset.tenant_id) & (Creative.id == Asset.creative_id),
            )
            .where(
                Asset.tenant_id == ctx.tenant_id,
                Asset.id == asset_id,
                Creative.study_id.in_(ctx.study_ids),
            )
        )
    ).scalar_one_or_none()
    if asset is None:
        raise NotFoundError("Asset not found.")
    ttl = settings.asset_url_ttl_seconds
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
    # The URL is returned to the caller only: never stored or logged.
    try:
        url = await storage.presigned_get(
            asset.s3_key, version_id=asset.s3_version_id, expires_in=ttl
        )
    except StorageError:
        raise DependencyUnavailableError("Image storage is temporarily unavailable.") from None
    return AssetRead(
        asset_id=asset.id,
        creative_id=asset.creative_id,
        origin=AssetOrigin(asset.origin),
        mime_type=asset.mime_type,
        width_px=asset.width_px,
        height_px=asset.height_px,
        size_bytes=asset.size_bytes,
        checksum_sha256=asset.checksum_sha256,
        provider=asset.provider,
        model=asset.model,
        created_at=asset.created_at,
        download_url=url,
        download_url_expires_at=expires_at,
    )


async def list_media(
    db: AsyncSession,
    ctx: TenantContext,
    storage: ObjectStorage,
    settings: Settings,
    *,
    before: datetime | None,
    limit: int,
) -> MediaPage:
    """List completed stored images with the same tenant/study boundary as get_asset.

    Keep the existing timestamp cursor contract. Include timestamp ties at the page boundary
    rather than silently dropping them on the next page (which uses a strict `<` cursor).
    """
    query = (
        select(Asset)
        .join(
            Creative,
            (Creative.tenant_id == Asset.tenant_id) & (Creative.id == Asset.creative_id),
        )
        .join(Job, (Job.tenant_id == Asset.tenant_id) & (Job.id == Asset.job_id))
        .where(
            Asset.tenant_id == ctx.tenant_id,
            Creative.study_id.in_(ctx.study_ids),
            Asset.origin == AssetOrigin.GENERATED.value,
            Asset.mime_type.in_(("image/png", "image/jpeg")),
            Job.state == ExecutionState.SUCCEEDED,
            Job.result_asset_id == Asset.id,
        )
        .order_by(Asset.created_at.desc())
    )
    if before is not None:
        query = query.where(Asset.created_at < before)
    # WITH TIES applies only to created_at; sort equal timestamps by id below for a stable
    # display order. Check for older rows separately: even limit+1 could land inside a tie.
    rows = (await db.scalars(query.fetch(limit, with_ties=True))).all()
    page = sorted(
        rows,
        key=lambda row: (row.created_at, row.id),
        reverse=True,
    )
    next_cursor = None
    if page:
        boundary = page[-1].created_at
        older = await db.scalar(
            query.with_only_columns(Asset.id).where(Asset.created_at < boundary).limit(1)
        )
        if older is not None:
            next_cursor = boundary
    names = await tenancy.display_names(db, ctx, {row.created_by for row in page})
    ttl = settings.asset_url_ttl_seconds
    items: list[MediaItem] = []
    for asset in page:
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
        try:
            url = await storage.presigned_get(
                asset.s3_key, version_id=asset.s3_version_id, expires_in=ttl
            )
        except StorageError:
            raise DependencyUnavailableError("Image storage is temporarily unavailable.") from None
        items.append(
            MediaItem(
                asset_id=asset.id,
                creative_id=asset.creative_id,
                creative_version=asset.creative_version,
                origin=AssetOrigin(asset.origin),
                mime_type=asset.mime_type,
                width_px=asset.width_px,
                height_px=asset.height_px,
                size_bytes=asset.size_bytes,
                created_at=asset.created_at,
                created_by=asset.created_by,
                created_by_name=names.get(asset.created_by),
                download_url=url,
                download_url_expires_at=expires_at,
            )
        )
    return MediaPage(items=items, next_cursor=next_cursor)
