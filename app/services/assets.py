# Generated-image assets: the worker's storage sink and the read path behind GET /assets/{id}.

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.access import TenantContext
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.models import Asset, Creative, Job
from app.schemas.assets import AssetOrigin, AssetRead
from app.services import media
from app.services.media import MediaRejectedError
from app.services.providers.base import FetchedImage, ProviderError, ProviderOutputError
from app.services.storage import ObjectStorage, StorageError

logger = logging.getLogger(__name__)


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
            info = await asyncio.to_thread(
                media.inspect_image, image.content, max_pixels=self._settings.max_image_pixels
            )
        except MediaRejectedError as error:
            raise ProviderOutputError(str(error)) from error
        if info.mime_type != image.content_type:
            raise ProviderOutputError("The provider output type does not match its content.")
        try:
            stored = await self._storage.save_generated_image(
                image.content, ext=info.ext, content_type=info.mime_type
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
        logger.info("asset stored asset=%s job=%s tenant=%s", asset.id, job.id, job.tenant_id)
        return asset.id


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
    url = await storage.presigned_get(asset.s3_key, version_id=asset.s3_version_id, expires_in=ttl)
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
