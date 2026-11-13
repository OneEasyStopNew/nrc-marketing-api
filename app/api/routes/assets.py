# Stored assets (metadata plus a short-lived download URL) and the media library.
# With DEMO_STUBS=true the demo stubs answer instead: their images live in memory, not in storage.

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import AwareDatetime

from app.api.deps import (
    CreativeReader,
    DbSession,
    DemoStubsDep,
    OptionalStorageDep,
    SettingsDep,
    require_storage,
)
from app.schemas.assets import AssetRead, MediaPage
from app.schemas.common import ErrorBody
from app.services import assets

router = APIRouter(tags=["assets"])


@router.get(
    "/assets/{asset_id}",
    summary="Asset metadata and a short-lived download URL (D3)",
    description="The URL is not stored or logged; ask again when it expires.",
    responses={
        404: {"model": ErrorBody, "description": "Asset not found or not permitted."},
        503: {
            "model": ErrorBody,
            "description": "Object storage is not configured or temporarily unavailable.",
        },
    },
)
async def get_asset(
    asset_id: UUID,
    ctx: CreativeReader,
    db: DbSession,
    stubs: DemoStubsDep,
    storage: OptionalStorageDep,
    settings: SettingsDep,
) -> AssetRead:
    if stubs is not None:
        return await stubs.get_asset(db, ctx, asset_id)
    return await assets.get_asset(db, ctx, asset_id, require_storage(storage), settings)


@router.get(
    "/media",
    summary="Generated images in this tenant, newest first (read-only)",
    description="Only finished images from studies the caller can use. Uploads are not "
    "supported yet. download_url is short-lived: list again, or GET /assets/{asset_id}, for a "
    "fresh one. A page includes all images tied at its last timestamp, so it may exceed limit.",
    responses={
        503: {
            "model": ErrorBody,
            "description": "Object storage is not configured or temporarily unavailable.",
        },
    },
)
async def list_media(
    ctx: CreativeReader,
    db: DbSession,
    stubs: DemoStubsDep,
    storage: OptionalStorageDep,
    settings: SettingsDep,
    cursor: Annotated[
        AwareDatetime | None, Query(description="next_cursor from the last page.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> MediaPage:
    if stubs is not None:
        return await stubs.list_media(db, ctx, before=cursor, limit=limit)
    return await assets.list_media(
        db, ctx, require_storage(storage), settings, before=cursor, limit=limit
    )
