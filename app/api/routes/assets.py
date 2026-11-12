# Stored assets: metadata plus a short-lived download URL.

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import CreativeReader, DbSession, SettingsDep, StorageDep
from app.schemas.assets import AssetRead
from app.schemas.common import ErrorBody
from app.services import assets

router = APIRouter(tags=["assets"])


@router.get(
    "/assets/{asset_id}",
    summary="Asset metadata and a short-lived download URL (D3)",
    description="The URL is not stored or logged; ask again when it expires.",
    responses={
        404: {"model": ErrorBody, "description": "Asset not found or not permitted."},
        503: {"model": ErrorBody, "description": "Object storage is not configured."},
    },
)
async def get_asset(
    asset_id: UUID, ctx: CreativeReader, db: DbSession, storage: StorageDep, settings: SettingsDep
) -> AssetRead:
    return await assets.get_asset(db, ctx, asset_id, storage, settings)
