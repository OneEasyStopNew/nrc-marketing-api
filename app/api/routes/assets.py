# Stored assets. Contract published now; implemented with generated-image storage (NRC-3).

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import CreativeReader
from app.core.errors import FeatureNotAvailableError
from app.schemas.assets import AssetRead
from app.schemas.common import ErrorBody

router = APIRouter(tags=["assets"])


@router.get(
    "/assets/{asset_id}",
    summary="Asset metadata and a short-lived download URL (D3)",
    responses={501: {"model": ErrorBody, "description": "Not implemented yet."}},
)
async def get_asset(asset_id: UUID, ctx: CreativeReader) -> AssetRead:
    raise FeatureNotAvailableError("Assets are not implemented yet.")
