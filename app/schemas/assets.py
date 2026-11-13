# Asset and draft-export contract (implemented in D3 and D5).

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.schemas.common import ApiModel
from app.schemas.validations import ValidationRunSummary


class AssetOrigin(StrEnum):
    GENERATED = "generated"
    UPLOADED = "uploaded"


class AssetRead(ApiModel):
    """Immutable stored bytes: any change produces a new asset_id and needs new validation."""

    asset_id: UUID
    creative_id: UUID
    origin: AssetOrigin
    mime_type: str
    width_px: int
    height_px: int
    size_bytes: int
    checksum_sha256: str
    provider: str | None
    model: str | None
    created_at: datetime
    download_url: str | None = Field(description="Short-lived; do not store or log it.")
    download_url_expires_at: datetime | None


class MediaItem(ApiModel):
    """A finished generated image the caller can see. Uploads are not supported yet."""

    asset_id: UUID
    creative_id: UUID
    creative_version: int
    origin: AssetOrigin
    mime_type: str
    width_px: int
    height_px: int
    size_bytes: int
    created_at: datetime
    created_by: str
    created_by_name: str | None = Field(
        description="Display name of whoever started the generation; null when none is set."
    )
    download_url: str | None = Field(description="Short-lived; do not store or log it.")
    download_url_expires_at: datetime | None


class MediaPage(ApiModel):
    items: list[MediaItem]
    next_cursor: datetime | None = Field(description="Pass as cursor for the next (older) page.")


class ExportManifest(ApiModel):
    """A review draft. It carries no approval and causes no publication."""

    status: Literal["DRAFT"]
    creative_id: UUID
    creative_version: int
    caption: str | None
    image: AssetRead
    study_snapshot_version: int
    policy_bundle_version: int | None
    latest_validation: ValidationRunSummary | None
    exported_at: datetime
