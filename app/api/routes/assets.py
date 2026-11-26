# Stored assets (metadata plus a short-lived download URL) and the media library.
# With DEMO_STUBS=true the demo stubs answer instead: their images live in memory, not in storage.

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from pydantic import AwareDatetime

from app.api.deps import (
    CreativeReader,
    CreativeWriter,
    DbSession,
    DemoStubsDep,
    IdempotencyKey,
    OptionalStorageDep,
    SettingsDep,
    StorageDep,
    require_storage,
)
from app.models.enums import UploadState
from app.schemas.assets import AssetRead, MediaPage
from app.schemas.common import ErrorBody
from app.schemas.uploads import UploadCreate, UploadLimits, UploadPage, UploadRead
from app.services import assets, uploads

router = APIRouter(tags=["assets"])

_STORAGE_UNAVAILABLE = {
    "model": ErrorBody,
    "description": "Object storage is not configured or temporarily unavailable.",
}
_UPLOAD_NOT_FOUND = {"model": ErrorBody, "description": "Upload not found or not permitted."}
_NOT_UPLOADER = {
    "model": ErrorBody,
    "description": "Role not allowed, or not the person who started this upload.",
}
_UPLOAD_LIMIT = {
    "model": ErrorBody,
    "description": "UPLOAD_LIMIT_REACHED (retryable): the tenant has the most uploads in progress "
    "it may have. Try again when one has finished processing.",
}

# Upload routes come before /assets/{asset_id}, so "uploads" is never parsed as an id.


@router.get(
    "/assets/upload-limits",
    summary="What uploads accept in this environment (types, size, URL lifetime)",
    description="Read these instead of hard-coding them: they are configured per environment.",
)
async def get_upload_limits(ctx: CreativeReader, settings: SettingsDep) -> UploadLimits:
    return uploads.upload_limits(settings)


@router.post(
    "/assets/uploads",
    status_code=status.HTTP_201_CREATED,
    summary="Start an upload: get a short-lived URL for one PUT of exactly the declared file",
    description="Declare the type, exact size and SHA-256. PUT the file to upload.url with "
    "upload.headers, then call POST /assets/{asset_id}/complete. The URL is secret and "
    "short-lived: never store or log it. A retry with the same Idempotency-Key and body returns "
    "the same upload (with a fresh URL while it is still pending).",
    responses={
        404: {"model": ErrorBody, "description": "Study or creative not found or not permitted."},
        409: {"model": ErrorBody, "description": "Idempotency-Key reused."},
        429: _UPLOAD_LIMIT,
        503: _STORAGE_UNAVAILABLE,
    },
)
async def create_upload(
    body: UploadCreate,
    ctx: CreativeWriter,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    storage: StorageDep,
    settings: SettingsDep,
    request: Request,
    response: Response,
) -> UploadRead:
    upload, replayed = await uploads.create_upload(
        db, ctx, body, idempotency_key, storage, settings
    )
    # The poll route's own path, so the API prefix has one source (app.main.API_PREFIX).
    response.headers["Location"] = request.app.url_path_for(
        "get_upload", asset_id=str(upload.asset_id)
    )
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return upload


@router.get(
    "/assets/uploads",
    summary="Uploads in this tenant, newest first, with their status",
    description="Shows every status, so the uploader can follow progress and see rejections. "
    "A page includes all uploads tied at its last timestamp, so it may exceed limit.",
)
async def list_uploads(
    ctx: CreativeReader,
    db: DbSession,
    status_filter: Annotated[
        list[UploadState] | None,
        Query(alias="status", description="Only these statuses; repeat to pass several."),
    ] = None,
    study_id: Annotated[UUID | None, Query(description="Only this study.")] = None,
    cursor: Annotated[
        AwareDatetime | None, Query(description="next_cursor from the last page.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> UploadPage:
    return await uploads.list_uploads(
        db, ctx, statuses=status_filter or (), study_id=study_id, before=cursor, limit=limit
    )


@router.get(
    "/assets/uploads/{asset_id}",
    summary="One upload and its status",
    description="Poll this after completing an upload until status is READY or REJECTED.",
    responses={404: _UPLOAD_NOT_FOUND},
)
async def get_upload(asset_id: UUID, ctx: CreativeReader, db: DbSession) -> UploadRead:
    return await uploads.get_upload(db, ctx, asset_id)


@router.post(
    "/assets/{asset_id}/upload-url",
    summary="A fresh upload URL for a pending upload (same file, same owner)",
    description="Use when the URL expired or was lost before the file was sent. An expired upload "
    "is admitted again like a new one, so it can meet the in-progress limit.",
    responses={
        403: _NOT_UPLOADER,
        404: _UPLOAD_NOT_FOUND,
        409: {"model": ErrorBody, "description": "The upload is no longer waiting for its file."},
        429: _UPLOAD_LIMIT,
        503: _STORAGE_UNAVAILABLE,
    },
)
async def reissue_upload_url(
    asset_id: UUID,
    ctx: CreativeWriter,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
) -> UploadRead:
    return await uploads.reissue_upload_url(db, ctx, asset_id, storage, settings)


@router.post(
    "/assets/{asset_id}/complete",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Tell the API the file was sent; it is checked and queued for inspection",
    description="Safe to repeat: returns the current status and queues inspection once. "
    "409 UPLOAD_NOT_RECEIVED (retryable) means the file has not arrived yet. If the upload URL "
    "had expired, the upload is admitted again like a new one, so it can meet the in-progress "
    "limit (429).",
    responses={
        403: _NOT_UPLOADER,
        404: _UPLOAD_NOT_FOUND,
        409: {"model": ErrorBody, "description": "The file has not been received yet."},
        429: _UPLOAD_LIMIT,
        503: _STORAGE_UNAVAILABLE,
    },
)
async def complete_upload(
    asset_id: UUID, ctx: CreativeWriter, db: DbSession, storage: StorageDep, settings: SettingsDep
) -> UploadRead:
    return await uploads.complete_upload(db, ctx, asset_id, storage, settings)


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
    summary="Generated images and checked uploads in this tenant, newest first (read-only)",
    description="Only finished images and READY uploads from studies the caller can use; "
    "uploads still in progress are in GET /assets/uploads. download_url is short-lived: list "
    "again, or GET /assets/{asset_id}, for a fresh one. A page includes all images tied at its "
    "last timestamp, so it may exceed limit.",
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
