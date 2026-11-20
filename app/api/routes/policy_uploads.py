# Policy source uploads (NRC-4): PDF, text or Markdown files for the tenant's policy library.
# The same flow as client media uploads, in its own bucket and for policy:manage only.

from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from app.api.deps import (
    DbSession,
    IdempotencyKey,
    PolicyManager,
    PolicySourceStorageDep,
    SettingsDep,
)
from app.schemas.common import ErrorBody
from app.schemas.uploads import PolicyUploadCreate, PolicyUploadRead
from app.services import uploads

router = APIRouter(tags=["policy"])

_STORAGE_UNAVAILABLE = {
    "model": ErrorBody,
    "description": "Policy file storage is not configured or temporarily unavailable.",
}
_NOT_FOUND = {"model": ErrorBody, "description": "Upload not found or not permitted."}
_NOT_UPLOADER = {
    "model": ErrorBody,
    "description": "Role not allowed, or not the person who started this upload.",
}


@router.post(
    "/policy-sources/uploads",
    status_code=status.HTTP_201_CREATED,
    summary="Start a policy source upload: a short-lived URL for one PUT of the declared file",
    description="Declare the type (application/pdf, text/plain or text/markdown), exact size and "
    "SHA-256. PUT the file to upload.url with upload.headers, then call "
    "POST /policy-sources/uploads/{upload_id}/complete. The URL is secret and short-lived: never "
    "store or log it. A retry with the same Idempotency-Key and body returns the same upload.",
    responses={
        409: {"model": ErrorBody, "description": "Idempotency-Key reused."},
        503: _STORAGE_UNAVAILABLE,
    },
)
async def create_policy_upload(
    body: PolicyUploadCreate,
    ctx: PolicyManager,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    storage: PolicySourceStorageDep,
    settings: SettingsDep,
    request: Request,
    response: Response,
) -> PolicyUploadRead:
    upload, replayed = await uploads.create_policy_upload(
        db, ctx, body, idempotency_key, storage, settings
    )
    response.headers["Location"] = request.app.url_path_for(
        "get_policy_upload", upload_id=str(upload.upload_id)
    )
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return upload


@router.post(
    "/policy-sources/uploads/{upload_id}/upload-url",
    summary="A fresh upload URL for a pending policy upload (same file, same owner)",
    description="Use when the URL expired or was lost before the file was sent.",
    responses={
        403: _NOT_UPLOADER,
        404: _NOT_FOUND,
        409: {"model": ErrorBody, "description": "The upload is no longer waiting for its file."},
        503: _STORAGE_UNAVAILABLE,
    },
)
async def reissue_policy_upload_url(
    upload_id: UUID,
    ctx: PolicyManager,
    db: DbSession,
    storage: PolicySourceStorageDep,
    settings: SettingsDep,
) -> PolicyUploadRead:
    return await uploads.reissue_policy_upload_url(db, ctx, upload_id, storage, settings)


@router.post(
    "/policy-sources/uploads/{upload_id}/complete",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Tell the API the file was sent; it is checked and queued for inspection",
    description="Safe to repeat: returns the current status and queues inspection once. "
    "409 UPLOAD_NOT_RECEIVED (retryable) means the file has not arrived yet.",
    responses={
        403: _NOT_UPLOADER,
        404: _NOT_FOUND,
        409: {"model": ErrorBody, "description": "The file has not been received yet."},
        503: _STORAGE_UNAVAILABLE,
    },
)
async def complete_policy_upload(
    upload_id: UUID, ctx: PolicyManager, db: DbSession, storage: PolicySourceStorageDep
) -> PolicyUploadRead:
    return await uploads.complete_policy_upload(db, ctx, upload_id, storage)


@router.get(
    "/policy-sources/uploads/{upload_id}",
    summary="One policy source upload and its status",
    description="Poll this after completing an upload until status is READY or REJECTED.",
    responses={404: _NOT_FOUND},
)
async def get_policy_upload(upload_id: UUID, ctx: PolicyManager, db: DbSession) -> PolicyUploadRead:
    return await uploads.get_policy_upload(db, ctx, upload_id)
