# Validation, generation, asset and media endpoints.
# Briefs and captions are validated for real (app.services.validations). Image validation,
# generation, assets, media and export answer from the demo stubs when DEMO_STUBS=true, else 501.

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Response, status
from pydantic import AwareDatetime

from app.api.deps import (
    CreativeReader,
    DbSession,
    DemoStubsDep,
    GenerationRunner,
    IdempotencyKey,
    ValidationRunner,
)
from app.core.errors import FeatureNotAvailableError, NotFoundError
from app.schemas.assets import AssetRead, MediaPage
from app.schemas.common import ErrorBody
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobRead
from app.schemas.validations import (
    ValidationAccepted,
    ValidationCreate,
    ValidationInputKind,
    ValidationRunRead,
)
from app.services import validations
from app.services.demo_stubs import DemoStubs

router = APIRouter()

_NOT_BUILT: dict[int | str, dict[str, Any]] = {
    501: {"model": ErrorBody, "description": "Not implemented yet (DEMO_STUBS is off)."},
}
_NOT_YET: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorBody, "description": "Not found, or not visible to the caller."},
    **_NOT_BUILT,
}
_VALIDATIONS: dict[int | str, dict[str, Any]] = {
    **_NOT_YET,
    409: {
        "model": ErrorBody,
        "description": "NO_ACTIVE_POLICY, or Idempotency-Key reused with a different body.",
    },
}
_GENERATION: dict[int | str, dict[str, Any]] = {
    **_NOT_YET,
    409: {
        "model": ErrorBody,
        "description": "GENERATION_NOT_ELIGIBLE (details.reason says why), or "
        "IDEMPOTENCY_KEY_REUSED.",
    },
}


def _stubs(stubs: DemoStubs | None, feature: str) -> DemoStubs:
    if stubs is None:
        raise FeatureNotAvailableError(f"{feature} is not implemented yet.")
    return stubs


@router.post(
    "/validations",
    tags=["validations"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Validate a brief, caption or image",
    description="Briefs and captions are checked against the active policy and the study facts "
    "before this returns, so state is already SUCCEEDED; job_id equals validation_run_id until "
    "validation runs as a job. Images need DEMO_STUBS (501 otherwise); poll those until "
    "SUCCEEDED. A retry with the same Idempotency-Key and body returns the original run.",
    responses=_VALIDATIONS,
)
async def create_validation(
    body: ValidationCreate,
    ctx: ValidationRunner,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    stubs: DemoStubsDep,
    response: Response,
) -> ValidationAccepted:
    if body.input_kind is ValidationInputKind.IMAGE:
        service = _stubs(stubs, "Image validation")
        accepted, replayed = await service.start_validation(db, ctx, body, idempotency_key)
    else:
        accepted, replayed = await validations.create_validation(db, ctx, body, idempotency_key)
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return accepted


@router.get(
    "/validations/{validation_run_id}",
    tags=["validations"],
    summary="Validation outcome, gates, findings and coverage gaps",
    responses=_NOT_YET,
)
async def get_validation(
    validation_run_id: UUID, ctx: CreativeReader, db: DbSession, stubs: DemoStubsDep
) -> ValidationRunRead:
    try:
        return await validations.get_validation(db, ctx, validation_run_id)
    except NotFoundError:
        # Image runs exist only in the demo stubs' memory.
        if stubs is None:
            raise
    return await stubs.get_validation(db, ctx, validation_run_id)


@router.post(
    "/creative-jobs",
    tags=["jobs"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start image generation for a validated brief (D3)",
    description="Only a version whose brief passed validation can generate. Poll "
    "GET /jobs/{job_id}; when it SUCCEEDED, result_asset_id is the image.",
    responses=_GENERATION,
)
async def create_creative_job(
    body: CreativeJobCreate,
    ctx: GenerationRunner,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    stubs: DemoStubsDep,
    response: Response,
) -> JobAccepted:
    service = _stubs(stubs, "Image generation")
    accepted, replayed = await service.start_generation(db, ctx, body, idempotency_key)
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return accepted


@router.get(
    "/jobs/{job_id}",
    tags=["jobs"],
    summary="Persisted job state; polling never starts another generation (D3)",
    responses=_NOT_YET,
)
async def get_job(job_id: UUID, ctx: CreativeReader, db: DbSession, stubs: DemoStubsDep) -> JobRead:
    return await _stubs(stubs, "Image generation").get_job(db, ctx, job_id)


@router.get(
    "/assets/{asset_id}",
    tags=["assets"],
    summary="Asset metadata and a short-lived download URL (D3)",
    responses=_NOT_YET,
)
async def get_asset(
    asset_id: UUID, ctx: CreativeReader, db: DbSession, stubs: DemoStubsDep
) -> AssetRead:
    return await _stubs(stubs, "Assets").get_asset(db, ctx, asset_id)


@router.get(
    "/media",
    tags=["assets"],
    summary="Generated images in this tenant, newest first (read-only)",
    description="Only finished images from studies the caller can use. Uploads are not "
    "supported yet. download_url is short-lived: list again, or GET /assets/{asset_id}, for a "
    "fresh one.",
    responses=_NOT_BUILT,
)
async def list_media(
    ctx: CreativeReader,
    db: DbSession,
    stubs: DemoStubsDep,
    cursor: Annotated[
        AwareDatetime | None, Query(description="next_cursor from the last page.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> MediaPage:
    return await _stubs(stubs, "The media library").list_media(db, ctx, before=cursor, limit=limit)
