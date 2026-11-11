# Validation, generation and asset endpoints. Contract published now; implemented in D2-D4.
# With DEMO_STUBS=true they answer with simulated data (app.services.demo_stubs); otherwise 501.

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Response, status

from app.api.deps import (
    CreativeReader,
    DbSession,
    DemoStubsDep,
    GenerationRunner,
    IdempotencyKey,
    ValidationRunner,
)
from app.core.errors import FeatureNotAvailableError
from app.schemas.assets import AssetRead
from app.schemas.common import ErrorBody
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobRead
from app.schemas.validations import ValidationAccepted, ValidationCreate, ValidationRunRead
from app.services.demo_stubs import DemoStubs

router = APIRouter()

_NOT_YET: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorBody, "description": "Not found, or not visible to the caller."},
    501: {"model": ErrorBody, "description": "Not implemented yet (DEMO_STUBS is off)."},
}
_COMMANDS: dict[int | str, dict[str, Any]] = {
    **_NOT_YET,
    409: {"model": ErrorBody, "description": "Idempotency-Key reused with a different body."},
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
    summary="Start validating a brief, caption or image (D2/D4)",
    description="Poll GET /validations/{validation_run_id} until state is SUCCEEDED. "
    "A retry with the same Idempotency-Key and body returns the original run.",
    responses=_COMMANDS,
)
async def create_validation(
    body: ValidationCreate,
    ctx: ValidationRunner,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    stubs: DemoStubsDep,
    response: Response,
) -> ValidationAccepted:
    service = _stubs(stubs, "Validation")
    accepted, replayed = await service.start_validation(db, ctx, body, idempotency_key)
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return accepted


@router.get(
    "/validations/{validation_run_id}",
    tags=["validations"],
    summary="Validation outcome, gates, findings and coverage gaps (D2/D4)",
    responses=_NOT_YET,
)
async def get_validation(
    validation_run_id: UUID, ctx: CreativeReader, db: DbSession, stubs: DemoStubsDep
) -> ValidationRunRead:
    return await _stubs(stubs, "Validation").get_validation(db, ctx, validation_run_id)


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
