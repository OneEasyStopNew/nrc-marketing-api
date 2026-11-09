# Generation jobs. Contract published now; implemented in D3.

from typing import Any
from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import CreativeReader, GenerationRunner, IdempotencyKey
from app.core.errors import FeatureNotAvailableError
from app.schemas.common import ErrorBody
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobRead

router = APIRouter(tags=["jobs"])

_NOT_YET: dict[int | str, dict[str, Any]] = {
    501: {"model": ErrorBody, "description": "Not implemented yet."}
}


@router.post(
    "/creative-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start image generation for a validated brief (D3)",
    responses=_NOT_YET,
)
async def create_creative_job(
    body: CreativeJobCreate, ctx: GenerationRunner, idempotency_key: IdempotencyKey
) -> JobAccepted:
    raise FeatureNotAvailableError("Image generation is not implemented yet.")


@router.get(
    "/jobs/{job_id}",
    summary="Persisted job state; polling never starts another generation (D3)",
    responses=_NOT_YET,
)
async def get_job(job_id: UUID, ctx: CreativeReader) -> JobRead:
    raise FeatureNotAvailableError("Image generation is not implemented yet.")
