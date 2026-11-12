# Generation jobs: accept a durable job, then poll its persisted state.
# With DEMO_STUBS=true the demo stubs answer instead, so the stubbed assets, image checks and
# export (which only know stub jobs) keep working until storage lands (NRC-3).

from uuid import UUID

from fastapi import APIRouter, Response, status

from app.api.deps import (
    CreativeReader,
    DbSession,
    DemoStubsDep,
    GenerationRunner,
    IdempotencyKey,
    SettingsDep,
)
from app.schemas.common import ErrorBody
from app.schemas.jobs import CreativeJobCreate, JobAccepted, JobRead
from app.services import jobs

router = APIRouter(tags=["jobs"])


@router.post(
    "/creative-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start image generation for a validated brief (D3)",
    description="Returns at once; poll the Location URL. A retry with the same Idempotency-Key "
    "and body returns the original job with Idempotent-Replayed: true.",
    responses={
        404: {"model": ErrorBody, "description": "Creative not found or not permitted."},
        409: {
            "model": ErrorBody,
            "description": "GENERATION_NOT_ELIGIBLE (details.reason), VALIDATION_NOT_PASSED "
            "(details.validation_run_id and status), or Idempotency-Key reused.",
        },
        412: {"model": ErrorBody, "description": "Not the current creative version."},
        429: {
            "model": ErrorBody,
            "description": "GENERATION_LIMIT_REACHED: active jobs or daily credits (retryable).",
        },
    },
)
async def create_creative_job(
    body: CreativeJobCreate,
    ctx: GenerationRunner,
    db: DbSession,
    settings: SettingsDep,
    idempotency_key: IdempotencyKey,
    stubs: DemoStubsDep,
    response: Response,
) -> JobAccepted:
    if stubs is not None:
        accepted, replayed = await stubs.start_generation(db, ctx, body, idempotency_key)
    else:
        accepted, replayed = await jobs.accept_creative_job(
            db, ctx, body, idempotency_key, settings
        )
    response.headers["Location"] = f"/v1/marketing/jobs/{accepted.job_id}"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return accepted


@router.get(
    "/jobs/{job_id}",
    summary="Persisted job state; polling never starts another generation (D3)",
    responses={404: {"model": ErrorBody}},
)
async def get_job(job_id: UUID, ctx: CreativeReader, db: DbSession, stubs: DemoStubsDep) -> JobRead:
    if stubs is not None:
        return await stubs.get_job(db, ctx, job_id)
    return await jobs.get_job(db, ctx, job_id)
