# Validation endpoints.
# Briefs and captions are validated for real (app.services.validations). Image validation
# answers from the demo stubs when DEMO_STUBS=true, else 501.

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Response, status

from app.api.deps import (
    CreativeReader,
    DbSession,
    DemoStubsDep,
    IdempotencyKey,
    ValidationRunner,
    require_stubs,
)
from app.core.errors import NotFoundError
from app.schemas.common import ErrorBody
from app.schemas.validations import (
    ValidationAccepted,
    ValidationCreate,
    ValidationInputKind,
    ValidationRunRead,
)
from app.services import validations

router = APIRouter()

_NOT_YET: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorBody, "description": "Not found, or not visible to the caller."},
    501: {"model": ErrorBody, "description": "Not implemented yet (DEMO_STUBS is off)."},
}
_VALIDATIONS: dict[int | str, dict[str, Any]] = {
    **_NOT_YET,
    409: {
        "model": ErrorBody,
        "description": "NO_ACTIVE_POLICY, or Idempotency-Key reused with a different body.",
    },
}


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
        service = require_stubs(stubs, "Image validation")
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
