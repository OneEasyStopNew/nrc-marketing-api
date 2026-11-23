# Validation endpoints.
# Briefs and captions are validated for real (app.services.validations). Image validation
# answers from the demo stubs when DEMO_STUBS=true, else 501.

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Response, status

from app.api.deps import (
    ContextProviderDep,
    ContextTimeoutDep,
    CreativeReader,
    DbSession,
    DemoStubsDep,
    IdempotencyKey,
    SettingsDep,
    ValidationRunner,
    require_stubs,
)
from app.core.errors import NotFoundError
from app.schemas.common import ErrorBody
from app.schemas.policy_evidence import ValidationContextRead
from app.schemas.validations import (
    ValidationAccepted,
    ValidationCreate,
    ValidationInputKind,
    ValidationRunRead,
)
from app.services import policy_evidence, validations

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
    description="Briefs and captions are checked against the active policy and the study facts. "
    "Without the model check (VALIDATION_MODEL_MODE=off) the run finishes before this returns "
    "(state SUCCEEDED; semantic rules stay UNABLE_TO_ASSESS). With it, this queues a job "
    "(state QUEUED): poll GET /validations/{validation_run_id} until SUCCEEDED (or FAILED). A "
    "finished PASS or FAIL with identical content, configuration and versions is reused "
    "(state SUCCEEDED). job_id equals validation_run_id. Images need DEMO_STUBS (501 "
    "otherwise); poll those until SUCCEEDED. A retry with the same Idempotency-Key and body "
    "returns the original run. Each "
    "brief or caption run also stores the policy context it was checked against (see "
    "GET /validations/{validation_run_id}/context). If that context can't be gathered (each "
    "of its database queries has a time limit), the run is stored as NEEDS_REVIEW (or FAIL), "
    "never PASS, with a coverage gap.",
    responses=_VALIDATIONS,
)
async def create_validation(
    body: ValidationCreate,
    ctx: ValidationRunner,
    db: DbSession,
    idempotency_key: IdempotencyKey,
    stubs: DemoStubsDep,
    context_provider: ContextProviderDep,
    context_timeout: ContextTimeoutDep,
    settings: SettingsDep,
    response: Response,
) -> ValidationAccepted:
    if body.input_kind is ValidationInputKind.IMAGE:
        service = require_stubs(stubs, "Image validation")
        accepted, replayed = await service.start_validation(db, ctx, body, idempotency_key)
    else:
        accepted, replayed = await validations.create_validation(
            db, ctx, body, idempotency_key, context_provider, context_timeout, settings
        )
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


@router.get(
    "/validations/{validation_run_id}/context",
    tags=["validations"],
    summary="Policy rules and evidence the run's text was checked against",
    description="The stored policy context of a brief or caption run: the rules selected by "
    "their conditions, their source clauses, disclosures and exceptions, search matches and "
    "anything retrieval couldn't settle (missing_context, never a pass). Evidence only: the "
    "outcome still comes from the rule checks. A run without stored context (older runs, or "
    "context unavailable, recorded as a coverage gap) answers 404 CONTEXT_NOT_AVAILABLE.",
    responses={
        404: {
            "model": ErrorBody,
            "description": "NOT_FOUND (no such run for the caller), or CONTEXT_NOT_AVAILABLE "
            "(the run stored no policy context).",
        }
    },
)
async def get_validation_context(
    validation_run_id: UUID, ctx: CreativeReader, db: DbSession
) -> ValidationContextRead:
    return await policy_evidence.get_validation_context(db, ctx, validation_run_id)
