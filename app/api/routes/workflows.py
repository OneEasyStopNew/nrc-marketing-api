# Validation endpoints. Contract published now; implemented in D2/D4.

from typing import Any
from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import CreativeReader, IdempotencyKey, ValidationRunner
from app.core.errors import FeatureNotAvailableError
from app.schemas.common import ErrorBody
from app.schemas.validations import ValidationAccepted, ValidationCreate, ValidationRunRead

router = APIRouter()

_NOT_YET: dict[int | str, dict[str, Any]] = {
    501: {"model": ErrorBody, "description": "Not implemented yet."}
}


@router.post(
    "/validations",
    tags=["validations"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start validating a brief, caption or image (D2/D4)",
    responses=_NOT_YET,
)
async def create_validation(
    body: ValidationCreate, ctx: ValidationRunner, idempotency_key: IdempotencyKey
) -> ValidationAccepted:
    raise FeatureNotAvailableError("Validation is not implemented yet.")


@router.get(
    "/validations/{validation_run_id}",
    tags=["validations"],
    summary="Validation outcome, gates, findings and coverage gaps (D2/D4)",
    responses=_NOT_YET,
)
async def get_validation(validation_run_id: UUID, ctx: CreativeReader) -> ValidationRunRead:
    raise FeatureNotAvailableError("Validation is not implemented yet.")
