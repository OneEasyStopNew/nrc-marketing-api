# Generation job contract (implemented in D3).

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.models.enums import ExecutionState, JobStage
from app.schemas.common import ApiModel, RequestModel


class JobKind(StrEnum):
    IMAGE_GENERATION = "image_generation"


class CreativeJobCreate(RequestModel):
    creative_id: UUID
    creative_version: int = Field(ge=1)
    kind: JobKind


class JobAccepted(ApiModel):
    job_id: UUID
    state: ExecutionState


class JobRead(ApiModel):
    job_id: UUID
    kind: JobKind
    creative_id: UUID
    creative_version: int
    state: ExecutionState
    stage: JobStage | None
    provider_request_id: str | None
    failure_code: str | None
    failure_message: str | None
    result_asset_id: UUID | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
