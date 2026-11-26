# Generation job contract (implemented in D3).

from datetime import date, datetime
from decimal import Decimal
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
    model: str | None = Field(
        default=None,
        # Only exact enabled names pass (MODEL_NOT_ENABLED), so no pattern is needed here.
        min_length=1,
        max_length=100,
        description="One of GET /generation-models; omitted means that list's default.",
    )


class GenerationModel(ApiModel):
    model: str
    is_default: bool
    text_in_image: bool = Field(
        description="True: used only for drafts with text to show in the image."
    )
    aspect_ratios: list[str]
    max_brief_chars: int = Field(
        description="Longest brief this model accepts; the rest is kept for the style template."
    )
    label: str = Field(description="Name to show in the model picker.")
    estimated_usd_per_image: Decimal = Field(
        description=(
            "Provider estimate in USD for one image at 1:1 with a typical prompt, as a decimal "
            "string. An estimate, not the billed amount; each job is estimated exactly before it "
            "is submitted."
        )
    )
    price_checked_at: date = Field(description="Day the estimate was last checked.")
    price_is_fixed: bool = Field(
        description="False: token-priced, so the estimate varies with the prompt and ratio."
    )


class GenerationModelList(ApiModel):
    items: list[GenerationModel]


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
