# Creative draft schemas.

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from app.models.enums import FORMAT_SPECS, Channel, Language, PostFormat
from app.schemas.common import ApiModel, CaptionText, PromptText, RequestModel
from app.schemas.validations import ValidationRunSummary


class CreativeCreate(RequestModel):
    study_id: UUID
    prompt: PromptText
    caption: CaptionText | None = None
    language: Language
    channel: Channel
    post_format: PostFormat

    @model_validator(mode="after")
    def _format_matches_channel(self) -> Self:
        if FORMAT_SPECS[self.post_format].channel is not self.channel:
            raise ValueError(
                f"post_format {self.post_format.value!r} is not offered on {self.channel.value!r}"
            )
        return self


class CreativeUpdate(RequestModel):
    """Fields to change. Omitted fields keep their value; send caption: null to clear it."""

    current_prompt: PromptText | None = None
    caption: CaptionText | None = None

    @model_validator(mode="after")
    def _has_changes(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("send at least one field to change")
        if "current_prompt" in self.model_fields_set and self.current_prompt is None:
            raise ValueError("current_prompt cannot be null")
        return self


class CreativeRead(ApiModel):
    id: UUID
    study_id: UUID
    version: int = Field(description='Send as If-Match: "<version>" when editing.')
    original_prompt: str = Field(description="The first prompt, kept unchanged across versions.")
    current_prompt: str
    caption: str | None
    language: Language
    channel: Channel
    post_format: PostFormat
    study_snapshot_id: UUID
    study_snapshot_version: int
    asset_id: UUID | None = Field(
        default=None,
        description="Newest generated image for the current version; null until one is stored "
        "and again after an edit creates a new version.",
    )
    created_by: str
    created_at: datetime
    updated_at: datetime
    validation_runs: list[ValidationRunSummary] = Field(default_factory=list)


class CreativeVersionRead(ApiModel):
    version: int
    current_prompt: str
    caption: str | None
    study_snapshot_id: UUID
    language: Language
    channel: Channel
    post_format: PostFormat
    created_by: str
    created_by_name: str | None = Field(
        description="The author's display name in this tenant; null when none is set."
    )
    created_at: datetime


class CreativeVersionPage(ApiModel):
    items: list[CreativeVersionRead]
    next_cursor: int | None = Field(description="Pass as cursor for the next (older) page.")
