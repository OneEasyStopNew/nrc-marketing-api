# Creative draft schemas.

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from app.models.enums import FORMAT_SPECS, BriefField, CallToAction, Channel, Language, PostFormat
from app.schemas.common import (
    ApiModel,
    AudienceText,
    CaptionText,
    DestinationUrl,
    HeadlineText,
    ImageText,
    PromptText,
    RequestModel,
    SiteName,
    VisualInstructionsText,
)
from app.schemas.validations import ValidationRunSummary

_SITE_HELP = "One of the study snapshot's sites, exactly; leave out when not tied to one site."
_LINK_HELP = "https only."


class CreativeCreate(RequestModel):
    study_id: UUID
    prompt: PromptText
    caption: CaptionText | None = None
    image_text: ImageText | None = Field(
        default=None, description="Leave out for an image without text."
    )
    language: Language
    channel: Channel
    post_format: PostFormat
    headline: HeadlineText | None = None
    call_to_action: CallToAction | None = None
    destination_url: DestinationUrl | None = Field(default=None, description=_LINK_HELP)
    site: SiteName | None = Field(default=None, description=_SITE_HELP)
    audience: AudienceText | None = None
    visual_instructions: VisualInstructionsText | None = None

    @model_validator(mode="after")
    def _format_matches_channel(self) -> Self:
        if FORMAT_SPECS[self.post_format].channel is not self.channel:
            raise ValueError(
                f"post_format {self.post_format.value!r} is not offered on {self.channel.value!r}"
            )
        return self


class CreativeUpdate(RequestModel):
    """Fields to change. Omitted fields keep their value; send any field but current_prompt as
    null to clear it."""

    current_prompt: PromptText | None = None
    caption: CaptionText | None = None
    image_text: ImageText | None = None
    headline: HeadlineText | None = None
    call_to_action: CallToAction | None = None
    destination_url: DestinationUrl | None = Field(default=None, description=_LINK_HELP)
    site: SiteName | None = Field(default=None, description=_SITE_HELP)
    audience: AudienceText | None = None
    visual_instructions: VisualInstructionsText | None = None

    @model_validator(mode="after")
    def _has_changes(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("send at least one field to change")
        if "current_prompt" in self.model_fields_set and self.current_prompt is None:
            raise ValueError("current_prompt cannot be null")
        return self


class BriefIssueCode(StrEnum):
    # A post with a link must fill every field in its format's link_fields.
    REQUIRED_WITH_LINK = "REQUIRED_WITH_LINK"


class BriefIssue(ApiModel):
    field: BriefField
    code: BriefIssueCode


class _BriefFields(ApiModel):
    headline: str | None = Field(description="Link headline under the image, not on it.")
    call_to_action: CallToAction | None
    destination_url: str | None
    site: str | None
    audience: str | None
    visual_instructions: str | None = Field(
        description="Stored with the brief; image generation uses the prompt and image text only."
    )


class CreativeRead(_BriefFields):
    id: UUID
    study_id: UUID
    version: int = Field(description='Send as If-Match: "<version>" when editing.')
    original_prompt: str = Field(description="The first prompt, kept unchanged across versions.")
    current_prompt: str
    caption: str | None
    image_text: str | None = Field(description="Null when the image has no text.")
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
    brief_issues: list[BriefIssue] = Field(
        default_factory=list,
        description="What the current version still needs before it can be approved. Saving "
        "works regardless; empty when the brief is complete.",
    )


class CreativeVersionRead(_BriefFields):
    version: int
    current_prompt: str
    caption: str | None
    image_text: str | None
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
