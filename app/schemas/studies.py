# Studies, their approved snapshots, and what drafts may use.

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.models.enums import Language
from app.schemas.common import ApiModel
from app.schemas.tenancy import PostFormatOption, StudySnapshotRead


class StudyAvailability(StrEnum):
    # Closed is a decision about the study; no_approved_snapshot means its facts are not in yet.
    RECRUITING = "recruiting"
    CLOSED = "closed"
    NO_APPROVED_SNAPSHOT = "no_approved_snapshot"


class StudyRead(ApiModel):
    study_id: UUID
    title: str
    external_ref: str | None
    availability: StudyAvailability = Field(
        description="Only recruiting studies accept new drafts. Existing drafts of closed "
        "studies stay readable."
    )
    latest_snapshot: StudySnapshotRead | None = Field(
        description="Null until an approved snapshot is imported."
    )


class StudyPage(ApiModel):
    items: list[StudyRead]
    next_cursor: UUID | None = Field(description="Pass as cursor for the next page.")


class StudySnapshotPage(ApiModel):
    items: list[StudySnapshotRead]
    next_cursor: int | None = Field(description="Pass as cursor for the next (older) page.")


class TextLimits(ApiModel):
    prompt_max_chars: int
    caption_max_chars: int
    image_text_max_chars: int
    image_text_max_lines: int
    generation_text_max_chars: int = Field(
        description="Prompt plus image text together. Checked when generation starts, not on "
        "save, because drafts that are only validated never generate."
    )


class PlatformCapabilities(ApiModel):
    languages: list[Language]
    post_formats: list[PostFormatOption]
    text_limits: TextLimits
