# Fixed vocabularies shared by the database models and the API schemas.

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from sqlalchemy import Enum


class RecruitmentStatus(StrEnum):
    RECRUITING = "recruiting"
    CLOSED = "closed"


class Language(StrEnum):
    # Spanish is planned; unsupported languages are rejected until they are evaluated.
    EN = "en"


class Channel(StrEnum):
    FACEBOOK = "facebook"


class PostFormat(StrEnum):
    FEED_IMAGE = "feed_image"


class ExecutionState(StrEnum):
    """Execution state of a job. It says nothing about whether content passed validation."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_PROVIDER = "WAITING_PROVIDER"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class ValidationInputKind(StrEnum):
    BRIEF = "brief"
    TEXT = "text"
    IMAGE = "image"


class Outcome(StrEnum):
    """Overall result. PASS is not approval and does not make content ready to publish."""

    PASS = "PASS"  # noqa: S105  validation outcome, not a credential
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class JobStage(StrEnum):
    """Where a job is inside its execution state. Shown in the UI, so values are part of the API."""

    SUBMITTING = "submitting"
    RETRY_WAIT = "retry_wait"
    RECONCILE = "reconcile"
    WAITING_PROVIDER = "waiting_provider"
    # Provider finished but no storage is wired yet (NRC-3): the job is parked at this result.
    OUTPUT_READY = "output_ready"
    STORED = "stored"


@dataclass(frozen=True, slots=True)
class FormatSpec:
    channel: Channel
    aspect_ratio: str
    width_px: int
    height_px: int


FORMAT_SPECS: Final[Mapping[PostFormat, FormatSpec]] = MappingProxyType(
    {PostFormat.FEED_IMAGE: FormatSpec(Channel.FACEBOOK, "1:1", 1080, 1080)}
)


def enum_column[E: StrEnum](enum_cls: type[E]) -> Enum:
    # Store a StrEnum as its string value (VARCHAR), validated on write.
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )
