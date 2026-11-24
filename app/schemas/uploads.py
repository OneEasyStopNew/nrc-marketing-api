# Client upload contract (NRC-3): limits, the upload request, its status and media authorization.

import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints

from app.models.enums import Channel, IdentifiablePerson, UploadPurpose, UploadState
from app.schemas.common import ApiModel, RequestModel

MAX_UPLOAD_NAME_CHARS = 255
MAX_AUTHORIZATION_REF_CHARS = 200
# A record id such as "RELEASE-2026-0142": no spaces or control characters, so a blank or
# whitespace value can't read as authorization on record.
AUTHORIZATION_REF_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"


def _display_name(value: str) -> str:
    # Shown in the library only; never used in object keys, tags or logs.
    value = unicodedata.normalize("NFC", value).strip()
    if not value:
        raise ValueError("must not be blank")
    # Checked again after normalizing: NFC can lengthen a string past the column size.
    if len(value) > MAX_UPLOAD_NAME_CHARS:
        raise ValueError(f"must be at most {MAX_UPLOAD_NAME_CHARS} characters")
    if any(unicodedata.category(char) in ("Cc", "Cf") for char in value):
        raise ValueError("must not contain control characters")
    return value


UploadName = Annotated[
    str, Field(min_length=1, max_length=MAX_UPLOAD_NAME_CHARS), AfterValidator(_display_name)
]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", examples=["9f86d0" + "0" * 58])]


class UploadLimits(ApiModel):
    """What POST /assets/uploads accepts. Configured per environment; read it, don't hard-code."""

    image_types: list[str] = Field(examples=[["image/png", "image/jpeg"]])
    image_max_bytes: int = Field(examples=[20_971_520])
    upload_url_ttl_seconds: int = Field(examples=[600])


class AuthorizationStatus(StrEnum):
    """Whether media authorization is on record. None of these means it was verified."""

    # The uploader declared that no identifiable person appears. A declaration, not a check.
    NO_PERSON_DECLARED = "NO_PERSON_DECLARED"
    # An identifiable person appears (or it is unknown) and no authorization reference is recorded.
    MISSING = "MISSING"
    RECORDED = "RECORDED"
    EXPIRED = "EXPIRED"


class MediaAuthorizationInput(RequestModel):
    identifiable_person: IdentifiablePerson = Field(
        description="Whether a person can be identified in the file. A declaration, not consent."
    )
    usage_authorization_ref: (
        Annotated[
            str,
            # Strips surrounding spaces (a typing slip) before the length and pattern checks.
            StringConstraints(
                strip_whitespace=True,
                min_length=1,
                max_length=MAX_AUTHORIZATION_REF_CHARS,
                pattern=AUTHORIZATION_REF_PATTERN,
            ),
        ]
        | None
    ) = Field(
        default=None,
        description="Reference to the signed release or licence record (not the document).",
        examples=["RELEASE-2026-0142"],
    )
    permitted_channels: list[Channel] = Field(
        default_factory=list,
        max_length=len(Channel),
        description="Channels the authorization covers; empty means none recorded.",
    )
    expires_at: AwareDatetime | None = Field(
        default=None, description="When the authorization ends; null when it does not expire."
    )


class MediaAuthorization(ApiModel):
    """Read by the approval and publication gates (NRC-12, NRC-14).

    Missing or expired authorization for a file with an identifiable person blocks them.
    """

    identifiable_person: IdentifiablePerson
    usage_authorization_ref: str | None
    permitted_channels: list[Channel]
    expires_at: datetime | None
    status: AuthorizationStatus


class UploadCreate(RequestModel):
    study_id: UUID = Field(description="The study the file is for; it decides who can see it.")
    creative_id: UUID | None = Field(
        default=None, description="Optional draft in the same study the file is meant for."
    )
    name: UploadName = Field(description="File name to show in the library.")
    mime_type: str = Field(
        max_length=64, description="One of GET /assets/upload-limits image_types."
    )
    size_bytes: int = Field(ge=1, description="Exact size of the file in bytes.")
    checksum_sha256: Sha256Hex = Field(description="SHA-256 of the file, lower-case hex.")
    authorization: MediaAuthorizationInput


class UploadTarget(ApiModel):
    """Where and how to send the file: one PUT of the exact bytes declared, with these headers."""

    url: str = Field(description="Short-lived and secret: do not store, log or share it.")
    method: Literal["PUT"]
    headers: dict[str, str] = Field(
        description="Send exactly these headers with the PUT, or storage refuses it."
    )
    expires_at: datetime


class UploadReason(ApiModel):
    """Why an upload was rejected, or the last temporary problem while it waits."""

    code: str = Field(examples=["MEDIA_UNSUPPORTED_TYPE"])
    message: str
    retryable: bool = Field(
        description="True: the system retries by itself (or upload again). False: final."
    )


class UploadScan(ApiModel):
    engine: str
    simulated: bool = Field(
        description="True only in local and demo environments: no real malware scan ran."
    )
    scanned_at: datetime


class UploadRead(ApiModel):
    """A client upload. Once READY, asset_id also works with GET /assets/{asset_id}."""

    asset_id: UUID
    study_id: UUID
    creative_id: UUID | None
    purpose: UploadPurpose
    status: UploadState
    reason: UploadReason | None
    name: str
    mime_type: str = Field(description="As declared; inspection checks the real type.")
    size_bytes: int
    checksum_sha256: str
    width_px: int | None = Field(description="Found by inspection; null until then.")
    height_px: int | None
    scan: UploadScan | None
    authorization: MediaAuthorization
    created_by: str
    created_at: datetime
    completed_at: datetime | None
    upload: UploadTarget | None = Field(
        default=None,
        description="Only in the create and upload-url responses while the file is awaited.",
    )


class PolicyUploadCreate(RequestModel):
    """A policy source file (PDF, plain text or Markdown) for the tenant's policy library."""

    name: UploadName = Field(description="File name to show in the policy library.")
    mime_type: str = Field(
        max_length=64, description="application/pdf, text/plain or text/markdown."
    )
    size_bytes: int = Field(ge=1, description="Exact size of the file in bytes.")
    checksum_sha256: Sha256Hex = Field(description="SHA-256 of the file, lower-case hex.")


class PolicyUploadRead(ApiModel):
    """A policy source upload. READY means it passed the checks; it is never an asset."""

    upload_id: UUID
    purpose: UploadPurpose
    status: UploadState
    reason: UploadReason | None
    name: str
    mime_type: str = Field(description="As declared; inspection checks the real type.")
    size_bytes: int
    checksum_sha256: str
    scan: UploadScan | None
    created_by: str
    created_at: datetime
    completed_at: datetime | None
    upload: UploadTarget | None = Field(
        default=None,
        description="Only in the create and upload-url responses while the file is awaited.",
    )


class UploadPage(ApiModel):
    items: list[UploadRead]
    next_cursor: datetime | None = Field(description="Pass as cursor for the next (older) page.")
