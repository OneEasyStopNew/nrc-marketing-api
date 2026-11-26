# Schemas for the current user and the tenant context.

import hashlib
import unicodedata
from datetime import datetime
from typing import Final, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.auth.access import Capability, Role
from app.models.enums import (
    BriefField,
    CallToAction,
    Channel,
    Language,
    PostFormat,
    RecruitmentStatus,
)
from app.schemas.common import ApiModel
from app.schemas.policies import PolicySummary


class TenantMembershipRead(ApiModel):
    tenant_id: UUID
    tenant_name: str
    roles: list[Role]
    capabilities: list[Capability]


class MeRead(ApiModel):
    sub: str
    username: str
    memberships: list[TenantMembershipRead]


# A wording's id within its study, the same across snapshots.
WORDING_KEY_PATTERN: Final = r"^[a-z0-9][a-z0-9-]{0,49}$"
TEXT_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"


def wording_sha256(text: str) -> str:
    """SHA-256 of the text's NFC form in UTF-8, lowercase hex: what a policy check pins."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode()).hexdigest()


class ApprovedWording(ApiModel):
    """One approved claim or required disclosure, with the approval it carries."""

    key: str = Field(
        pattern=WORDING_KEY_PATTERN, description="Stays the same for this wording across snapshots."
    )
    text: str
    language: Language
    version: int = Field(ge=1, description="Goes up each time the text under this key changes.")
    text_sha256: str = Field(pattern=TEXT_SHA256_PATTERN, description="See wording_sha256.")
    approved_by: str = Field(min_length=1)
    approved_at: AwareDatetime

    @model_validator(mode="after")
    def _hash_matches_text(self) -> Self:
        if self.text_sha256 != wording_sha256(self.text):
            raise ValueError(f"text_sha256 does not match the text of {self.key!r}")
        return self

    @classmethod
    def approve(
        cls,
        key: str,
        text: str,
        language: Language,
        *,
        approved_by: str,
        approved_at: datetime,
        version: int = 1,
    ) -> Self:
        return cls(
            key=key,
            text=text,
            language=language,
            version=version,
            text_sha256=wording_sha256(text),
            approved_by=approved_by,
            approved_at=approved_at,
        )


class StudyFacts(ApiModel):
    """Approved, marketing-safe facts. Validation and generation may only rely on these.

    approved_claims and required_disclosures are the plain texts; the *_records lists hold the
    same texts, in the same order, each with its approval (key, version, hash, who and when).
    """

    condition: str
    eligibility_summary: str
    min_age: int | None = Field(default=None, ge=0, le=120)
    max_age: int | None = Field(default=None, ge=0, le=120)
    sites: list[str]
    approved_claims: list[str]
    required_disclosures: list[str]
    approved_claim_records: list[ApprovedWording]
    required_disclosure_records: list[ApprovedWording]
    compensation_statement: str | None = None
    contact_url: str | None = None

    @model_validator(mode="after")
    def _records_match_texts(self) -> Self:
        for name, texts, records in (
            ("approved_claim_records", self.approved_claims, self.approved_claim_records),
            (
                "required_disclosure_records",
                self.required_disclosures,
                self.required_disclosure_records,
            ),
        ):
            if [record.text for record in records] != texts:
                raise ValueError(f"{name} must hold the same texts, in the same order")
            keys = [record.key for record in records]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name} has a key more than once")
        return self


class StudySnapshotRead(ApiModel):
    study_id: UUID
    study_title: str
    study_external_ref: str | None
    snapshot_id: UUID
    version: int
    recruitment_status: RecruitmentStatus
    language: Language
    facts: StudyFacts
    source_ref: str
    approved_by: str
    approved_at: datetime
    is_synthetic: bool = Field(description="True for development fixtures, never real studies.")


class PostFormatOption(ApiModel):
    channel: Channel
    post_format: PostFormat
    aspect_ratio: str
    width_px: int
    height_px: int
    link_fields: list[BriefField] = Field(
        description="A post with a link fills all of these, or none. Not enforced on save; "
        "missing ones show in the creative's brief_issues."
    )


class ContextRead(ApiModel):
    tenant_id: UUID
    tenant_name: str
    roles: list[Role]
    capabilities: list[Capability]
    studies: list[StudySnapshotRead]
    languages: list[Language]
    post_formats: list[PostFormatOption]
    call_to_actions: list[CallToAction]
    active_policy: PolicySummary | None = Field(description="Null until a policy is loaded.")
