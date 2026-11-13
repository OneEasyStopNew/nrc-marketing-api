# Schemas for the current user and the tenant context.

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.auth.access import Capability, Role
from app.models.enums import Channel, Language, PostFormat, RecruitmentStatus
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


class StudyFacts(ApiModel):
    """Approved, marketing-safe facts. Validation and generation may only rely on these."""

    condition: str
    eligibility_summary: str
    min_age: int | None = Field(default=None, ge=0, le=120)
    max_age: int | None = Field(default=None, ge=0, le=120)
    sites: list[str]
    approved_claims: list[str]
    required_disclosures: list[str]
    compensation_statement: str | None = None
    contact_url: str | None = None


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


class ContextRead(ApiModel):
    tenant_id: UUID
    tenant_name: str
    roles: list[Role]
    capabilities: list[Capability]
    studies: list[StudySnapshotRead]
    languages: list[Language]
    post_formats: list[PostFormatOption]
    active_policy: PolicySummary | None = Field(description="Null until a policy is loaded.")
