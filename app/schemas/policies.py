# Policy bundle and generation-eligibility contracts.

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.schemas.common import ApiModel
from app.schemas.validations import Gate, Severity, SourceReference, ValidationInputKind

MAX_RULES_PER_BUNDLE = 200


class PolicyRule(ApiModel):
    """One rule a creative is validated against. Findings cite rule_id and rule_version."""

    rule_id: str = Field(pattern=r"^[A-Z][A-Z0-9-]{1,39}$", examples=["REC-01"])
    rule_version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    requirement: str = Field(min_length=1, max_length=1000)
    gate: Gate
    severity: Severity
    mandatory: bool = Field(description="A failed mandatory rule always fails the validation.")
    applies_to: list[ValidationInputKind] = Field(min_length=1, max_length=3)
    source: SourceReference


class PolicyBundleRead(ApiModel):
    policy_bundle_id: UUID
    version: int
    name: str
    source_ref: str
    is_synthetic: bool = Field(description="True for demo rules, never an approved client policy.")
    approved_by: str
    approved_at: datetime
    rules: list[PolicyRule] = Field(max_length=MAX_RULES_PER_BUNDLE)


class PolicySummary(ApiModel):
    policy_bundle_id: UUID
    version: int
    name: str
    is_synthetic: bool
    rule_count: int


class EligibilityReason(StrEnum):
    """Why generation is or is not allowed for one creative version."""

    ELIGIBLE = "ELIGIBLE"
    STALE_VERSION = "STALE_VERSION"
    STUDY_CLOSED = "STUDY_CLOSED"
    NO_ACTIVE_POLICY = "NO_ACTIVE_POLICY"
    NOT_VALIDATED = "NOT_VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class GenerationEligibility(ApiModel):
    """Whether image generation may start. Only a brief that passed validation is eligible."""

    creative_id: UUID
    creative_version: int
    eligible: bool
    reason: EligibilityReason
    message: str
    policy_bundle_version: int | None = Field(description="Active policy the check used.")
    validation_run_id: UUID | None = Field(
        description="The passing brief validation the decision relies on, when eligible."
    )
