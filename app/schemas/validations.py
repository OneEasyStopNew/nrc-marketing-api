# Validation contract (implemented in D2 for text and D4 for images).

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from app.models.enums import ExecutionState
from app.schemas.common import ApiModel, RequestModel


class ValidationInputKind(StrEnum):
    BRIEF = "brief"
    TEXT = "text"
    IMAGE = "image"


class Outcome(StrEnum):
    """Overall result. PASS is not approval and does not make content ready to publish."""

    PASS = "PASS"  # noqa: S105  validation outcome, not a credential
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class RuleResultValue(StrEnum):
    PASS = "PASS"  # noqa: S105  validation outcome, not a credential
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNABLE_TO_ASSESS = "UNABLE_TO_ASSESS"


class Gate(StrEnum):
    COMPLETENESS = "completeness"
    STUDY_FACTS = "study_facts"
    POLICY = "policy"
    MEDIA = "media"


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ValidationCreate(RequestModel):
    creative_id: UUID
    creative_version: int = Field(ge=1)
    input_kind: ValidationInputKind
    asset_id: UUID | None = Field(default=None, description="Required when input_kind is image.")

    @model_validator(mode="after")
    def _asset_matches_kind(self) -> Self:
        if (self.input_kind is ValidationInputKind.IMAGE) != (self.asset_id is not None):
            raise ValueError("asset_id is required for image input and not allowed otherwise")
        return self


class ValidationAccepted(ApiModel):
    job_id: UUID
    validation_run_id: UUID
    state: ExecutionState


class TextSpan(ApiModel):
    """Zero-based Unicode code-point offsets into the validated text; end is exclusive."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class ImageRegion(ApiModel):
    """Fractions (0..1) of the stored image's original width and height, top-left origin."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class SourceReference(ApiModel):
    source_id: str
    clause: str
    excerpt: str | None = None


class Finding(ApiModel):
    rule_id: str
    rule_version: int
    severity: Severity
    explanation: str
    source: SourceReference
    text_span: TextSpan | None = None
    image_region: ImageRegion | None = None
    offending_excerpt: str | None = None
    suggested_correction: str | None = None
    needs_human_decision: bool


class RuleResult(ApiModel):
    rule_id: str
    rule_version: int
    mandatory: bool
    result: RuleResultValue
    reason: str | None = None


class GateResult(ApiModel):
    gate: Gate
    outcome: Outcome


class CoverageGap(ApiModel):
    rule_id: str | None = None
    reason: str


class ValidationRunSummary(ApiModel):
    validation_run_id: UUID
    input_kind: ValidationInputKind
    creative_version: int
    state: ExecutionState
    outcome: Outcome | None = Field(description="Null until the run finishes.")
    is_current: bool = Field(description="False once the creative or its inputs have changed.")
    evaluated_at: datetime | None


class ValidationRunRead(ValidationRunSummary):
    job_id: UUID
    input_hash: str
    asset_id: UUID | None
    study_snapshot_version: int
    policy_bundle_version: str | None
    gates: list[GateResult]
    rule_results: list[RuleResult]
    findings: list[Finding]
    coverage_gaps: list[CoverageGap]
