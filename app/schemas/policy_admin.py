# Requests that build, review and move policy bundles through their lifecycle (NRC-4).

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StringConstraints

from app.models.enums import (
    CheckType,
    Gate,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    Severity,
    ValidationInputKind,
    WordingKind,
)
from app.models.policy_graph import RULE_ID_PATTERN
from app.schemas.common import RequestModel

Text = Annotated[str, StringConstraints(min_length=1, max_length=20_000, pattern=r"\S")]
Tag = Annotated[str, StringConstraints(min_length=1, max_length=64)]
Note = Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=r"\S")]
# A RULE node is created together with its rule (RuleCreate), never on its own.
NonRuleNodeType = Literal[
    PolicyNodeType.CLAUSE,
    PolicyNodeType.DISCLOSURE,
    PolicyNodeType.EXCEPTION,
    PolicyNodeType.INTERPRETATION,
    PolicyNodeType.EXAMPLE,
]


class BundleCreate(RequestModel):
    name: str = Field(min_length=1, max_length=200)
    source_ref: str = Field(min_length=1, max_length=300)
    is_synthetic: bool = Field(description="True until the bundle holds the client's real policy.")
    based_on_bundle_id: UUID | None = Field(
        default=None, description="Copy this bundle's graph into the new draft, review included."
    )


class Applicability(RequestModel):
    # Empty lists mean "no restriction", as retrieval reads them.
    input_kinds: list[ValidationInputKind] = Field(default_factory=list, max_length=3)
    channels: list[Tag] = Field(default_factory=list, max_length=20)
    languages: list[Tag] = Field(default_factory=list, max_length=10)
    audiences: list[Tag] = Field(default_factory=list, max_length=20)
    jurisdictions: list[Tag] = Field(default_factory=list, max_length=20)
    study_ids: list[UUID] = Field(default_factory=list, max_length=100)
    effective_from: AwareDatetime | None = None
    effective_to: AwareDatetime | None = None
    required_study_fact: Tag | None = None


class NodeCreate(Applicability):
    node_key: str = Field(min_length=1, max_length=80)
    node_type: NonRuleNodeType
    text: Text
    language: PolicyLanguage = PolicyLanguage.EN
    source_clause_id: UUID | None = None


class NodeUpdate(RequestModel):
    # Only the fields sent change. Any change sends the node back to PROPOSED for review.
    text: Text | None = None
    source_clause_id: UUID | None = None
    input_kinds: list[ValidationInputKind] | None = Field(default=None, max_length=3)
    channels: list[Tag] | None = Field(default=None, max_length=20)
    languages: list[Tag] | None = Field(default=None, max_length=10)
    audiences: list[Tag] | None = Field(default=None, max_length=20)
    jurisdictions: list[Tag] | None = Field(default=None, max_length=20)
    study_ids: list[UUID] | None = Field(default=None, max_length=100)
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    required_study_fact: Tag | None = None


class RuleCreate(Applicability):
    rule_id: str = Field(pattern=RULE_ID_PATTERN, examples=["REC-01"])
    rule_version: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    requirement: str = Field(min_length=1, max_length=1000)
    gate: Gate
    severity: Severity
    mandatory: bool
    check_type: CheckType
    text: Text = Field(description="The rule as it reads in the policy; searched by retrieval.")
    language: PolicyLanguage = PolicyLanguage.EN


class RuleUpdate(RequestModel):
    # Only the fields sent change. Any change sends the rule's node back to PROPOSED.
    title: str | None = Field(default=None, min_length=1, max_length=200)
    requirement: str | None = Field(default=None, min_length=1, max_length=1000)
    gate: Gate | None = None
    severity: Severity | None = None
    mandatory: bool | None = None
    check_type: CheckType | None = None


class EdgeCreate(RequestModel):
    edge_type: PolicyEdgeType
    from_node_id: UUID
    to_node_id: UUID
    provenance_clause_id: UUID | None = None
    provenance_note: Note | None = Field(
        default=None, description="Why the link holds when no source clause states it."
    )


class WordingCreate(RequestModel):
    language: PolicyLanguage = PolicyLanguage.EN
    kind: WordingKind
    text: Text = Field(description="Exact required wording, or one prohibited phrase.")
    source_clause_id: UUID | None = None
    translation_of_id: UUID | None = None


class ReviewDecision(RequestModel):
    decision: Literal["APPROVED", "REJECTED"]
    note: Note | None = None


class TransitionRequest(RequestModel):
    reason: Note | None = Field(default=None, description="Required when rejecting a bundle.")
