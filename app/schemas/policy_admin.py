# Requests and responses for policy sources and bundles: build, review and move a bundle through
# its lifecycle (NRC-4).

from datetime import datetime
from typing import Annotated, Final, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from app.models.enums import (
    REASON_REQUIRED_COMMANDS,
    BundleLifecycleState,
    Channel,
    CheckCode,
    CheckType,
    Command,
    ExtractionCode,
    Gate,
    PolicyAuditAction,
    PolicyBuildFailure,
    PolicyEdgeType,
    PolicyLanguage,
    PolicyNodeType,
    PostFormat,
    ProposalOrigin,
    ReviewItemKind,
    ReviewState,
    Severity,
    SourceExtractionState,
    StudyWordingSource,
    ValidationInputKind,
    WordingKind,
)
from app.models.policy_graph import (
    NODE_KEY_PATTERN,
    PROHIBITED_PHRASE_MAX_CHARS,
    RULE_ID_PATTERN,
)
from app.schemas.common import ApiModel, RequestModel

# Names, titles, references and texts need a visible character: " " would pass min_length=1.
NOT_BLANK = r"\S"
Text = Annotated[str, StringConstraints(min_length=1, max_length=20_000, pattern=NOT_BLANK)]
Tag = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=NOT_BLANK)]
Note = Annotated[str, StringConstraints(min_length=1, max_length=1000, pattern=NOT_BLANK)]
# A RULE node is created together with its rule (RuleCreate), never on its own.
NonRuleNodeType = Literal[
    PolicyNodeType.CLAUSE,
    PolicyNodeType.DISCLOSURE,
    PolicyNodeType.EXCEPTION,
    PolicyNodeType.INTERPRETATION,
    PolicyNodeType.EXAMPLE,
]


class BundleCreate(RequestModel):
    name: str = Field(min_length=1, max_length=200, pattern=NOT_BLANK)
    source_ref: str = Field(min_length=1, max_length=300, pattern=NOT_BLANK)
    is_synthetic: bool = Field(description="True until the bundle holds the client's real policy.")
    based_on_bundle_id: UUID | None = Field(
        default=None, description="Copy this bundle's graph into the new draft, review included."
    )


class Applicability(RequestModel):
    # Empty lists mean "no restriction", as retrieval reads them. Conditions with an enum use it,
    # as the import script does: a typo is a 422 instead of a rule that never applies.
    input_kinds: list[ValidationInputKind] = Field(default_factory=list, max_length=3)
    channels: list[Channel] = Field(default_factory=list, max_length=20)
    post_formats: list[PostFormat] = Field(default_factory=list, max_length=20)
    languages: list[PolicyLanguage] = Field(default_factory=list, max_length=10)
    audiences: list[Tag] = Field(default_factory=list, max_length=20)
    jurisdictions: list[Tag] = Field(default_factory=list, max_length=20)
    study_ids: list[UUID] = Field(default_factory=list, max_length=100)
    effective_from: AwareDatetime | None = None
    effective_to: AwareDatetime | None = None
    required_study_fact: Tag | None = None

    @model_validator(mode="after")
    def _window_in_order(self) -> Self:
        _check_window(self.effective_from, self.effective_to)
        return self


def _check_window(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError("effective_from must not be after effective_to.")


def _check_not_cleared(model: RequestModel, nullable: frozenset[str]) -> None:
    # null clears an optional field; a required one can only be changed, never emptied.
    cleared = sorted(
        name for name in model.model_fields_set - nullable if getattr(model, name) is None
    )
    if cleared:
        raise ValueError(f"These fields cannot be empty: {', '.join(cleared)}.")


class NodeCreate(Applicability):
    node_key: str = Field(min_length=1, max_length=80, pattern=NODE_KEY_PATTERN)
    node_type: NonRuleNodeType
    text: Text
    language: PolicyLanguage = PolicyLanguage.EN
    source_clause_id: UUID | None = None


class NodeUpdate(RequestModel):
    # Only the fields sent change. Any change sends the node back to PROPOSED for review.
    text: Text | None = None
    source_clause_id: UUID | None = None
    input_kinds: list[ValidationInputKind] | None = Field(default=None, max_length=3)
    channels: list[Channel] | None = Field(default=None, max_length=20)
    post_formats: list[PostFormat] | None = Field(default=None, max_length=20)
    languages: list[PolicyLanguage] | None = Field(default=None, max_length=10)
    audiences: list[Tag] | None = Field(default=None, max_length=20)
    jurisdictions: list[Tag] | None = Field(default=None, max_length=20)
    study_ids: list[UUID] | None = Field(default=None, max_length=100)
    # Aware, as in NodeCreate: a naive time cannot be compared with the stored window.
    effective_from: AwareDatetime | None = None
    effective_to: AwareDatetime | None = None
    required_study_fact: Tag | None = None

    @model_validator(mode="after")
    def _valid_change(self) -> Self:
        _check_not_cleared(self, NULLABLE_NODE_FIELDS)
        # Only when both are sent; one against the stored other is checked by the service.
        _check_window(self.effective_from, self.effective_to)
        return self


NULLABLE_NODE_FIELDS: Final = frozenset(
    {"source_clause_id", "effective_from", "effective_to", "required_study_fact"}
)


class RuleCreate(Applicability):
    rule_id: str = Field(pattern=RULE_ID_PATTERN, examples=["REC-01"])
    rule_version: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200, pattern=NOT_BLANK)
    requirement: str = Field(min_length=1, max_length=1000, pattern=NOT_BLANK)
    gate: Gate
    severity: Severity
    mandatory: bool
    check_type: CheckType
    text: Text = Field(description="The rule as it reads in the policy; searched by retrieval.")
    language: PolicyLanguage = PolicyLanguage.EN


class RuleUpdate(RequestModel):
    # Only the fields sent change. Any change sends the rule's node back to PROPOSED.
    title: str | None = Field(default=None, min_length=1, max_length=200, pattern=NOT_BLANK)
    requirement: str | None = Field(default=None, min_length=1, max_length=1000, pattern=NOT_BLANK)
    gate: Gate | None = None
    severity: Severity | None = None
    mandatory: bool | None = None
    check_type: CheckType | None = None

    @model_validator(mode="after")
    def _nothing_cleared(self) -> Self:
        _check_not_cleared(self, frozenset())
        return self


class EdgeCreate(RequestModel):
    edge_type: PolicyEdgeType
    from_node_id: UUID
    to_node_id: UUID
    provenance_clause_id: UUID | None = None
    provenance_note: Note | None = Field(
        default=None, description="Why the link holds when no source clause states it."
    )

    @model_validator(mode="after")
    def _two_nodes(self) -> Self:
        if self.from_node_id == self.to_node_id:
            raise ValueError("An edge must join two different nodes.")
        return self


class WordingCreate(RequestModel):
    language: PolicyLanguage = PolicyLanguage.EN
    kind: WordingKind
    text: Text = Field(
        description="EXACT_WORDING: required wording quoted word for word from a source clause "
        "(regulatory or platform text, or wording that is the same for every study). "
        "PROHIBITED_PHRASE: one phrase. STUDY_APPROVED_WORDING: where the study's approved "
        "wording comes from, one of "
        + ", ".join(f"`{source.value}`" for source in StudyWordingSource)
        + "; the bundle never holds that wording's text, which comes from the study snapshot."
    )
    source_clause_id: UUID | None = None
    translation_of_id: UUID | None = None

    @model_validator(mode="after")
    def _phrase_fits(self) -> Self:
        if (
            self.kind is WordingKind.PROHIBITED_PHRASE
            and len(self.text) > PROHIBITED_PHRASE_MAX_CHARS
        ):
            raise ValueError(
                f"A prohibited phrase is at most {PROHIBITED_PHRASE_MAX_CHARS} characters."
            )
        return self

    @model_validator(mode="after")
    def _study_wording_is_a_pointer(self) -> Self:
        # A pointer has no text of its own to quote, so it names no clause and translates nothing.
        if self.kind is not WordingKind.STUDY_APPROVED_WORDING:
            return self
        if self.text not in set(StudyWordingSource):
            raise ValueError(
                "text must name where the study's approved wording comes from: "
                + ", ".join(source.value for source in StudyWordingSource)
            )
        if self.source_clause_id is not None or self.translation_of_id is not None:
            raise ValueError(
                "Study-approved wording points at the study snapshot; it cites no source clause "
                "and is not a translation."
            )
        return self


class ReviewDecision(RequestModel):
    decision: Literal[ReviewState.APPROVED, ReviewState.REJECTED]
    note: Note | None = None


class ReviewDecisionRequest(ReviewDecision):
    item_kind: ReviewItemKind
    item_id: UUID


class TransitionRequest(RequestModel):
    reason: Note | None = Field(
        default=None,
        description="Required for reject, activate, rollback and retire; kept in the audit trail.",
    )


class TransitionCommandRequest(TransitionRequest):
    command: Command

    @model_validator(mode="after")
    def _reason_when_required(self) -> Self:
        if self.command in REASON_REQUIRED_COMMANDS and self.reason is None:
            raise ValueError(f"{self.command.value} needs a reason.")
        return self


class PolicySourceCreate(RequestModel):
    upload_id: UUID = Field(description="A READY policy upload (POST /policy-sources/uploads).")
    title: str = Field(min_length=1, max_length=300, pattern=NOT_BLANK)
    source_ref: str = Field(min_length=1, max_length=300, pattern=NOT_BLANK)
    language: PolicyLanguage = PolicyLanguage.EN
    is_synthetic: bool = Field(description="True until the file is the client's real policy.")


# ---- Responses


class PolicySourceRead(ApiModel):
    id: UUID
    title: str
    source_ref: str
    checksum_sha256: str
    language: PolicyLanguage
    is_synthetic: bool
    upload_id: UUID | None
    extraction_state: SourceExtractionState = Field(
        description="Clauses can be cited once this is EXTRACTED."
    )
    extraction_error_code: ExtractionCode | None
    page_count: int | None
    extracted_at: datetime | None
    created_by: str
    created_at: datetime


class PolicySourcePage(ApiModel):
    items: list[PolicySourceRead]
    next_cursor: datetime | None = Field(description="Pass as before for the next (older) page.")


class PolicyClauseRead(ApiModel):
    id: UUID
    clause_ref: str
    section: str | None
    page: int | None
    text: str = Field(description="Verbatim from the source.")
    checksum_sha256: str
    language: PolicyLanguage


class PolicyClauseList(ApiModel):
    items: list[PolicyClauseRead]


class PolicyBundleDetail(ApiModel):
    id: UUID
    version: int
    name: str
    source_ref: str
    is_synthetic: bool
    lifecycle_state: BundleLifecycleState
    revision: int = Field(description='Send as If-Match: "<revision>" when editing or moving it.')
    based_on_bundle_id: UUID | None
    approved_by: str | None
    approved_at: datetime | None
    activated_by: str | None
    activated_at: datetime | None
    last_error_code: PolicyBuildFailure | None = Field(
        description="Why the last index build failed, if it did."
    )
    created_at: datetime
    updated_at: datetime
    rule_count: int


class PolicyBundlePage(ApiModel):
    items: list[PolicyBundleDetail]
    next_cursor: int | None = Field(description="Pass as before_version for the next page.")


class ReviewRead(ApiModel):
    review_state: ReviewState
    origin: ProposalOrigin
    reviewed_by: str | None
    reviewed_at: datetime | None
    review_note: str | None


class PolicyNodeRead(ReviewRead):
    id: UUID
    node_key: str
    node_type: PolicyNodeType
    text: str
    language: PolicyLanguage
    source_clause_id: UUID | None
    # Applicability, as NodeCreate takes it; an empty list means no restriction.
    input_kinds: list[str]
    channels: list[str]
    post_formats: list[str]
    languages: list[str]
    audiences: list[str]
    jurisdictions: list[str]
    study_ids: list[UUID]
    effective_from: datetime | None
    effective_to: datetime | None
    required_study_fact: str | None


class PolicyEdgeRead(ReviewRead):
    id: UUID
    edge_type: PolicyEdgeType
    from_node_id: UUID
    to_node_id: UUID
    provenance_clause_id: UUID | None
    provenance_note: str | None


class PolicyWordingRead(ReviewRead):
    id: UUID
    policy_rule_id: UUID = Field(description="The rule record id (PolicyRuleRead.id).")
    language: PolicyLanguage
    kind: WordingKind
    text: str = Field(description="Exact required wording, or one prohibited phrase.")
    source_clause_id: UUID | None
    translation_of_id: UUID | None
    created_at: datetime


class PolicyRuleRead(ApiModel):
    id: UUID = Field(description="The rule record id, used in paths (not the rule_id code).")
    rule_id: str
    rule_version: int
    title: str
    requirement: str
    gate: Gate
    severity: Severity
    mandatory: bool
    check_type: CheckType
    node: PolicyNodeRead
    wordings: list[PolicyWordingRead]


class PolicyRuleList(ApiModel):
    items: list[PolicyRuleRead]


class PolicyNodeList(ApiModel):
    items: list[PolicyNodeRead]


class PolicyEdgeList(ApiModel):
    items: list[PolicyEdgeRead]


class PolicyIssueRead(ApiModel):
    code: CheckCode
    subject: str = Field(description='What it is about, e.g. "rule:REC-01".')
    message: str


class PolicyBundleChecks(ApiModel):
    issues: list[PolicyIssueRead] = Field(description="Empty when the bundle can be approved.")


class PolicyAuditRead(ApiModel):
    id: UUID
    action: PolicyAuditAction
    from_state: BundleLifecycleState | None
    to_state: BundleLifecycleState | None
    actor: str
    reason: str | None
    details: dict[str, object] = Field(description="Ids and counts only, never policy text.")
    request_id: str | None
    created_at: datetime


class PolicyAuditPage(ApiModel):
    items: list[PolicyAuditRead]
    next_cursor: datetime | None = Field(description="Pass as before for the next (older) page.")
