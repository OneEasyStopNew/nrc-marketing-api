# Policy evidence contracts (NRC-5): a bundle's rules with their sources, and the policy context
# stored for one validation run. Read-only. They show rule ids, node keys and clause references,
# never the internal ids of nodes or edges, search scores or statement text.

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.models.enums import (
    Applicability,
    BundleLifecycleState,
    Channel,
    CheckType,
    Gate,
    MissingContextCode,
    PolicyLanguage,
    PolicyNodeType,
    PostFormat,
    RetrievalMode,
    Severity,
    ValidationInputKind,
)
from app.schemas.common import ApiModel
from app.schemas.validations import TextSpan


class RuleApplicabilityRead(ApiModel):
    """When a rule or exception applies. An empty list means no restriction on that dimension.

    The values are the policy's own tags, so they may name channels or languages the API doesn't
    offer yet; that's why they are strings, not the API's enums.
    """

    input_kinds: list[str]
    channels: list[str]
    post_formats: list[str]
    languages: list[str]
    audiences: list[str]
    jurisdictions: list[str]
    study_specific: bool = Field(description="True when the rule applies to named studies only.")
    effective_from: datetime | None
    effective_to: datetime | None


class RuleSourceRead(ApiModel):
    source_title: str
    clause_ref: str
    section: str | None
    page: int | None
    language: PolicyLanguage
    excerpt: str


class RuleTextRead(ApiModel):
    # A disclosure, interpretation or example linked to the rule.
    node_key: str
    language: PolicyLanguage
    text: str


class RuleExceptionRead(ApiModel):
    node_key: str
    language: PolicyLanguage
    text: str
    requires_study_fact: str | None = Field(
        description="The study fact that must be present for the exception to apply."
    )
    applicability: RuleApplicabilityRead


class BundleRuleRead(ApiModel):
    rule_id: str
    rule_version: int
    title: str
    requirement: str
    gate: Gate
    severity: Severity
    mandatory: bool
    check_type: CheckType
    exact_wording: str | None = Field(description="Required wording, never paraphrased.")
    prohibited_phrases: list[str]
    applicability: RuleApplicabilityRead
    sources: list[RuleSourceRead]
    disclosures: list[RuleTextRead] = Field(description="Wording the post must carry.")
    interpretations: list[RuleTextRead] = Field(
        description="Approved readings of the rule, e.g. which study facts it checks."
    )
    examples: list[RuleTextRead]
    exceptions: list[RuleExceptionRead]


class PolicyBundleRulesRead(ApiModel):
    policy_bundle_id: UUID
    version: int
    name: str
    lifecycle_state: BundleLifecycleState
    is_synthetic: bool = Field(description="True for demo rules, never an approved client policy.")
    rules: list[BundleRuleRead]


class ContextStatementRead(ApiModel):
    # Where in the validated text the statement is; null when it is the whole text.
    span: TextSpan | None


class ContextRuleRead(ApiModel):
    rule_id: str
    rule_version: int
    node_key: str
    mandatory: bool
    check_type: CheckType
    gate: Gate
    severity: Severity
    unresolved: list[str] = Field(
        description="Dimensions the request didn't know; the rule is kept, never dropped."
    )


class ContextMatchRead(ApiModel):
    statement_index: int
    found_by: list[Literal["lexical", "semantic"]]
    rank: int = Field(description="1 is the best match for that statement.")


class ContextEvidenceRead(ApiModel):
    node_key: str
    node_type: PolicyNodeType
    language: PolicyLanguage
    excerpt: str
    clause_ref: str | None
    paths: list[list[str]] = Field(
        description="How each selected rule reaches this node: rule key, then edge and node key."
    )
    matches: list[ContextMatchRead] = Field(description="The statements whose search found it.")


class ContextExceptionRead(ApiModel):
    node_key: str
    for_rule_id: str
    applicable: Applicability
    reasons: list[str]
    excerpt: str
    path: list[str]


class SupersededRuleRead(ApiModel):
    rule_id: str
    superseded_by: str


class MissingContextRead(ApiModel):
    code: MissingContextCode
    rule_id: str | None
    detail: str | None


class ValidationContextRead(ApiModel):
    """The policy context stored for one validation run: which rules it had to use and why.

    Evidence only for now: the run's outcome comes from the rule checks until NRC-7's judge uses
    this context. A code in missing_context means the context is incomplete, never a pass.
    """

    validation_run_id: UUID
    manifest_sha256: str | None = Field(
        description=(
            "sha256 of the stored context's canonical JSON (sorted keys, no whitespace) when the"
            " run used it. Null for runs stored before it was recorded."
        )
    )
    policy_bundle_id: UUID
    policy_bundle_version: int
    is_synthetic: bool
    study_snapshot_id: UUID
    study_snapshot_version: int
    input_kind: ValidationInputKind
    channel: Channel
    post_format: PostFormat | None = Field(
        description="Null when the run's manifest predates recording it."
    )
    language: PolicyLanguage
    audience: str | None
    jurisdiction: str | None
    evaluated_at: datetime
    retrieval_mode: RetrievalMode
    embedding_model_id: str | None
    statements: list[ContextStatementRead]
    required_rules: list[ContextRuleRead]
    optional_rules: list[ContextRuleRead]
    evidence: list[ContextEvidenceRead]
    exceptions: list[ContextExceptionRead]
    superseded: list[SupersededRuleRead]
    missing_context: list[MissingContextRead]
