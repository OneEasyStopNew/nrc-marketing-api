# The context manifest (NRC-5): what one retrieval decided, stored as JSON and validated here.

from enum import StrEnum
from typing import Final, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from app.models.enums import (
    Channel,
    CheckType,
    Gate,
    PolicyLanguage,
    PolicyNodeType,
    Severity,
    ValidationInputKind,
)

# Bumped whenever the selection or walk logic changes, so cached manifests built by older logic
# are never reused (the version is part of every cache key).
RETRIEVAL_CONFIG_VERSION: Final = 1
MANIFEST_SCHEMA_VERSION: Final = 1
# Evidence quotes node text; long clauses are cut so a manifest stays a bounded size.
EXCERPT_MAX_CHARS: Final = 1000


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Applicability(StrEnum):
    TRUE = "true"
    FALSE = "false"
    # A condition restricts a dimension the request doesn't know (e.g. no audience yet).
    UNRESOLVED = "unresolved"


class MissingContextCode(StrEnum):
    """Why a manifest is incomplete. A caller must never treat any of these as a PASS."""

    APPLICABILITY_UNRESOLVED = "APPLICABILITY_UNRESOLVED"
    TRAVERSAL_TRUNCATED = "TRAVERSAL_TRUNCATED"
    CONFLICT = "CONFLICT"
    SOURCE_MISSING = "SOURCE_MISSING"


class ContextScope(_Frozen):
    input_kind: ValidationInputKind
    channel: Channel
    # Policy language (en or es), wider than the public Language enum until Spanish ships (F33).
    language: PolicyLanguage
    # Not on creatives yet: None means unknown, so rules restricted by it are flagged, not dropped.
    audience: str | None = Field(default=None, max_length=64)
    jurisdiction: str | None = Field(default=None, max_length=64)
    # Effective dates are judged at this moment. A reused manifest keeps the moment it was first
    # built (the cache key holds which dated nodes were in effect, so the result is the same);
    # callers record their own time, e.g. validation_runs.evaluated_at.
    evaluated_at: AwareDatetime


class ContextRequest(_Frozen):
    # The caller pins the bundle (the one its run records), never "whatever is active now".
    policy_bundle_id: UUID
    study_snapshot_id: UUID
    scope: ContextScope


class BundleRef(_Frozen):
    id: UUID
    version: int
    is_synthetic: bool


class SnapshotRef(_Frozen):
    id: UUID
    version: int


class ManifestRule(_Frozen):
    rule_id: str
    rule_version: int
    node_key: str
    mandatory: bool
    check_type: CheckType
    gate: Gate
    severity: Severity
    # Selected by the stored conditions only, never by a search score.
    selected_by: Literal["conditions"] = "conditions"
    # Dimensions the request didn't know; the rule is kept and also listed in missing_context.
    unresolved: tuple[str, ...] = ()


class EvidenceItem(_Frozen):
    node_key: str
    node_type: PolicyNodeType
    language: PolicyLanguage
    excerpt: str
    # The source clause the node quotes, if any.
    clause_ref: str | None
    found_by: Literal["graph"] = "graph"
    # One path per selected rule that reaches this node: rule node key, then edge label and node
    # key per hop. "applied_by" is an applies_to edge followed backwards (interpretation -> rule).
    paths: tuple[tuple[str, ...], ...]


class ExceptionItem(_Frozen):
    node_key: str
    for_rule_id: str
    applicable: Applicability
    # Why it does not apply (e.g. STUDY_FACT_MISSING:compensation_statement), or which
    # dimensions are unknown.
    reasons: tuple[str, ...] = ()
    excerpt: str
    path: tuple[str, ...]


class SupersededRule(_Frozen):
    rule_id: str
    superseded_by: str


class MissingContext(_Frozen):
    code: MissingContextCode
    rule_id: str | None = None
    detail: str | None = None


class RetrievalConfig(_Frozen):
    version: int
    max_hops: int
    max_nodes: int


class ContextManifest(_Frozen):
    """Everything a check must use for one creative version, and what retrieval couldn't settle.

    Exceptions never remove a rule from required_rules: a judge may only treat a rule as not
    applicable by citing an exception listed here with applicable = "true".
    """

    schema_version: int = MANIFEST_SCHEMA_VERSION
    policy_bundle: BundleRef
    study_snapshot: SnapshotRef
    scope: ContextScope
    required_rules: tuple[ManifestRule, ...]
    optional_rules: tuple[ManifestRule, ...]
    evidence: tuple[EvidenceItem, ...]
    exceptions: tuple[ExceptionItem, ...]
    superseded: tuple[SupersededRule, ...]
    missing_context: tuple[MissingContext, ...]
    retrieval_config: RetrievalConfig
