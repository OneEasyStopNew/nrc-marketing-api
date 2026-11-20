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


class PolicyLanguage(StrEnum):
    # Policy text may be Spanish before the public Language enum accepts it (F33).
    EN = "en"
    ES = "es"


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


class Gate(StrEnum):
    COMPLETENESS = "completeness"
    STUDY_FACTS = "study_facts"
    POLICY = "policy"
    MEDIA = "media"


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class BundleLifecycleState(StrEnum):
    """Where a policy bundle is in its NRC-4 lifecycle. Only ACTIVE bundles validate content."""

    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    BUILDING = "BUILDING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class PolicyNodeType(StrEnum):
    RULE = "RULE"
    CLAUSE = "CLAUSE"
    DISCLOSURE = "DISCLOSURE"
    EXCEPTION = "EXCEPTION"
    INTERPRETATION = "INTERPRETATION"
    EXAMPLE = "EXAMPLE"


class ReviewState(StrEnum):
    """Review decision on one node, edge or wording. APPROVED rows are the ones retrieval reads."""

    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ProposalOrigin(StrEnum):
    """Who produced a graph row. LLM_SIMULATED is a labelled stand-in, never a real model."""

    HUMAN = "HUMAN"
    IMPORT = "IMPORT"
    LLM_SIMULATED = "LLM_SIMULATED"
    LLM = "LLM"


class PolicyAuditAction(StrEnum):
    CREATE = "CREATE"
    EDIT = "EDIT"
    SUBMIT = "SUBMIT"
    REVIEW = "REVIEW"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    RETURN_TO_DRAFT = "RETURN_TO_DRAFT"
    BUILD_STARTED = "BUILD_STARTED"
    BUILD_READY = "BUILD_READY"
    BUILD_FAILED = "BUILD_FAILED"
    ACTIVATE = "ACTIVATE"
    ROLLBACK = "ROLLBACK"
    RETIRE = "RETIRE"


class SourceExtractionState(StrEnum):
    """Where a policy source is in text extraction; one that arrives with clauses is EXTRACTED."""

    PENDING = "PENDING"
    EXTRACTING = "EXTRACTING"
    EXTRACTED = "EXTRACTED"
    FAILED = "FAILED"


class WordingKind(StrEnum):
    EXACT_WORDING = "EXACT_WORDING"
    PROHIBITED_PHRASE = "PROHIBITED_PHRASE"


class PolicyEdgeType(StrEnum):
    SUPPORTED_BY = "supported_by"
    REQUIRES_DISCLOSURE = "requires_disclosure"
    EXCEPT_WHEN = "except_when"
    SUPERSEDES = "supersedes"
    APPLIES_TO = "applies_to"


# Policy context retrieval (NRC-5): stored in manifests and shown by the evidence API.


class RetrievalMode(StrEnum):
    # No statements: rule selection and the graph walk only.
    GRAPH_ONLY = "graph_only"
    # Statements searched with full-text only: no embedder (e.g. inside an API request, where
    # provider calls aren't allowed).
    LEXICAL_ONLY = "lexical_only"
    # Full-text and vector search, fused.
    FULL = "full"


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
    # Statements were searched with full-text only (no embedder).
    SEMANTIC_SEARCH_SKIPPED = "SEMANTIC_SEARCH_SKIPPED"
    # More statements than POLICY_RETRIEVAL_MAX_STATEMENTS; only the first ones were searched.
    STATEMENTS_TRUNCATED = "STATEMENTS_TRUNCATED"


class CheckType(StrEnum):
    """What makes the final decision for a rule: code alone, or the model after a phrase check."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class IndexBuildState(StrEnum):
    BUILDING = "BUILDING"
    READY = "READY"
    FAILED = "FAILED"


class JobStage(StrEnum):
    """Where a job is inside its execution state. Shown in the UI, so values are part of the API."""

    SUBMITTING = "submitting"
    RETRY_WAIT = "retry_wait"
    RECONCILE = "reconcile"
    WAITING_PROVIDER = "waiting_provider"
    # Provider finished but the worker has no storage configured: the job is parked at this result.
    OUTPUT_READY = "output_ready"
    # The provider reported the generation completed (so billed) and the worker is fetching it.
    FETCHING_OUTPUT = "fetching_output"
    STORED = "stored"


class UploadState(StrEnum):
    """Where a client upload is. Only READY uploads become assets that later steps may use."""

    # Upload URL issued; the client has not completed the upload yet.
    PENDING_UPLOAD = "PENDING_UPLOAD"
    # Upload completed and checked against what was declared; waiting for inspection.
    QUARANTINED = "QUARANTINED"
    SCANNING = "SCANNING"
    READY = "READY"
    REJECTED = "REJECTED"
    # The URL was never used; set by the cleanup job.
    EXPIRED = "EXPIRED"


class UploadPurpose(StrEnum):
    CREATIVE_MEDIA = "creative_media"
    # A policy-source document (PDF or text) uploaded for NRC-4 ingestion; it has no study.
    POLICY_SOURCE = "policy_source"


class IdentifiablePerson(StrEnum):
    # The uploader's declaration. It is not proof of consent; the authorization reference is.
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


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
