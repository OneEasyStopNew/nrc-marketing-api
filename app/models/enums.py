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
    # Drafts and generation only; publishing Stories comes with NRC-32.
    STORY_IMAGE = "story_image"


class CallToAction(StrEnum):
    # The link button of a post. Fixed values, so they map straight to Meta's CTA types later.
    LEARN_MORE = "learn_more"
    SIGN_UP = "sign_up"
    APPLY_NOW = "apply_now"
    CONTACT_US = "contact_us"


class BriefField(StrEnum):
    # Brief fields beyond the prompt, caption and image text; named in brief_issues and formats.
    HEADLINE = "headline"
    CALL_TO_ACTION = "call_to_action"
    DESTINATION_URL = "destination_url"
    SITE = "site"
    AUDIENCE = "audience"
    VISUAL_INSTRUCTIONS = "visual_instructions"


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


class PolicyIndexReason(StrEnum):
    """Why an index build was refused: details.reason of POLICY_INDEX_ERROR (owner-run build)."""

    NO_APPROVED_NODES = "NO_APPROVED_NODES"
    INDEX_CONTENT_CHANGED = "INDEX_CONTENT_CHANGED"


class PolicyContextReason(StrEnum):
    """Why no policy context was built. Sent as details.reason of POLICY_CONTEXT_UNAVAILABLE (409)
    and in brackets in a validation run's coverage gap; the frontend branches on these values."""

    # Raised by the context provider and the rules endpoint.
    BUNDLE_NOT_FOUND = "BUNDLE_NOT_FOUND"
    BUNDLE_NOT_USABLE = "BUNDLE_NOT_USABLE"
    SNAPSHOT_NOT_FOUND = "SNAPSHOT_NOT_FOUND"
    GRAPH_MISSING = "GRAPH_MISSING"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    INDEX_STALE = "INDEX_STALE"
    MANIFEST_STORE_FAILED = "MANIFEST_STORE_FAILED"
    # Decided by the validation before or around the build.
    LANGUAGE_NOT_SUPPORTED = "LANGUAGE_NOT_SUPPORTED"
    STATEMENT_TOO_LONG = "STATEMENT_TOO_LONG"
    RETRIEVAL_TIMEOUT = "RETRIEVAL_TIMEOUT"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"


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
    # Policy-source documents (PDF, text) are added by NRC-4.
    CREATIVE_MEDIA = "creative_media"


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
    # A post with a link fills all of these, or none. Checked for readiness, never on save,
    # because drafts autosave while the user is still filling them in.
    link_fields: frozenset[BriefField]


FORMAT_SPECS: Final[Mapping[PostFormat, FormatSpec]] = MappingProxyType(
    {
        # The feed shows the headline as the link headline under the image.
        PostFormat.FEED_IMAGE: FormatSpec(
            Channel.FACEBOOK,
            "1:1",
            1080,
            1080,
            frozenset({BriefField.DESTINATION_URL, BriefField.CALL_TO_ACTION, BriefField.HEADLINE}),
        ),
        # A Story link has a button but no headline.
        PostFormat.STORY_IMAGE: FormatSpec(
            Channel.FACEBOOK,
            "9:16",
            1080,
            1920,
            frozenset({BriefField.DESTINATION_URL, BriefField.CALL_TO_ACTION}),
        ),
    }
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
