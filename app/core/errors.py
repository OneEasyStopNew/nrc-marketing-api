# Domain errors. Services raise these; the API layer maps each class to an HTTP status.

from collections.abc import Mapping
from typing import ClassVar


class AppError(Exception):
    # Base class for expected, client-visible failures.

    code: ClassVar[str] = "INTERNAL_ERROR"
    default_message: ClassVar[str] = "Unexpected error."
    retryable: ClassVar[bool] = False

    def __init__(
        self, message: str | None = None, *, details: Mapping[str, object] | None = None
    ) -> None:
        self.message = message or self.default_message
        self.details = dict(details) if details else None
        super().__init__(self.message)


class AuthenticationError(AppError):
    code = "UNAUTHENTICATED"
    default_message = "Authentication is required."


class PermissionDeniedError(AppError):
    code = "FORBIDDEN"
    default_message = "You do not have permission for this action."


class NotFoundError(AppError):
    code = "NOT_FOUND"
    default_message = "Resource not found."


class ContextNotAvailableError(NotFoundError):
    # The run exists but stored no policy context (older runs, or retrieval was unavailable and
    # the run recorded a coverage gap). A separate code so the UI can tell it from NOT_FOUND.
    code = "CONTEXT_NOT_AVAILABLE"
    default_message = "No policy context was stored for this validation run."


class ConflictError(AppError):
    code = "CONFLICT"
    default_message = "The request conflicts with the current state."


class IdempotencyKeyReusedError(ConflictError):
    code = "IDEMPOTENCY_KEY_REUSED"
    default_message = "This Idempotency-Key was already used with a different request."


class StudyClosedError(ConflictError):
    code = "STUDY_CLOSED"
    default_message = "The study is closed, so new creatives cannot be started."


class StudySnapshotMissingError(ConflictError):
    code = "STUDY_SNAPSHOT_MISSING"
    default_message = "The study has no approved snapshot yet."


class GenerationNotEligibleError(ConflictError):
    # details.reason says which check failed (e.g. STUDY_CLOSED).
    code = "GENERATION_NOT_ELIGIBLE"
    default_message = "This creative version cannot be sent for image generation."


class ValidationNotPassedError(ConflictError):
    # details: validation_run_id and status of the latest run for that exact creative version.
    code = "VALIDATION_NOT_PASSED"
    default_message = "This creative version has not passed validation."


class NoActivePolicyError(ConflictError):
    code = "NO_ACTIVE_POLICY"
    default_message = "This tenant has no active policy to validate against."


class PolicyContextUnavailableError(ConflictError):
    # details.reason: BUNDLE_NOT_FOUND, BUNDLE_NOT_USABLE, GRAPH_MISSING, SNAPSHOT_NOT_FOUND, or
    # (with an embedder) INDEX_NOT_READY or INDEX_STALE. Callers record it as a coverage gap; it
    # must never turn into a PASS.
    code = "POLICY_CONTEXT_UNAVAILABLE"
    default_message = "The policy context for this check could not be built."


class PolicyIndexError(ConflictError):
    # details.reason: NO_APPROVED_NODES or INDEX_CONTENT_CHANGED. Raised by the owner-run index
    # build only; the API never builds an index.
    code = "POLICY_INDEX_ERROR"
    default_message = "The policy index could not be built."


class ExportNotReadyError(ConflictError):
    code = "EXPORT_NOT_READY"
    default_message = "Generate an image for the current version before exporting."


class GenerationLimitError(AppError):
    # details.reason is ACTIVE_JOB_LIMIT or CREDIT_CEILING.
    code = "GENERATION_LIMIT_REACHED"
    default_message = "The tenant's generation limit is reached. Try again later."
    retryable = True


class VersionRequiredError(AppError):
    code = "PRECONDITION_REQUIRED"
    default_message = 'Send If-Match with the version you edited, e.g. If-Match: "3".'


class VersionConflictError(AppError):
    code = "VERSION_CONFLICT"
    default_message = "The creative changed since you loaded it. Reload and reapply your edit."


class DependencyUnavailableError(AppError):
    code = "DEPENDENCY_UNAVAILABLE"
    default_message = "A required service is temporarily unavailable."
    retryable = True


class FeatureNotAvailableError(AppError):
    code = "NOT_IMPLEMENTED"
    default_message = "This endpoint is part of the agreed contract but is not implemented yet."
