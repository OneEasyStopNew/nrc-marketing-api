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
