# Maps exceptions to the shared error body: {code, message, request_id, details, retryable}.

import logging
from collections.abc import Mapping
from http import HTTPStatus
from types import MappingProxyType
from typing import Any, Final, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ExceptionHandler

from app.core.errors import (
    AppError,
    AuthenticationError,
    ConflictError,
    DependencyUnavailableError,
    FeatureNotAvailableError,
    GenerationLimitError,
    NotFoundError,
    PermissionDeniedError,
    VersionConflictError,
    VersionRequiredError,
)
from app.core.request_context import request_id_var
from app.schemas.common import ErrorBody

logger = logging.getLogger(__name__)

STATUS_BY_ERROR: Final[Mapping[type, int]] = MappingProxyType(
    {
        AuthenticationError: 401,
        PermissionDeniedError: 403,
        NotFoundError: 404,
        ConflictError: 409,
        VersionConflictError: 412,
        VersionRequiredError: 428,
        GenerationLimitError: 429,
        FeatureNotAvailableError: 501,
        DependencyUnavailableError: 503,
    }
)
_HTTP_CODES: Final[Mapping[int, str]] = MappingProxyType(
    {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 413: "PAYLOAD_TOO_LARGE"}
)

# Documented on every API route so the generated client knows the error shape.
COMMON_ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    401: {"model": ErrorBody, "description": "Missing, invalid or expired access token."},
    403: {"model": ErrorBody, "description": "Not a member of the tenant, or role not allowed."},
    422: {"model": ErrorBody, "description": "Invalid fields; see details.fields."},
}


def _status_for(error: AppError) -> int:
    for cls in type(error).__mro__:
        status = STATUS_BY_ERROR.get(cls)
        if status is not None:
            return status
    return 500


def _error_response(
    status: int,
    code: str,
    message: str,
    *,
    details: dict[str, object] | None = None,
    retryable: bool = False,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ErrorBody(
        code=code,
        message=message,
        request_id=request_id_var.get(),
        details=details,
        retryable=retryable,
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=status, headers=headers)


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    status = _status_for(exc)
    headers = {"WWW-Authenticate": "Bearer"} if status == HTTPStatus.UNAUTHORIZED else None
    return _error_response(
        status, exc.code, exc.message, details=exc.details, retryable=exc.retryable, headers=headers
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Rejected input values are not echoed back: prompts may contain sensitive text.
    fields = [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return _error_response(
        422, "VALIDATION_FAILED", "The request has invalid fields.", details={"fields": fields}
    )


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error_response(
        exc.status_code,
        _HTTP_CODES.get(exc.status_code, "HTTP_ERROR"),
        HTTPStatus(exc.status_code).phrase,
        headers=exc.headers,
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.scope.get("path"))
    return _error_response(500, "INTERNAL_ERROR", "Unexpected error.")


def register_exception_handlers(app: FastAPI) -> None:
    # cast: Starlette types handlers as taking any Exception, but calls each one only with
    # instances of the class it is registered for.
    app.add_exception_handler(AppError, cast(ExceptionHandler, handle_app_error))
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, handle_validation_error)
    )
    app.add_exception_handler(StarletteHTTPException, cast(ExceptionHandler, handle_http_exception))
    app.add_exception_handler(Exception, handle_unexpected_error)
