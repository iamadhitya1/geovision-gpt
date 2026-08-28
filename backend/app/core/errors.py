"""Translation of domain errors into HTTP responses.

This module is the *only* place in the codebase that knows a status code corresponds to a
business failure. Adding a domain error means adding one line here; no service, repository
or router ever raises ``HTTPException``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.application.dto import ErrorResponse
from app.core.logging import get_logger, request_id_ctx
from app.domain.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DomainError,
    FileTooLargeError,
    NotFoundError,
    RateLimitExceededError,
    ValidationError,
)

logger = get_logger(__name__)

# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY to ..._CONTENT and
# HTTP_413_REQUEST_ENTITY_TOO_LARGE to ..._CONTENT_TOO_LARGE, deprecating the old names.
# Referencing the numbers directly keeps this module working across both versions without
# a deprecation warning on every error response.
HTTP_413_TOO_LARGE = 413
HTTP_422_UNPROCESSABLE = 422

# Ordered most-specific first; the first isinstance match wins. FileTooLargeError must
# precede ValidationError (its own parent) or the more specific 413 would never be reached.
_STATUS_MAP: tuple[tuple[type[DomainError], int], ...] = (
    (RateLimitExceededError, status.HTTP_429_TOO_MANY_REQUESTS),
    (AuthenticationError, status.HTTP_401_UNAUTHORIZED),
    (AuthorizationError, status.HTTP_403_FORBIDDEN),
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (ConflictError, status.HTTP_409_CONFLICT),
    (FileTooLargeError, HTTP_413_TOO_LARGE),
    (ValidationError, HTTP_422_UNPROCESSABLE),
)


def status_for(error: DomainError) -> int:
    for error_type, code in _STATUS_MAP:
        if isinstance(error, error_type):
            return code
    return status.HTTP_400_BAD_REQUEST


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the project's error handlers to ``app``."""

    @app.exception_handler(DomainError)
    async def _handle_domain_error(_request: Request, exc: DomainError) -> JSONResponse:
        http_status = status_for(exc)
        body = ErrorResponse(
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
            request_id=request_id_ctx.get(),
        )

        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitExceededError):
            headers["Retry-After"] = str(exc.retry_after_seconds)
        if isinstance(exc, AuthenticationError):
            headers["WWW-Authenticate"] = "Bearer"

        if http_status >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.error("domain_error", code=exc.code, message=exc.message)
        else:
            logger.info("domain_error", code=exc.code, status=http_status)

        return JSONResponse(
            status_code=http_status,
            content=body.model_dump(mode="json"),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Normalise FastAPI's validation errors into the project's error envelope."""
        body = ErrorResponse(
            code="validation_error",
            message="Request validation failed.",
            detail={"errors": exc.errors()},
            request_id=request_id_ctx.get(),
        )
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        """Last resort. The exception text is logged but never returned — stack traces
        and driver messages leak schema and file-system details to the caller."""
        logger.exception("unhandled_exception", error=str(exc))
        body = ErrorResponse(
            code="internal_error",
            message="An unexpected error occurred.",
            request_id=request_id_ctx.get(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(mode="json"),
        )
