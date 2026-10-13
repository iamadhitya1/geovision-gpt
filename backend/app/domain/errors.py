"""Domain-level exceptions.

These carry no HTTP semantics. ``app/core/errors.py`` owns the single mapping from these
types to status codes, so the domain never learns that HTTP exists and the same services
could be driven by a CLI or a message consumer.
"""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """Base class for all expected, business-rule failures."""

    code: str = "domain_error"
    message: str = "A domain error occurred."

    def __init__(self, message: str | None = None, **detail: Any) -> None:
        self.message = message or self.message
        self.detail: dict[str, Any] = detail
        super().__init__(self.message)


# --- Authentication / authorisation -----------------------------------------


class AuthenticationError(DomainError):
    code = "authentication_failed"
    message = "Authentication failed."


class InvalidCredentialsError(AuthenticationError):
    code = "invalid_credentials"
    # Deliberately identical whether the email is unknown or the password is wrong:
    # distinguishing them turns the login endpoint into a user-enumeration oracle.
    message = "Incorrect email or password."


class InactiveUserError(AuthenticationError):
    code = "inactive_user"
    message = "This account has been deactivated."


class InvalidTokenError(AuthenticationError):
    code = "invalid_token"
    message = "The provided token is invalid or has expired."


class TokenReuseError(AuthenticationError):
    code = "token_reuse_detected"
    message = "Refresh token reuse detected; the session has been revoked."


class AuthorizationError(DomainError):
    code = "forbidden"
    message = "You do not have permission to perform this action."


# --- Resource / validation ---------------------------------------------------


class NotFoundError(DomainError):
    code = "not_found"
    message = "The requested resource was not found."


class ConflictError(DomainError):
    code = "conflict"
    message = "The resource already exists."


class EmailAlreadyRegisteredError(ConflictError):
    code = "email_already_registered"
    message = "An account with this email already exists."


class ValidationError(DomainError):
    code = "validation_error"
    message = "The submitted data is invalid."


class WeakPasswordError(ValidationError):
    code = "weak_password"
    message = "The password does not meet the strength policy."


# --- Image management --------------------------------------------------------


class UnsupportedFileTypeError(ValidationError):
    code = "unsupported_file_type"
    message = "This file type is not supported. Upload a GeoTIFF, PNG, or JPEG."


class FileTooLargeError(ValidationError):
    code = "file_too_large"
    message = "The uploaded file exceeds the maximum allowed size."


class CorruptImageError(ValidationError):
    code = "corrupt_image"
    message = (
        "The file's contents don't match a valid image of the claimed type — it may be "
        "corrupt, empty, or mislabeled."
    )


class ImageNotReadyError(ValidationError):
    code = "image_not_ready"
    message = (
        "This image hasn't finished processing yet, or processing failed. "
        "Only images with status 'ready' can be queried."
    )


class NothingToHighlightError(ValidationError):
    code = "nothing_to_highlight"
    message = (
        "This question doesn't name one thing that can be highlighted. Highlighting works "
        "for vegetation, water, buildings and roads."
    )


class UnsupportedQuestionError(ValidationError):
    code = "unsupported_question"
    message = "That isn't a question this model can answer."


class ModelUnavailableError(DomainError):
    code = "model_unavailable"
    message = (
        "The requested model variant isn't loaded on this server. This is expected if its "
        "trained checkpoint hasn't been deployed yet."
    )


class ImageryUnavailableError(DomainError):
    code = "imagery_unavailable"
    message = (
        "Couldn't fetch satellite imagery for that location right now. The imagery service "
        "may be unreachable, or the place may be outside its coverage."
    )


class RateLimitExceededError(DomainError):
    code = "rate_limit_exceeded"
    message = "Too many requests. Please try again later."

    def __init__(self, retry_after_seconds: int, message: str | None = None) -> None:
        super().__init__(message, retry_after_seconds=retry_after_seconds)
        self.retry_after_seconds = retry_after_seconds
