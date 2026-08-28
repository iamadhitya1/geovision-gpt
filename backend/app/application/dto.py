"""Data Transfer Objects — the wire contract.

Separate from domain entities on purpose: ``UserRead`` must never grow a
``password_hash`` field just because ``User`` has one. Keeping the types distinct makes
that class of leak impossible rather than merely discouraged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.domain.entities import AuditEvent, TokenPair, User
from app.domain.enums import AuditAction, AuditOutcome, Role

T = TypeVar("T")

PasswordStr = Annotated[str, Field(min_length=8, max_length=128)]
NameStr = Annotated[str, Field(min_length=1, max_length=120)]


# --- Requests ----------------------------------------------------------------


class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "email": "researcher@geovisiongpt.io",
                "full_name": "Ada Lovelace",
                "password": "Ge0Vision_Strong_2026",
            }
        },
    )

    email: EmailStr
    full_name: NameStr
    password: PasswordStr

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        # Stored lowercase so that lookups and the unique constraint agree.
        return value.lower()


class LoginRequest(BaseModel):
    # The example matches the bootstrap admin created on first startup, so a first-time
    # reader can click "Try it out" -> "Execute" with zero edits and see a real result.
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "email": "admin@geovisiongpt.io",
                "password": "ChangeMe_Admin_123",
            }
        },
    )

    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.lower()


class RefreshRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"refresh_token": "<paste the refresh_token from your login response>"}
        }
    )

    refresh_token: str = Field(min_length=1)


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_password": "ChangeMe_Admin_123",
                "new_password": "An0ther_Strong_2026",
            }
        }
    )

    current_password: str = Field(min_length=1)
    new_password: PasswordStr


class UpdateProfileRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"full_name": "Ada Lovelace"}},
    )

    full_name: NameStr | None = None


class AdminUpdateUserRequest(BaseModel):
    """Admin-only mutations, kept apart from self-service updates so that role
    escalation can never ride in on a profile edit."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"role": "admin", "is_active": True, "is_verified": True}}
    )

    role: Role | None = None
    is_active: bool | None = None
    is_verified: bool | None = None


# --- Responses ---------------------------------------------------------------


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
            }
        }
    )

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

    @classmethod
    def from_domain(cls, pair: TokenPair) -> TokenResponse:
        return cls(
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_in=pair.expires_in,
        )


class UserRead(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "6bdc72d1-dc8a-4dc0-a972-082f42c25c76",
                "email": "researcher@geovisiongpt.io",
                "full_name": "Ada Lovelace",
                "role": "researcher",
                "is_active": True,
                "is_verified": False,
                "created_at": "2026-08-27T09:15:13.280499Z",
                "last_login_at": "2026-08-27T09:19:12.817502Z",
            }
        }
    )

    id: UUID
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None = None

    @classmethod
    def from_domain(cls, user: User) -> UserRead:
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )


class AuthenticatedUserResponse(BaseModel):
    """Returned by registration and login: identity plus credentials in one round trip."""

    user: UserRead
    tokens: TokenResponse


class AuditEventRead(BaseModel):
    id: UUID
    user_id: UUID | None
    action: AuditAction
    outcome: AuditOutcome
    ip_address: str | None
    user_agent: str | None
    detail: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_domain(cls, event: AuditEvent) -> AuditEventRead:
        return cls(
            id=event.id,
            user_id=event.user_id,
            action=event.action,
            outcome=event.outcome,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            detail=event.detail,
            created_at=event.created_at,
        )


class Page(BaseModel, Generic[T]):
    """Uniform pagination envelope reused by every list endpoint in the project."""

    items: list[T]
    total: int
    offset: int
    limit: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    """The single error shape emitted by the API.

    Every error the API returns — validation, auth, not-found, rate limit, anything —
    has exactly this shape, so a client only ever needs one error-handling code path.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "invalid_credentials",
                "message": "Incorrect email or password.",
                "detail": {},
                "request_id": "4833200cf6764c68b9198cb2f674427e",
            }
        }
    )

    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


# --- Internal command objects ------------------------------------------------


class RequestContext(BaseModel):
    """Ambient caller information threaded into services for auditing.

    Passing this explicitly keeps services free of any dependency on the web request
    object, which is what allows them to be unit-tested without FastAPI.
    """

    ip_address: str | None = None
    user_agent: str | None = None
