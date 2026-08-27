"""Domain entities — framework-free business objects.

Deliberately plain dataclasses rather than SQLAlchemy models or Pydantic schemas. The
persistence model (``infrastructure/db/models.py``) and the wire model
(``application/dto.py``) are separate types that map onto these, so a column rename or an
API field change cannot silently alter business rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.domain.enums import AuditAction, AuditOutcome, Role


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class User:
    """A platform user.

    ``token_version`` is the global revocation lever: every refresh token embeds the
    version current at issue time, so incrementing it invalidates all outstanding
    sessions for this user in one write, without a token table to sweep.
    """

    email: str
    full_name: str
    password_hash: str
    role: Role = Role.RESEARCHER
    id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    is_verified: bool = False
    token_version: int = 0
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    last_login_at: datetime | None = None

    # --- Business rules ------------------------------------------------------

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def can_login(self) -> bool:
        return self.is_active

    def has_role(self, required: Role) -> bool:
        """Hierarchical check: an admin satisfies a researcher requirement."""
        return self.role.outranks_or_equals(required)

    def record_login(self) -> None:
        self.last_login_at = _now()
        self.updated_at = self.last_login_at

    def revoke_all_sessions(self) -> None:
        """Invalidate every outstanding refresh token for this user."""
        self.token_version += 1
        self.updated_at = _now()

    def change_password(self, new_hash: str) -> None:
        """A password change must also end existing sessions, or a stolen refresh token
        survives the very rotation intended to defeat it."""
        self.password_hash = new_hash
        self.revoke_all_sessions()

    def deactivate(self) -> None:
        self.is_active = False
        self.revoke_all_sessions()

    def activate(self) -> None:
        self.is_active = True
        self.updated_at = _now()

    def change_role(self, new_role: Role) -> None:
        self.role = new_role
        self.updated_at = _now()


@dataclass(frozen=True, slots=True)
class TokenPair:
    """Issued credential pair returned to the client."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0  # seconds until the access token expires


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """Decoded, validated JWT payload."""

    subject: UUID
    role: Role
    token_type: str
    jti: str
    issued_at: datetime
    expires_at: datetime
    token_version: int = 0


@dataclass(slots=True)
class AuditEvent:
    """An immutable record of a security-relevant action.

    Written on both success and failure — a log that only records successes cannot
    evidence an attack.
    """

    action: AuditAction
    outcome: AuditOutcome
    id: UUID = field(default_factory=uuid4)
    user_id: UUID | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
