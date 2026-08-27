"""Domain enumerations.

Pure Python. No framework imports — see the dependency rule in docs/00-architecture.md.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Access roles, ordered from least to most privileged.

    ``GUEST``      — read-only demo access; may query public sample imagery.
    ``RESEARCHER`` — full analysis workflow; owns projects, images and conversations.
    ``ADMIN``      — platform administration, user management, audit inspection.
    """

    GUEST = "guest"
    RESEARCHER = "researcher"
    ADMIN = "admin"

    @property
    def level(self) -> int:
        return _ROLE_LEVELS[self]

    def outranks_or_equals(self, other: Role) -> bool:
        """True when this role satisfies a requirement for ``other``."""
        return self.level >= other.level


_ROLE_LEVELS: dict[Role, int] = {
    Role.GUEST: 0,
    Role.RESEARCHER: 10,
    Role.ADMIN: 100,
}


class TokenType(StrEnum):
    """Discriminator embedded in the JWT ``typ`` claim.

    Without it, a refresh token would be accepted wherever an access token is expected —
    a long-lived credential used as a short-lived one.
    """

    ACCESS = "access"
    REFRESH = "refresh"


class AuditAction(StrEnum):
    """Auditable events. Module 1 defines the authentication subset."""

    USER_REGISTERED = "user.registered"
    LOGIN_SUCCEEDED = "auth.login.succeeded"
    LOGIN_FAILED = "auth.login.failed"
    LOGOUT = "auth.logout"
    TOKEN_REFRESHED = "auth.token.refreshed"
    TOKEN_REUSE_DETECTED = "auth.token.reuse_detected"
    PASSWORD_CHANGED = "user.password.changed"
    ROLE_CHANGED = "user.role.changed"
    USER_DEACTIVATED = "user.deactivated"
    USER_ACTIVATED = "user.activated"
    ACCESS_DENIED = "authz.access_denied"


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
