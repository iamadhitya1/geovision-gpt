"""Ports (interfaces) the application layer depends on.

Declared as ``typing.Protocol`` so implementations need no inheritance and test doubles
need no mocking framework — structural typing is checked statically by mypy.

This file is the concrete expression of the Dependency Inversion Principle in this
codebase: services depend on these abstractions, and ``api/deps.py`` (the composition
root) is the only place that names a concrete class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from app.domain.entities import AuditEvent, TokenClaims, User
from app.domain.enums import Role, TokenType


@runtime_checkable
class UserRepository(Protocol):
    """Persistence port for :class:`~app.domain.entities.User`."""

    async def add(self, user: User) -> User: ...

    async def get_by_id(self, user_id: UUID) -> User | None: ...

    async def get_by_email(self, email: str) -> User | None: ...

    async def update(self, user: User) -> User: ...

    async def list_users(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        role: Role | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[User], int]:
        """Return a page of users and the total count matching the filters."""
        ...

    async def exists_by_email(self, email: str) -> bool: ...


@runtime_checkable
class AuditRepository(Protocol):
    """Append-only persistence port for audit events."""

    async def add(self, event: AuditEvent) -> AuditEvent: ...

    async def list_events(
        self,
        *,
        user_id: UUID | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[AuditEvent], int]: ...


@runtime_checkable
class PasswordHasher(Protocol):
    """Password hashing port.

    ``needs_rehash`` allows transparent upgrades when cost parameters are raised, so the
    hashing algorithm can evolve without forcing a password reset.
    """

    def hash(self, plain_password: str) -> str: ...

    def verify(self, plain_password: str, password_hash: str) -> bool: ...

    def needs_rehash(self, password_hash: str) -> bool: ...

    def dummy_verify(self) -> None:
        """Burn equivalent CPU when no user exists, to keep login timing flat."""
        ...


@runtime_checkable
class TokenService(Protocol):
    """JWT issuing and verification port."""

    def create_access_token(self, user: User) -> tuple[str, int]:
        """Return ``(token, expires_in_seconds)``."""
        ...

    def create_refresh_token(self, user: User) -> tuple[str, str]:
        """Return ``(token, jti)``."""
        ...

    def decode(self, token: str, *, expected_type: TokenType) -> TokenClaims: ...


@runtime_checkable
class Cache(Protocol):
    """Minimal key/value port used for rate limiting and refresh-token tracking."""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...

    async def increment(self, key: str, *, ttl_seconds: int) -> int:
        """Atomically increment ``key``, setting its TTL on first write."""
        ...

    async def ttl(self, key: str) -> int:
        """Remaining TTL in seconds; ``-1`` when the key has none or is absent."""
        ...


@runtime_checkable
class RateLimiter(Protocol):
    """Fixed-window rate limiting port."""

    async def hit(self, key: str, *, limit: int, window_seconds: int) -> None:
        """Register one request; raise ``RateLimitExceededError`` when over budget."""
        ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Transaction boundary port."""

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
