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

from app.domain.entities import (
    AuditEvent,
    ImageMetadata,
    SatelliteImage,
    TokenClaims,
    User,
    VqaAnswer,
    VqaQuery,
)
from app.domain.enums import FusionType, ImageFormat, ImageStatus, Role, TokenType


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
class ImageRepository(Protocol):
    """Persistence port for :class:`~app.domain.entities.SatelliteImage`."""

    async def add(self, image: SatelliteImage) -> SatelliteImage: ...

    async def get_by_id(self, image_id: UUID) -> SatelliteImage | None: ...

    async def update(self, image: SatelliteImage) -> SatelliteImage: ...

    async def delete(self, image_id: UUID) -> None: ...

    async def list_images(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        owner_id: UUID | None = None,
        image_format: ImageFormat | None = None,
        status: ImageStatus | None = None,
    ) -> tuple[list[SatelliteImage], int]:
        """``owner_id=None`` means "no ownership filter" — the admin, all-images view.
        Owner-scoped listing is the caller's responsibility to enforce by always passing
        the requesting user's id unless they are an admin."""
        ...


@runtime_checkable
class VqaQueryRepository(Protocol):
    """Persistence port for :class:`~app.domain.entities.VqaQuery`."""

    async def add(self, query: VqaQuery) -> VqaQuery: ...

    async def get_by_id(self, query_id: UUID) -> VqaQuery | None: ...

    async def list_for_image(
        self, image_id: UUID, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[VqaQuery], int]: ...

    async def list_recent_by_user(
        self, user_id: UUID | None, *, limit: int
    ) -> tuple[list[VqaQuery], int]:
        """Newest-first queries asked by ``user_id`` (``None`` = everyone's, for admins),
        plus the total count regardless of ``limit`` — feeds the dashboard's usage stats."""
        ...


@runtime_checkable
class ObjectStorage(Protocol):
    """Binary object storage port. Filesystem in the local-lite profile, MinIO/S3 beyond it.

    Deliberately just four methods over raw bytes — no presigned URLs, no multipart
    upload — because nothing in this project yet needs them. Add methods when a real
    caller needs them, not preemptively.
    """

    async def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...


@runtime_checkable
class ImageAnalyzer(Protocol):
    """Raster inspection port — the only place `rasterio`/`Pillow` are allowed to be
    imported outside `infrastructure/imaging/`.

    Split into three steps rather than one "process" call because each has a distinct
    failure mode the service needs to react to differently: a sniff mismatch is a client
    error (422), an inspection failure after a successful sniff means the file is
    corrupt in a way the initial check didn't catch, and thumbnailing is best-effort and
    should not fail the whole upload if it errors.
    """

    async def sniff_format(self, data: bytes, claimed_extension: str) -> ImageFormat:
        """Determine the file's real format from its content and confirm it matches the
        claimed extension. Raises :class:`~app.domain.errors.CorruptImageError` or
        :class:`~app.domain.errors.UnsupportedFileTypeError` on mismatch."""
        ...

    async def inspect(self, data: bytes, image_format: ImageFormat) -> ImageMetadata: ...

    async def generate_thumbnail(
        self, data: bytes, image_format: ImageFormat, *, max_size: int
    ) -> bytes:
        """Return JPEG-encoded thumbnail bytes, longest edge at most ``max_size`` px."""
        ...


@runtime_checkable
class VqaModelPort(Protocol):
    """Inference port over the trained VQA model(s) — the only place ``torch``,
    ``transformers``, and ``timm`` are allowed to be imported outside
    ``infrastructure/vqa/``, mirroring how :class:`ImageAnalyzer` owns ``rasterio``.

    A concrete implementation may hold more than one trained checkpoint in memory at once
    (the elementwise baseline and the cross-attention experiment) — :meth:`available_fusions`
    reports which are actually loaded, since a checkpoint that hasn't been copied down from
    the HPC cluster yet is a legitimate, expected state during development, not an error.
    """

    async def predict(
        self, image_data: bytes, question: str, *, fusion: FusionType, top_k: int = 3
    ) -> VqaAnswer:
        """Raises :class:`~app.domain.errors.ModelUnavailableError` if ``fusion`` isn't
        loaded, and :class:`~app.domain.errors.CorruptImageError` if ``image_data`` can't
        be decoded as an image."""
        ...

    def available_fusions(self) -> frozenset[FusionType]: ...


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
