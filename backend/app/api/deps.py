"""Composition root.

This is the single module permitted to name concrete implementations. Everything above it
receives abstractions, which is what makes the swaps promised in the architecture doc
(SQLite ↔ PostgreSQL, memory ↔ Redis, stub ↔ real VLM) configuration changes rather than
code changes.

Object lifetimes:

* **Process-scoped** (built once in ``main.lifespan``, held on ``app.state``): settings,
  engine, session factory, cache, password hasher, token service, password policy.
  The Argon2 hasher in particular must not be rebuilt per request — constructing it
  computes the dummy hash, which is deliberately expensive.
* **Request-scoped** (built per request here): session, repositories, unit of work,
  services.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.audit_service import AuditService
from app.application.auth_service import AuthService
from app.application.dto import RequestContext
from app.application.image_service import ImageService
from app.application.password_policy import PasswordPolicy
from app.application.user_service import UserService
from app.core.config import Settings
from app.core.logging import user_id_ctx
from app.domain.entities import User
from app.domain.enums import Role, TokenType
from app.domain.errors import AuthorizationError, InactiveUserError, InvalidTokenError
from app.domain.interfaces import (
    AuditRepository,
    Cache,
    ImageAnalyzer,
    ImageRepository,
    ObjectStorage,
    PasswordHasher,
    RateLimiter,
    TokenService,
    UnitOfWork,
    UserRepository,
)
from app.infrastructure.cache.rate_limiter import FixedWindowRateLimiter, NullRateLimiter
from app.infrastructure.db.repositories import (
    SqlAlchemyAuditRepository,
    SqlAlchemyImageRepository,
    SqlAlchemyUserRepository,
)
from app.infrastructure.db.session import SqlAlchemyUnitOfWork

bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


# --- Process-scoped singletons ----------------------------------------------


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_cache(request: Request) -> Cache:
    return request.app.state.cache  # type: ignore[no-any-return]


def get_password_hasher(request: Request) -> PasswordHasher:
    return request.app.state.password_hasher  # type: ignore[no-any-return]


def get_token_service(request: Request) -> TokenService:
    return request.app.state.token_service  # type: ignore[no-any-return]


def get_password_policy(request: Request) -> PasswordPolicy:
    return request.app.state.password_policy  # type: ignore[no-any-return]


def get_storage(request: Request) -> ObjectStorage:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_image_analyzer(request: Request) -> ImageAnalyzer:
    return request.app.state.image_analyzer  # type: ignore[no-any-return]


def get_rate_limiter(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings_dep)],
    cache: Annotated[Cache, Depends(get_cache)],
) -> RateLimiter:
    if not settings.rate_limit_enabled:
        return NullRateLimiter()
    return FixedWindowRateLimiter(cache)


# --- Request-scoped -----------------------------------------------------------


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a session that is rolled back if the request raises.

    Services commit explicitly at the end of each use case. The rollback here catches the
    case where an exception escapes mid-use-case, so nothing partially applied survives.
    """
    factory = request.app.state.session_factory
    session: AsyncSession = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_user_repository(session: SessionDep) -> UserRepository:
    return SqlAlchemyUserRepository(session)


def get_audit_repository(session: SessionDep) -> AuditRepository:
    return SqlAlchemyAuditRepository(session)


def get_image_repository(session: SessionDep) -> ImageRepository:
    return SqlAlchemyImageRepository(session)


def get_unit_of_work(session: SessionDep) -> UnitOfWork:
    return SqlAlchemyUnitOfWork(session)


def get_audit_service(
    repository: Annotated[AuditRepository, Depends(get_audit_repository)],
) -> AuditService:
    return AuditService(repository)


def get_auth_service(
    users: Annotated[UserRepository, Depends(get_user_repository)],
    hasher: Annotated[PasswordHasher, Depends(get_password_hasher)],
    tokens: Annotated[TokenService, Depends(get_token_service)],
    cache: Annotated[Cache, Depends(get_cache)],
    policy: Annotated[PasswordPolicy, Depends(get_password_policy)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> AuthService:
    return AuthService(
        users=users,
        hasher=hasher,
        tokens=tokens,
        cache=cache,
        policy=policy,
        audit=audit,
        uow=uow,
        refresh_ttl_seconds=settings.refresh_token_expire_days * 86_400,
    )


def get_user_service(
    users: Annotated[UserRepository, Depends(get_user_repository)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
) -> UserService:
    return UserService(users=users, audit=audit, uow=uow)


def get_image_service(
    images: Annotated[ImageRepository, Depends(get_image_repository)],
    storage: Annotated[ObjectStorage, Depends(get_storage)],
    analyzer: Annotated[ImageAnalyzer, Depends(get_image_analyzer)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> ImageService:
    return ImageService(
        images=images,
        storage=storage,
        analyzer=analyzer,
        audit=audit,
        uow=uow,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        thumbnail_max_size_px=settings.thumbnail_max_size_px,
    )


# --- Caller context -----------------------------------------------------------


def client_ip(request: Request) -> str:
    """Best-effort client address.

    ``X-Forwarded-For`` is trusted only because the deployment terminates at NGINX, which
    overwrites it. Behind an untrusted proxy this header is attacker-controlled and would
    let a caller sidestep per-IP rate limits by rotating the value.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_request_context(request: Request) -> RequestContext:
    return RequestContext(
        ip_address=client_ip(request),
        user_agent=request.headers.get("User-Agent", "")[:512] or None,
    )


# --- Authentication / authorisation ------------------------------------------


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    tokens: Annotated[TokenService, Depends(get_token_service)],
    users: Annotated[UserRepository, Depends(get_user_repository)],
) -> User:
    """Resolve the caller from a bearer access token.

    The user row is loaded on every request rather than trusting the token's claims alone.
    That costs one indexed primary-key lookup and buys immediate effect for deactivation
    and session revocation, instead of waiting out the access token's lifetime.
    """
    if credentials is None or not credentials.credentials:
        raise InvalidTokenError("Missing bearer token.")

    claims = tokens.decode(credentials.credentials, expected_type=TokenType.ACCESS)

    user = await users.get_by_id(claims.subject)
    if user is None:
        raise InvalidTokenError("The token subject no longer exists.")
    if claims.token_version != user.token_version:
        raise InvalidTokenError("This session has been revoked.")
    if not user.can_login():
        raise InactiveUserError()

    user_id_ctx.set(str(user.id))
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(
    minimum: Role,
) -> Callable[[User], Coroutine[Any, Any, User]]:
    """Build a dependency enforcing a minimum role.

    The check is hierarchical (:meth:`Role.outranks_or_equals`), so an admin passes a
    researcher requirement without every route having to enumerate roles.
    """

    async def _dependency(user: CurrentUser) -> User:
        if not user.has_role(minimum):
            raise AuthorizationError(
                f"This action requires the '{minimum.value}' role or higher.",
                required_role=minimum.value,
                actual_role=user.role.value,
            )
        return user

    return _dependency


require_researcher = require_role(Role.RESEARCHER)
require_admin = require_role(Role.ADMIN)

ResearcherUser = Annotated[User, Depends(require_researcher)]
AdminUser = Annotated[User, Depends(require_admin)]
